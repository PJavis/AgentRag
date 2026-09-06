"""LLM cost ledger backed by a Valkey stream.

Every successful LLM call in the API + worker pushes one entry through
:func:`record_llm_call`. Entries land in the Valkey/Redis stream
``agentrag:llm:calls:v1`` (capped via ``MAXLEN ~ 5000`` so it auto-trims).
Survives process restart, aggregates across multi-worker.

Falls back to a process-local deque when Valkey is unreachable (or when
``REDIS_URL`` is not configured) so tests and offline dev still work. The
fallback is per-process and will not aggregate across workers; if you see
the dashboard reporting partial data, check Valkey connectivity.

USD estimates use public Gemini / OpenAI pricing. When the provider returns
a ``usage`` object we prefer those token counts; otherwise we fall back to a
char-density heuristic so the dashboard still has signal.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import threading
import time
from collections import deque
from typing import Any

from redis.exceptions import RedisError

from src.agentrag.config import settings

logger = logging.getLogger(__name__)

# ── Stream config ────────────────────────────────────────────────────────────
_STREAM_KEY = "agentrag:llm:calls:v1"
_STREAM_MAXLEN = 5000

# ── Sync/async ledger state ──────────────────────────────────────────────────
_LOCK = threading.Lock()
_LEDGER: deque[dict[str, Any]] = deque(maxlen=_STREAM_MAXLEN)  # fallback + tests
_ID_SEQ = itertools.count(1)

# ── Valkey clients (lazily created; one per loop is fine for our load) ───────
_ASYNC_CLIENT = None
_SYNC_CLIENT = None
_VALKEY_DISABLED = False  # latched true after first failure

_PRICE_PER_1M = {
    "gemini-2.5-pro":          (1.25, 10.00),
    "gemini-2.5-flash":        (0.075, 0.30),
    "gemini-2.5-flash-lite":   (0.05, 0.20),
    "gemini-2.0-flash":        (0.10, 0.40),
    "gemini-1.5-pro":          (1.25, 5.00),
    "gemini-1.5-flash":        (0.075, 0.30),
    "gpt-4o":                  (2.50, 10.00),
    "gpt-4o-mini":             (0.15, 0.60),
    # DeepSeek: (uncached input, output). Cached input is priced separately in
    # _CACHE_HIT_PRICE_PER_1M. VERIFY against current published pricing — these
    # are a local estimate. Only the token COUNTS recorded below are provider facts.
    "deepseek-chat":           (0.27, 1.10),
    "deepseek-v4-flash":       (0.27, 1.10),
    "deepseek-v4-pro":         (0.55, 2.19),
}

#: Input price for tokens the provider served from its own prefix cache. Roughly
#: an order of magnitude below the uncached rate, which is why a workload with a
#: large stable prefix (ingest contextualisation) costs far less than its raw
#: token count suggests.
_CACHE_HIT_PRICE_PER_1M = {
    "deepseek-chat":           0.027,
    "deepseek-v4-flash":       0.027,
    "deepseek-v4-pro":         0.055,
}


def _cache_hit_price_for(model: str) -> float:
    """Price for cached input tokens; falls back to the uncached input price."""
    for key, price in _CACHE_HIT_PRICE_PER_1M.items():
        if model == key or key in model:
            return price
    return _price_for(model)[0]


def _estimate_tokens(text: str) -> int:
    """Char-density estimate. ~4 chars/tok ASCII, ~1.8 for VN/CJK."""
    if not text:
        return 0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ascii_chars = len(text) - non_ascii
    return int(ascii_chars / 4 + non_ascii * 0.55) + 1


def _price_for(model: str) -> tuple[float, float]:
    if not model:
        return _PRICE_PER_1M["gemini-2.5-flash"]
    if model in _PRICE_PER_1M:
        return _PRICE_PER_1M[model]
    for k, v in _PRICE_PER_1M.items():
        if model.startswith(k):
            return v
    return _PRICE_PER_1M["gemini-2.5-flash"]


def _sync_client():
    """Sync Redis/Valkey client for blocking contexts (e.g. tests)."""
    global _SYNC_CLIENT, _VALKEY_DISABLED
    if _VALKEY_DISABLED or not settings.REDIS_URL:
        return None
    if _SYNC_CLIENT is None:
        try:
            from redis import Redis  # sync client
            _SYNC_CLIENT = Redis.from_url(settings.REDIS_URL, decode_responses=True)
        except Exception as exc:  # pragma: no cover — import-only guard
            logger.debug("cost: sync redis init failed (%s); using fallback", exc)
            _VALKEY_DISABLED = True
            return None
    return _SYNC_CLIENT


def _async_client():
    global _ASYNC_CLIENT, _VALKEY_DISABLED
    if _VALKEY_DISABLED or not settings.REDIS_URL:
        return None
    if _ASYNC_CLIENT is None:
        try:
            from redis.asyncio import Redis as AsyncRedis
            _ASYNC_CLIENT = AsyncRedis.from_url(settings.REDIS_URL, decode_responses=True)
        except Exception as exc:  # pragma: no cover
            logger.debug("cost: async redis init failed (%s); using fallback", exc)
            _VALKEY_DISABLED = True
            return None
    return _ASYNC_CLIENT


def _coerce_entry(fields: dict[str, str], entry_id: str | None = None) -> dict[str, Any]:
    """Stream entries store strings; coerce back to typed dict."""
    return {
        "id": int(fields.get("id", 0)) if fields.get("id", "").isdigit() else fields.get("id", entry_id),
        "timestamp": float(fields.get("timestamp", 0.0) or 0.0),
        "task": fields.get("task", ""),
        "model": fields.get("model", ""),
        "latency_ms": float(fields.get("latency_ms", 0.0) or 0.0),
        "in_tokens": int(fields.get("in_tokens", 0) or 0),
        "out_tokens": int(fields.get("out_tokens", 0) or 0),
        "usd": float(fields.get("usd", 0.0) or 0.0),
        "usage_source": fields.get("usage_source", "estimate"),
        "cache_hit_tokens": int(fields.get("cache_hit_tokens", 0) or 0),
        "cache_miss_tokens": int(fields.get("cache_miss_tokens", 0) or 0),
    }


def _stream_xadd_sync(entry: dict[str, Any]) -> bool:
    """Push entry to Valkey stream synchronously. Returns False on failure."""
    global _VALKEY_DISABLED
    client = _sync_client()
    if client is None:
        return False
    try:
        fields = {k: str(v) for k, v in entry.items()}
        client.xadd(_STREAM_KEY, fields, maxlen=_STREAM_MAXLEN, approximate=True)
        return True
    except RedisError as exc:
        logger.warning("cost: stream XADD failed (%s); disabling Valkey ledger", exc)
        _VALKEY_DISABLED = True
        return False


async def _stream_xadd_async(entry: dict[str, Any]) -> bool:
    global _VALKEY_DISABLED
    client = _async_client()
    if client is None:
        return False
    try:
        fields = {k: str(v) for k, v in entry.items()}
        await client.xadd(_STREAM_KEY, fields, maxlen=_STREAM_MAXLEN, approximate=True)
        return True
    except RedisError as exc:
        logger.warning("cost: async XADD failed (%s); disabling Valkey ledger", exc)
        _VALKEY_DISABLED = True
        return False


def record_llm_call(
    *,
    task: str,
    model: str,
    latency_ms: float,
    in_text: str = "",
    out_text: str = "",
    usage: Any = None,
) -> None:
    """Append one call to the Valkey-backed ledger. No-op when tracking off.

    Always writes to the in-process deque so introspection still works under
    the fallback path. When Valkey is reachable, also writes the entry to the
    stream (XADD).
    """
    if not settings.LLM_COST_TRACKING_ENABLED:
        return

    in_tokens: int | None = None
    out_tokens: int | None = None
    hit_tokens = 0
    miss_tokens: int | None = None
    if usage is not None:
        in_tokens = getattr(usage, "prompt_tokens", None)
        out_tokens = getattr(usage, "completion_tokens", None)
        # DeepSeek reports these; most providers do not. Absent → all misses.
        hit_tokens = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
        raw_miss = getattr(usage, "prompt_cache_miss_tokens", None)
        miss_tokens = int(raw_miss) if raw_miss is not None else None
    if in_tokens is None:
        in_tokens = _estimate_tokens(in_text)
    if out_tokens is None:
        out_tokens = _estimate_tokens(out_text)
    if miss_tokens is None:
        miss_tokens = max(int(in_tokens) - hit_tokens, 0)

    in_price, out_price = _price_for(model)
    hit_price = _cache_hit_price_for(model)
    usd = (
        miss_tokens * in_price + hit_tokens * hit_price + out_tokens * out_price
    ) / 1_000_000.0

    with _LOCK:
        entry_id = next(_ID_SEQ)
    entry = {
        "id": entry_id,
        "timestamp": time.time(),
        "task": task,
        "model": model,
        "latency_ms": round(latency_ms, 2),
        "in_tokens": int(in_tokens),
        "out_tokens": int(out_tokens),
        "usd": usd,
        "usage_source": "provider" if usage is not None else "estimate",
        "cache_hit_tokens": int(hit_tokens),
        "cache_miss_tokens": int(miss_tokens),
    }
    with _LOCK:
        _LEDGER.append(entry)

    # Best-effort Valkey persist — try async if we're in a loop, else sync.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        loop.create_task(_stream_xadd_async(entry))
    else:
        _stream_xadd_sync(entry)


def _read_entries() -> list[dict[str, Any]]:
    """Read full ledger (Valkey stream if available, else deque)."""
    client = _sync_client()
    if client is not None:
        try:
            raw = client.xrange(_STREAM_KEY, min="-", max="+", count=_STREAM_MAXLEN)
            return [_coerce_entry(fields, entry_id) for entry_id, fields in raw]
        except RedisError as exc:
            logger.warning("cost: XRANGE failed (%s); falling back to in-process ledger", exc)
    with _LOCK:
        return list(_LEDGER)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def recent(n: int = 20) -> list[dict[str, Any]]:
    """Last n ledger entries, oldest first. Test/debug introspection."""
    return _read_entries()[-n:]


def reset_ledger() -> None:
    """Clear the in-process ledger. Tests only — never called by the app."""
    with _LOCK:
        _LEDGER.clear()


def cost_summary() -> dict[str, Any]:
    entries = _read_entries()
    per_task: dict[str, dict[str, Any]] = {}
    per_model: dict[str, dict[str, Any]] = {}
    per_task_lat: dict[str, list[float]] = {}
    per_model_lat: dict[str, list[float]] = {}
    total_in = total_out = 0
    total_usd = 0.0
    for e in entries:
        for bucket, lat_bucket, key in (
            (per_task, per_task_lat, e["task"]),
            (per_model, per_model_lat, e["model"]),
        ):
            s = bucket.setdefault(key, {
                "calls": 0, "in_tokens": 0, "out_tokens": 0,
                "total_latency_ms": 0.0, "usd": 0.0,
            })
            s["calls"] += 1
            s["in_tokens"] += e["in_tokens"]
            s["out_tokens"] += e["out_tokens"]
            s["total_latency_ms"] += e["latency_ms"]
            s["usd"] += e["usd"]
            lat_bucket.setdefault(key, []).append(float(e["latency_ms"]))
        total_in += e["in_tokens"]
        total_out += e["out_tokens"]
        total_usd += e["usd"]
    for bucket, lat_bucket in ((per_task, per_task_lat), (per_model, per_model_lat)):
        for k, s in bucket.items():
            s["avg_latency_ms"] = round(s["total_latency_ms"] / s["calls"], 1) if s["calls"] else 0.0
            s["total_latency_ms"] = round(s["total_latency_ms"], 1)
            s["usd"] = round(s["usd"], 6)
            samples = lat_bucket.get(k, [])
            s["p50_latency_ms"] = round(_percentile(samples, 50), 1)
            s["p95_latency_ms"] = round(_percentile(samples, 95), 1)
    backend = "valkey" if (_sync_client() is not None and not _VALKEY_DISABLED) else "in-process"
    # Provider-side prefix cache accounting. These are counts the provider
    # reported, not estimates — a low hit rate on a workload with a large stable
    # prefix means the prompt is shaped wrong, not that caching is unavailable.
    cache_hit = sum(e.get("cache_hit_tokens", 0) for e in entries)
    cache_miss = sum(e.get("cache_miss_tokens", 0) for e in entries)
    prompt_total = cache_hit + cache_miss
    return {
        "total_calls": len(entries),
        "total_in_tokens": total_in,
        "total_out_tokens": total_out,
        "total_usd": round(total_usd, 6),
        "cache_hit_tokens": cache_hit,
        "cache_miss_tokens": cache_miss,
        "cache_hit_rate": (cache_hit / prompt_total) if prompt_total else None,
        "per_task": per_task,
        "per_model": per_model,
        "backend": backend,
        "note": f"USD = char-density estimate OR provider usage when available; pricing per 1M tokens; backend={backend}, MAXLEN~{_STREAM_MAXLEN}.",
    }


def recent_calls(limit: int = 50, since: float | None = None) -> list[dict[str, Any]]:
    """Most-recent calls, newest first. Optional `since` filters by epoch seconds."""
    limit = max(1, min(limit, 500))
    entries = _read_entries()
    if since is not None:
        entries = [e for e in entries if e.get("timestamp", 0) >= since]
    entries.reverse()
    return [
        {
            **e,
            "usd": round(e["usd"], 6),
            "latency_ms": round(e["latency_ms"], 1),
        }
        for e in entries[:limit]
    ]


def reset_ledger() -> None:
    """Clear both Valkey stream and in-process fallback."""
    with _LOCK:
        _LEDGER.clear()
    client = _sync_client()
    if client is not None:
        try:
            client.delete(_STREAM_KEY)
        except RedisError as exc:
            logger.warning("cost: reset DELETE failed (%s)", exc)
