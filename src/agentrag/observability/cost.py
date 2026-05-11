"""Process-global LLM cost ledger.

Every successful LLM call in the API + worker processes pushes one entry
through :func:`record_llm_call`. The ledger is in-memory only (cleared on
process restart) — fine for development visibility. For long-lived multi-
process tracking, swap the deque for a Redis/SQL backend.

USD estimates use public Gemini / OpenAI pricing tables. When the provider
returns a `usage` object we prefer those token counts; otherwise we fall
back to a char-density heuristic so the dashboard still has signal.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any

from src.agentrag.config import settings

_LOCK = threading.Lock()
_LEDGER: deque[dict[str, Any]] = deque(maxlen=5000)  # keep last 5k calls

# USD per 1M tokens (input, output). Adjust when provider pricing changes.
_PRICE_PER_1M = {
    "gemini-2.5-pro":          (1.25, 10.00),
    "gemini-2.5-flash":        (0.075, 0.30),
    "gemini-2.5-flash-lite":   (0.05, 0.20),
    "gemini-2.0-flash":        (0.10, 0.40),
    "gemini-1.5-pro":          (1.25, 5.00),
    "gemini-1.5-flash":        (0.075, 0.30),
    "gpt-4o":                  (2.50, 10.00),
    "gpt-4o-mini":             (0.15, 0.60),
}


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


def record_llm_call(
    *,
    task: str,
    model: str,
    latency_ms: float,
    in_text: str = "",
    out_text: str = "",
    usage: Any = None,
) -> None:
    """Append one call to the in-memory ledger. No-op when cost tracking off."""
    if not settings.LLM_COST_TRACKING_ENABLED:
        return

    in_tokens: int | None = None
    out_tokens: int | None = None
    if usage is not None:
        # OpenAI-compat clients return CompletionUsage with these attrs.
        in_tokens = getattr(usage, "prompt_tokens", None)
        out_tokens = getattr(usage, "completion_tokens", None)
    if in_tokens is None:
        in_tokens = _estimate_tokens(in_text)
    if out_tokens is None:
        out_tokens = _estimate_tokens(out_text)

    in_price, out_price = _price_for(model)
    usd = (in_tokens * in_price + out_tokens * out_price) / 1_000_000.0

    entry = {
        "task": task,
        "model": model,
        "latency_ms": round(latency_ms, 2),
        "in_tokens": int(in_tokens),
        "out_tokens": int(out_tokens),
        "usd": usd,
        "usage_source": "provider" if usage is not None else "estimate",
    }
    with _LOCK:
        _LEDGER.append(entry)


def cost_summary() -> dict[str, Any]:
    with _LOCK:
        entries = list(_LEDGER)

    per_task: dict[str, dict[str, Any]] = {}
    per_model: dict[str, dict[str, Any]] = {}
    total_in = total_out = 0
    total_usd = 0.0
    for e in entries:
        for bucket, key in ((per_task, e["task"]), (per_model, e["model"])):
            s = bucket.setdefault(key, {
                "calls": 0, "in_tokens": 0, "out_tokens": 0,
                "total_latency_ms": 0.0, "usd": 0.0,
            })
            s["calls"] += 1
            s["in_tokens"] += e["in_tokens"]
            s["out_tokens"] += e["out_tokens"]
            s["total_latency_ms"] += e["latency_ms"]
            s["usd"] += e["usd"]
        total_in += e["in_tokens"]
        total_out += e["out_tokens"]
        total_usd += e["usd"]
    for bucket in (per_task, per_model):
        for s in bucket.values():
            s["avg_latency_ms"] = round(s["total_latency_ms"] / s["calls"], 1) if s["calls"] else 0.0
            s["total_latency_ms"] = round(s["total_latency_ms"], 1)
            s["usd"] = round(s["usd"], 6)
    return {
        "total_calls": len(entries),
        "total_in_tokens": total_in,
        "total_out_tokens": total_out,
        "total_usd": round(total_usd, 6),
        "per_task": per_task,
        "per_model": per_model,
        "note": "USD = char-density estimate OR provider usage when available; pricing per 1M tokens; in-memory, cleared on restart.",
    }


def reset_ledger() -> None:
    with _LOCK:
        _LEDGER.clear()
