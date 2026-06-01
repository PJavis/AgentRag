# src/agentrag/agent/starters.py
"""Empty-state starter questions.

When a chat session has no messages yet, the UI shows starter suggestions.
Two come from fixed client-side chips; up to 3 are generated here by one
cheap LLM call (task name 'starter', routable via LLM_TASK_MODEL_MAP)
from the document/notebook title(s) + existing summary/insight text.
Cached 10 min by (kind, titles, summary) hash. Failure → [] (never breaks).
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from cachetools import TTLCache

logger = logging.getLogger(__name__)

_CACHE: TTLCache[str, list[str]] = TTLCache(maxsize=512, ttl=600)

_SYS = (
    "You are a Vietnamese-language study tutor. Given a document or notebook "
    "TITLE(S) and a short SUMMARY, propose up to 3 short starter questions a "
    "student could click to begin exploring the material. Each question must be "
    "specific to the content, ≤ 80 characters, and end with '?'. "
    "Return strict JSON: {\"starters\": [\"...\", \"...\", \"...\"]}. Vietnamese only."
)


def _cache_key(kind: str, titles: list[str], summary_text: str) -> str:
    blob = kind + "\n" + "|".join(titles) + "\n###\n" + summary_text[:1500]
    return hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()


async def generate_starters(
    kind: str,
    titles: list[str],
    summary_text: str,
    llm_gateway: Any,
) -> list[str]:
    titles = [t for t in titles if t and t.strip()]
    if not titles:
        return []
    key = _cache_key(kind, titles, summary_text or "")
    cached = _CACHE.get(key)
    if cached is not None:
        return list(cached)

    user_payload = json.dumps(
        {"kind": kind, "titles": titles[:10], "summary": (summary_text or "")[:1500]},
        ensure_ascii=False,
    )
    try:
        payload, _latency = await llm_gateway.json_response(
            system_prompt=_SYS,
            user_prompt=user_payload,
            task="starter",
        )
    except Exception:
        logger.warning("generate_starters: LLM call failed", exc_info=True)
        _CACHE[key] = []
        return []

    raw = payload.get("starters") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        _CACHE[key] = []
        return []
    cleaned: list[str] = []
    for item in raw[:3]:
        if isinstance(item, str):
            s = item.strip()
            if s:
                cleaned.append(s[:120])
    _CACHE[key] = cleaned
    return list(cleaned)
