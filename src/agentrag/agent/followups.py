"""B2 — Follow-up question generator.

After agent.chat returns an answer, fire one cheap LLM call (task name
'followup' so users can route to a small model via LLM_TASK_MODEL_MAP)
to propose 3 short Vietnamese follow-up questions. Cached 5 min by
(question, answer) hash. Failure → empty list (never breaks the chat).
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from cachetools import TTLCache

logger = logging.getLogger(__name__)

_CACHE: TTLCache[str, list[str]] = TTLCache(maxsize=512, ttl=300)

_SYS = (
    "You are a helpful Vietnamese-language study tutor. Given a QUESTION "
    "and the ANSWER given to the student, propose up to 3 short follow-up "
    "questions the student might ask next. Each ≤ 80 characters. "
    "Return strict JSON: {\"follow_ups\": [\"...\", \"...\", \"...\"]}. "
    "Vietnamese only."
)


def _cache_key(question: str, answer: str) -> str:
    return hashlib.sha256(
        (question + "\n###\n" + answer).encode("utf-8", errors="replace")
    ).hexdigest()


async def generate_followups(
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    llm_gateway: Any,
) -> list[str]:
    if not question or not answer:
        return []
    key = _cache_key(question, answer)
    cached = _CACHE.get(key)
    if cached is not None:
        return list(cached)

    user_payload = json.dumps(
        {
            "question": question,
            "answer": answer[:1500],
            "doc_titles": list({(c.get("document_title") or "")[:80] for c in citations}),
        },
        ensure_ascii=False,
    )
    try:
        payload, _latency = await llm_gateway.json_response(
            system_prompt=_SYS,
            user_prompt=user_payload,
            task="followup",
        )
    except Exception:
        logger.warning("generate_followups: LLM call failed", exc_info=True)
        _CACHE[key] = []
        return []

    raw = payload.get("follow_ups") if isinstance(payload, dict) else None
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
