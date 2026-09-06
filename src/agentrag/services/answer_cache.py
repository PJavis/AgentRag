"""Exact-match answer cache, keyed on the corpus version.

Deliberately NOT semantic. A similarity-keyed answer cache over a medical corpus
serves the wrong dose to the wrong patient class: at a 0.97 cosine threshold
"liều dùng cho người lớn" and "liều dùng cho trẻ em" are neighbours. Exact match
has no such failure mode, and on repeated demo/eval traffic it still hits.

It also does not try to be clever about conversation. Only stateless first turns
are cacheable — `graph_service.chat` rewrites a verbose follow-up using the
previous user message, so two identical question strings can legitimately need
different answers.

Fails closed everywhere: unknown corpus version, disabled flag, unreachable
valkey, or an unparseable payload all mean "no cache", never "cache anyway".
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from src.agentrag.common.corpus_version import get_corpus_version
from src.agentrag.config import settings

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_PREFIX = "agentrag:answer_cache:"
_CLIENT = None


class AnswerCache:
    def _client(self):
        global _CLIENT
        if _CLIENT is None:
            if not settings.REDIS_URL:
                return None
            try:
                from redis import Redis

                _CLIENT = Redis.from_url(settings.REDIS_URL, decode_responses=True)
            except Exception as exc:  # noqa: BLE001 — valkey optional
                logger.debug("answer_cache: redis init failed (%s)", exc)
                return None
        return _CLIENT

    @staticmethod
    def _normalize(question: str) -> str:
        return _WS.sub(" ", (question or "").strip().lower())

    def key(
        self,
        question: str,
        corpus_version: str | None,
        document_title: str | None,
        domain_filter: dict[str, Any] | None,
        verbosity: str | None,
        model: str,
    ) -> str | None:
        """Cache key, or None when caching must not happen at all.

        No corpus version → None. A cached answer that outlives its corpus cites
        segments that no longer exist.
        """
        if not corpus_version:
            return None
        payload = json.dumps(
            {
                "q": self._normalize(question),
                "corpus": corpus_version,
                "doc": document_title or "",
                # sort_keys below so an equivalent filter written in another
                # order produces the same key.
                "filter": domain_filter or {},
                "verbosity": verbosity or "",
                "model": model or "",
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return _PREFIX + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str | None) -> dict | None:
        if not key or not settings.ANSWER_CACHE_ENABLED:
            return None
        client = self._client()
        if client is None:
            return None
        try:
            raw = client.get(key)
        except Exception as exc:  # noqa: BLE001 — valkey unreachable
            logger.debug("answer_cache: get failed (%s)", exc)
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            # A corrupt entry is a miss, never an exception on the answer path.
            return None

    def put(self, key: str | None, payload: dict) -> None:
        if not key or not settings.ANSWER_CACHE_ENABLED:
            return
        client = self._client()
        if client is None:
            return
        try:
            client.setex(
                key,
                settings.ANSWER_CACHE_TTL_SECONDS,
                json.dumps(payload, ensure_ascii=False, default=str),
            )
        except Exception as exc:  # noqa: BLE001 — caching must never break answering
            logger.debug("answer_cache: put failed (%s)", exc)


def cacheable_turn(chat_history: list[dict] | None) -> bool:
    """Only stateless first turns may be cached.

    `graph_service.chat` rewrites a verbose follow-up using the previous user
    message, so two identical question strings can legitimately require
    different answers. Caching those would serve the wrong one.
    """
    return not chat_history


def cacheable_result(result: dict | None) -> bool:
    """False for anything that must not be pinned for the whole TTL.

    `graph_service.chat` has a graceful-timeout path that returns "Hệ thống
    đang bận, vui lòng thử lại sau giây lát." with no citations. Caching that
    would serve a transient load message as the answer to a real question for
    a full day. An empty answer is refused for the same reason.
    """
    if not result or result.get("timed_out"):
        return False
    return bool((result.get("answer") or "").strip())


def current_corpus_version() -> str | None:
    """Indirection so callers need not import corpus_version directly."""
    return get_corpus_version()
