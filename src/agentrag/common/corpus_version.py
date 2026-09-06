"""Shared marker for "which corpus generation is currently indexed".

Anything that caches an answer derived from the corpus must key on this, so a
re-ingest cannot leave a cached answer pointing at segments that no longer
exist. Returns None when the value is unknown — callers MUST fail closed on
None rather than caching against a guess.
"""
from __future__ import annotations

import logging
import time
import uuid

from src.agentrag.config import settings

logger = logging.getLogger(__name__)

_KEY = "agentrag:corpus_version"
_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        if not settings.REDIS_URL:
            return None
        try:
            from redis import Redis

            _CLIENT = Redis.from_url(settings.REDIS_URL, decode_responses=True)
        except Exception as exc:  # noqa: BLE001 — valkey optional
            logger.debug("corpus_version: redis init failed (%s)", exc)
            return None
    return _CLIENT


def get_corpus_version() -> str | None:
    client = _client()
    if client is None:
        return None
    try:
        return client.get(_KEY) or None
    except Exception as exc:  # noqa: BLE001 — valkey unreachable
        logger.debug("corpus_version: get failed (%s)", exc)
        return None


def bump_corpus_version() -> str | None:
    """Mark a new corpus generation. Called when an ingest run completes."""
    client = _client()
    if client is None:
        return None
    version = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    try:
        client.set(_KEY, version)
        return version
    except Exception as exc:  # noqa: BLE001 — valkey unreachable
        logger.debug("corpus_version: set failed (%s)", exc)
        return None
