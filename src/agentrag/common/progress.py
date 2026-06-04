"""Ingest progress pub/sub over Valkey (Redis-compatible).

Workers publish a tiny `{source_id, stage}` event per status transition to a
per-user channel; the `/sources/progress/stream` SSE endpoint relays it so the
UI refreshes the source list live instead of waiting for the next poll. Best
effort — a publish failure never breaks ingestion.
"""
from __future__ import annotations

import json
import logging

from src.agentrag.config import settings

logger = logging.getLogger(__name__)

_CHANNEL_PREFIX = "ingest:progress:"


def channel_for(user_id: str | None) -> str:
    return f"{_CHANNEL_PREFIX}{user_id or 'anonymous'}"


async def publish_progress(user_id: str | None, source_id: str, stage: str) -> None:
    """Publish a stage transition for one document. Never raises."""
    if not settings.REDIS_URL:
        return
    try:
        from redis.asyncio import Redis

        client = Redis.from_url(settings.REDIS_URL, decode_responses=True)
        try:
            await client.publish(
                channel_for(user_id),
                json.dumps({"source_id": str(source_id), "stage": stage}),
            )
        finally:
            await client.aclose()
    except Exception as exc:  # noqa: BLE001 — best effort
        logger.debug("publish_progress failed: %s", exc)
