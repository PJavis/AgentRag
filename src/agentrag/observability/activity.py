"""Activity event log (S6).

record_event() is sync-inline: caller awaits, failure logs + swallows so the
business path never breaks. Cheap INSERT (~1ms) per chat turn / upload /
search.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import EventLog

logger = logging.getLogger(__name__)


def _coerce_uuid(v: uuid.UUID | str | None) -> uuid.UUID | None:
    if v is None or v == "anonymous":
        return None
    if isinstance(v, uuid.UUID):
        return v
    try:
        return uuid.UUID(str(v))
    except (ValueError, TypeError):
        return None


async def record_event(
    user_id: uuid.UUID | str | None,
    event_type: str,
    *,
    target_kind: str | None = None,
    target_id: uuid.UUID | str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Insert a single event_log row. Best-effort — never raises."""
    try:
        row = EventLog(
            user_id=_coerce_uuid(user_id),
            event_type=event_type,
            target_kind=target_kind,
            target_id=_coerce_uuid(target_id),
            payload=payload or {},
        )
        async with AsyncSessionLocal() as session:
            session.add(row)
            await session.commit()
    except Exception as exc:
        logger.warning("activity.record_event failed: %s (%s)", exc, event_type)
