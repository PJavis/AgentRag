"""Shared chat-turn plumbing for every chat entry point.

There are four places that run a chat turn: the root `/chat` and `/chat/stream`
routes in main.py, and the adapter's notebook (`/chat/execute`,
`/chat/execute/stream`, `/regenerate`) and source-chat routes. They had drifted:

- only the root routes enqueued the `chat_memory` job, so with
  CHAT_STRUCTMEM_ENABLED the UI (which talks only to the adapter) never wrote
  chat memory at all;
- the history window was CHAT_HISTORY_WINDOW on the root routes, a hard-coded
  20 on notebook chat and 10 on source chat;
- the follow-up persistence block was copy-pasted twice.

Each helper here is the one place that behaviour lives. They are best-effort
where a failure must not cost the user an answer they already received.
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from src.agentrag.config import settings

logger = logging.getLogger(__name__)


async def load_history(store: Any, conversation_id: str) -> list[dict]:
    """The prior turns the agent sees: the last CHAT_HISTORY_WINDOW messages."""
    return await store.list_messages(
        conversation_id, limit=settings.CHAT_HISTORY_WINDOW
    )


async def record_turn_memory(
    conversation_id: str,
    question: str,
    answer: str,
    *,
    failed: bool = False,
) -> bool:
    """Enqueue the `chat_memory` job for a finished turn.

    Returns True only when a job was actually enqueued. A failed turn (the
    agent errored and `answer` is an error banner) is not memory, and neither is
    an empty answer. Enqueue errors are logged, never raised: the answer has
    already been produced and persisted by the time this runs.
    """
    if not settings.CHAT_STRUCTMEM_ENABLED or failed:
        return False
    if not (answer or "").strip() or not (question or "").strip():
        return False
    try:
        from src.agentrag.worker.pool import get_pool

        await get_pool().enqueue_job(
            "chat_memory",
            conversation_id=conversation_id,
            user_message=question,
            assistant_message=answer,
            turn_id=str(uuid.uuid4()),
            turn_timestamp=datetime.now(UTC).isoformat(),
        )
        return True
    except Exception:
        logger.exception("record_turn_memory: enqueue chat_memory failed for %s", conversation_id)
        return False


async def attach_follow_ups(
    store: Any,
    conversation_id: str,
    appended: dict | None,
    follow_ups: list[str],
) -> bool:
    """Persist follow-up chips onto an already-appended assistant message.

    Returns True when the row was updated. The message cache is dropped so the
    next list_messages sees the new metadata instead of a stale copy.
    """
    if not follow_ups or not isinstance(appended, dict):
        return False
    message_id = appended.get("message_id")
    if not message_id:
        return False
    try:
        from src.agentrag.database import AsyncSessionLocal
        from src.agentrag.database.models import ChatMessage

        async with AsyncSessionLocal() as session:
            row = await session.get(ChatMessage, uuid.UUID(str(message_id)))
            if row is None:
                return False
            meta = dict(row.extra_metadata or {})
            meta["follow_ups"] = follow_ups
            row.extra_metadata = meta
            await session.commit()
    except Exception:
        logger.exception("attach_follow_ups: persist failed for %s", conversation_id)
        return False
    try:
        await store._delete_messages_cache(conversation_id)
    except Exception:
        pass
    return True
