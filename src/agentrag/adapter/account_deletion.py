"""Full user-data wipe (P4 right-to-delete). Postgres is authoritative; the ES and
image-file purges are best-effort so a search-layer hiccup can't half-delete PG."""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from sqlalchemy import delete, select

from src.agentrag.adapter.db import AdapterChatFeedback
from src.agentrag.config import settings
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import (
    ChatMessage,
    Conversation,
    Document,
    EventLog,
    Segment,
    User,
)

logger = logging.getLogger(__name__)


async def delete_user_data(user_id: str) -> dict[str, int]:
    """Erase ALL data owned by user_id across Postgres (authoritative), then best-effort
    ES + image-file purge. Only ever touches rows matching this user_id."""
    counts = {"documents": 0, "segments": 0, "conversations": 0,
              "messages": 0, "feedback": 0, "events": 0}
    titles: list[str] = []

    async with AsyncSessionLocal() as s:
        docs = (await s.execute(select(Document).where(Document.user_id == user_id))).scalars().all()
        for doc in docs:
            seg = await s.execute(delete(Segment).where(Segment.document_id == doc.id))
            counts["segments"] += seg.rowcount or 0
            if doc.title:
                titles.append(doc.title)
            await s.delete(doc)
            counts["documents"] += 1

        conv_ids = (
            await s.execute(select(Conversation.id).where(Conversation.user_id == user_id))
        ).scalars().all()
        if conv_ids:
            m = await s.execute(delete(ChatMessage).where(ChatMessage.conversation_id.in_(conv_ids)))
            counts["messages"] += m.rowcount or 0
        c = await s.execute(delete(Conversation).where(Conversation.user_id == user_id))
        counts["conversations"] += c.rowcount or 0

        f = await s.execute(delete(AdapterChatFeedback).where(AdapterChatFeedback.user_id == str(user_id)))
        counts["feedback"] += f.rowcount or 0
        e = await s.execute(delete(EventLog).where(EventLog.user_id == user_id))
        counts["events"] += e.rowcount or 0

        await s.execute(delete(User).where(User.id == user_id))
        await s.commit()

    # Best-effort search/file purge — Postgres is already committed above.
    for title in titles:
        try:
            from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore

            await ElasticsearchStore().delete_document(title)
        except Exception as exc:
            logger.debug("ES purge skipped for %r: %s", title, exc)
        try:
            safe = re.sub(r"[^\w\-]", "_", title)[:80]
            shutil.rmtree(Path(settings.IMAGE_STORAGE_DIR) / safe, ignore_errors=True)
        except Exception as exc:
            logger.debug("image purge skipped for %r: %s", title, exc)

    logger.info("account wipe user=%s counts=%s", user_id, counts)
    return counts
