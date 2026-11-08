"""Deleting a document — the one implementation every delete path uses.

Before this, four routes deleted documents four different ways:

- root `DELETE /documents/{id}` removed the Postgres rows only, so the
  document's segments stayed in Elasticsearch and kept being retrieved and
  cited by `/chat`;
- adapter `DELETE /sources/{id}` purged ES and the image folder, but never
  invalidated the answer cache;
- notebook delete with `delete_exclusive_sources` deleted Document rows only;
- account deletion purged ES per title on its own.

A deleted document must stop being retrievable everywhere at once: Postgres,
every ES index that carries its title, its extracted images, and any cached
answer built while it existed (bumping the corpus version re-keys the cache).
"""
from __future__ import annotations

import logging
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select

from src.agentrag.config import settings
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document, Segment

logger = logging.getLogger(__name__)


def image_dir_for(title: str) -> Path:
    """Same sanitiser image_parser uses when it writes a document's images."""
    return Path(settings.IMAGE_STORAGE_DIR) / re.sub(r"[^\w\-]", "_", title)[:80]


async def _titles_still_used(titles: list[str]) -> set[str]:
    """ES and the image folder are keyed by title, not id. Titles another
    document still carries must not be purged (it would delete ITS hits too).
    One query for all titles."""
    if not titles:
        return set()
    async with AsyncSessionLocal() as session:
        rows = await session.execute(
            select(Document.title).where(Document.title.in_(titles)).distinct()
        )
        return {t for (t,) in rows.all()}


async def purge_document_artifacts(titles: list[str]) -> dict[str, Any]:
    """Best-effort purge of everything keyed by title, after the Postgres rows
    are gone. Bumps the corpus version once, only if something was purged."""
    import asyncio

    unique = list(dict.fromkeys(t for t in titles if t))
    still_used = await _titles_still_used(unique)
    purged: dict[str, Any] = {}
    removed_any = False
    for title in unique:
        if title in still_used:
            purged[title] = "kept: title shared with another document"
            continue
        removed_any = True
        try:
            from src.agentrag.ingestion.stores.elasticsearch_store import (
                ElasticsearchStore,
            )

            purged[title] = await ElasticsearchStore().delete_document(title)
        except Exception as exc:
            logger.warning("ES purge failed for %r: %s", title, exc)
            purged[title] = f"error: {type(exc).__name__}"
        shutil.rmtree(image_dir_for(title), ignore_errors=True)
    if removed_any:
        try:
            from src.agentrag.common.corpus_version import bump_corpus_version

            # Sync redis call — keep it off the event loop.
            await asyncio.to_thread(bump_corpus_version)
        except Exception as exc:
            logger.warning("corpus version bump failed after delete: %s", exc)
    return purged


async def delete_document(document_id: uuid.UUID) -> dict[str, Any] | None:
    """Delete one document everywhere. Returns None when it does not exist."""
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            return None
        title = doc.title
        await session.execute(delete(Segment).where(Segment.document_id == doc.id))
        await session.delete(doc)
        await session.commit()
    purged = await purge_document_artifacts([title] if title else [])
    return {"document_id": str(document_id), "title": title, "purged": purged.get(title, {})}


async def delete_documents_in_session(session, docs: list[Document]) -> list[str]:
    """Delete Document rows (and their segments) inside a caller's transaction.

    Returns the titles to hand to `purge_document_artifacts` AFTER the caller
    commits — purging before the commit would leave ES empty for rows a failed
    transaction kept.
    """
    titles: list[str] = []
    for doc in docs:
        await session.execute(delete(Segment).where(Segment.document_id == doc.id))
        if doc.title:
            titles.append(doc.title)
        await session.delete(doc)
    return titles
