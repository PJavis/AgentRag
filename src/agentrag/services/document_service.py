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
its ES docs (segments by document_id, memory by StructMem group), its extracted images, and any cached
answer built while it existed (bumping the corpus version re-keys the cache).
"""
from __future__ import annotations

import logging
import re
import shutil
import uuid
from dataclasses import dataclass
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


@dataclass(frozen=True)
class PurgeRef:
    """What purging needs from a deleted Document row (read before the delete)."""

    document_id: str
    title: str | None
    source_id: str | None = None

    @classmethod
    def of(cls, doc: Any) -> "PurgeRef":
        return cls(str(doc.id), doc.title, getattr(doc, "source_id", None))


async def _titles_still_used(titles: list[str]) -> set[str]:
    """Titles another document still carries. Segments are purged by
    document_id, but legacy segments (indexed before that field), entities,
    relationships and the image folder are keyed by title only, and must not be
    purged while another document uses the title. One query for all titles."""
    if not titles:
        return set()
    async with AsyncSessionLocal() as session:
        rows = await session.execute(
            select(Document.title).where(Document.title.in_(titles)).distinct()
        )
        return {t for (t,) in rows.all()}


async def _source_ids_still_used(source_ids: list[str]) -> set[str]:
    """source_ids another document still carries. StructMem memory is grouped
    by source_id, and legacy uploads used the bare filename, so two users'
    "report.pdf" share one group: purging it would delete the other's memory."""
    if not source_ids:
        return set()
    async with AsyncSessionLocal() as session:
        rows = await session.execute(
            select(Document.source_id).where(Document.source_id.in_(source_ids)).distinct()
        )
        return {s for (s,) in rows.all()}


async def purge_document_artifacts(refs: list[PurgeRef]) -> dict[str, Any]:
    """Best-effort purge of a deleted document's ES docs and images, after the
    Postgres rows are gone. Bumps the corpus version once, only if something
    was purged. Returns per-document results keyed by document id."""
    import asyncio

    from src.agentrag.graph.structmem_service import StructMemService

    unique = list({r.document_id: r for r in refs if r.document_id}.values())
    still_used = await _titles_still_used([r.title for r in unique if r.title])
    groups_used = await _source_ids_still_used([r.source_id for r in unique if r.source_id])
    purged: dict[str, Any] = {}
    removed_any = False
    for ref in unique:
        shared = ref.title in still_used
        removed_any = True
        try:
            from src.agentrag.ingestion.stores.elasticsearch_store import (
                ElasticsearchStore,
            )

            purged[ref.document_id] = await ElasticsearchStore().delete_document(
                ref.title or "",
                document_id=ref.document_id,
                group_id=(
                    StructMemService.normalize_group_id(ref.source_id)
                    if ref.source_id and ref.source_id not in groups_used else None
                ),
                title_shared=shared or not ref.title,
            )
        except Exception as exc:
            logger.warning("ES purge failed for %s (%r): %s", ref.document_id, ref.title, exc)
            purged[ref.document_id] = f"error: {type(exc).__name__}"
        if ref.title and not shared:
            shutil.rmtree(image_dir_for(ref.title), ignore_errors=True)
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
        ref = PurgeRef.of(doc)
        await session.execute(delete(Segment).where(Segment.document_id == doc.id))
        await session.delete(doc)
        await session.commit()
    purged = await purge_document_artifacts([ref])
    return {"document_id": str(document_id), "title": ref.title, "purged": purged.get(ref.document_id, {})}


async def delete_documents_in_session(session, docs: list[Document]) -> list[PurgeRef]:
    """Delete Document rows (and their segments) inside a caller's transaction.

    Returns the refs to hand to `purge_document_artifacts` AFTER the caller
    commits — purging before the commit would leave ES empty for rows a failed
    transaction kept.
    """
    refs: list[PurgeRef] = []
    for doc in docs:
        await session.execute(delete(Segment).where(Segment.document_id == doc.id))
        refs.append(PurgeRef.of(doc))
        await session.delete(doc)
    return refs
