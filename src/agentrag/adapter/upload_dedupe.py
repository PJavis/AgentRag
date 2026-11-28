"""Upload dedupe helper.

Looks up an existing Document holding the same file bytes
(`Document.source_bytes_sha`) for the same owner. If found, reuse it instead of
re-ingesting.
"""
from __future__ import annotations

import hashlib

import uuid

from sqlalchemy import exists, select

from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document, Segment


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def find_existing_document(source_bytes_sha: str, owner: uuid.UUID | None) -> Document | None:
    """Most recent Document with these file bytes, owned by `owner` (None =
    the anonymous/shared bucket), that has segments. Owner-scoped: matching
    another user's document would link their private file into the caller's
    notebook. A copy without segments (failed parse, upload still in flight)
    is no copy: the upload ingests again.

    Keyed on the bytes, not on content_hash: content_hash mixes in the parse
    settings, so it changes under every parse-setting change and the lookup
    never matched an ingested PDF.
    """
    if not source_bytes_sha:
        return None
    owner_clause = Document.user_id.is_(None) if owner is None else Document.user_id == owner
    async with AsyncSessionLocal() as session:
        row = await session.execute(
            select(Document)
            .where(
                Document.source_bytes_sha == source_bytes_sha,
                owner_clause,
                exists().where(Segment.document_id == Document.id),
            )
            .order_by(Document.created_at.desc())
            .limit(1)
        )
        return row.scalar_one_or_none()
