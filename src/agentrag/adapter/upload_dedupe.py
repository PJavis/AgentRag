"""Upload dedupe helper.

Computes a content hash of the uploaded bytes and looks up an existing
Document with the same hash. If found, reuse it instead of re-ingesting.
"""
from __future__ import annotations

import hashlib

from sqlalchemy import select

from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def find_existing_document(content_hash: str) -> Document | None:
    """Return the most recently created Document matching `content_hash`, or None."""
    if not content_hash:
        return None
    async with AsyncSessionLocal() as session:
        row = await session.execute(
            select(Document)
            .where(Document.content_hash == content_hash)
            .order_by(Document.created_at.desc())
            .limit(1)
        )
        return row.scalar_one_or_none()
