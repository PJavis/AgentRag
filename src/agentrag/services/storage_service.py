"""StorageService — Execution Plane facade for document storage (S4).

Aggregates the bits of CRUD that Reasoning Plane code needs to read:
- ES chunk lookup by content hash
- Postgres document listing

Write operations (ingestion) still go through the dedicated stores; this
facade is read-mostly for the agent loop.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select

from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document
from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore


class StorageService:
    def __init__(self, es_store: ElasticsearchStore | None = None) -> None:
        self._es = es_store or ElasticsearchStore()

    async def get_chunks_by_hashes(self, content_hashes: list[str]) -> list[dict[str, Any]]:
        if not content_hashes:
            return []
        return await self._es.get_chunks_by_hashes(content_hashes)

    async def list_documents(self, limit: int = 20) -> list[dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Document).limit(limit))
            documents = result.scalars().all()
        return [
            {
                "document_id": str(doc.id),
                "title": doc.title,
                "source_id": doc.source_id,
                "graph_status": doc.graph_status,
            }
            for doc in documents
        ]

    async def get_document_by_title(self, title: str) -> list[dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Document).where(Document.title == title)
            )
            documents = result.scalars().all()
        return [
            {
                "document_id": str(doc.id),
                "title": doc.title,
                "source_id": doc.source_id,
                "graph_status": doc.graph_status,
            }
            for doc in documents
        ]
