# src/pam/graph/graph_jobs.py
"""Hàng đợi ingest Graphiti chạy nền (GRAPH_INGEST_MODE=async)."""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import update

from src.pam.config import settings
from src.pam.database import AsyncSessionLocal
from src.pam.database.models import Document
from src.pam.graph.graphiti_service import GraphitiService
from src.pam.ingestion.chunkers.hybrid_chunker import HybridChunker

logger = logging.getLogger(__name__)

graph_ingest_queue: asyncio.Queue[GraphIngestJob] = asyncio.Queue()

_graph_service: GraphitiService | None = None


@dataclass(frozen=True)
class GraphIngestJob:
    document_id: uuid.UUID
    folder_path: str
    source_id: str
    title: str


async def _get_graph_service() -> GraphitiService:
    global _graph_service
    if _graph_service is None:
        _graph_service = GraphitiService()
        await _graph_service.build_indices()
    return _graph_service


async def process_graph_job(job: GraphIngestJob) -> None:
    graph_svc = await _get_graph_service()
    chunker = HybridChunker(
        max_tokens=settings.GRAPH_CHUNK_MAX_TOKENS,
        overlap_tokens=settings.GRAPH_CHUNK_OVERLAP_TOKENS,
        tokenizer_model=settings.CHUNK_TOKENIZER_MODEL,
        split_on_headings=True,
    )
    path = Path(job.folder_path) / job.source_id
    content = path.read_text(encoding="utf-8")
    chunks = chunker.chunk(content, metadata={"document_title": job.title})

    async with AsyncSessionLocal() as session:
        await session.execute(
            update(Document)
            .where(Document.id == job.document_id)
            .values(graph_status="processing")
        )
        await session.commit()

    try:
        await graph_svc.sync_chunks(
            chunks=chunks,
            group_id=job.source_id,
            document_ref=str(job.document_id),
        )
    except Exception as e:
        logger.exception("Graph ingest failed for document %s", job.document_id)
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(Document)
                .where(Document.id == job.document_id)
                .values(
                    graph_status="failed",
                    graph_last_error=str(e)[:8000],
                    graph_synced=False,
                )
            )
            await session.commit()
        return

    async with AsyncSessionLocal() as session:
        await session.execute(
            update(Document)
            .where(Document.id == job.document_id)
            .values(
                graph_status="done",
                graph_synced=True,
                graph_last_error=None,
            )
        )
        await session.commit()


async def run_graph_worker(stop_event: asyncio.Event) -> None:
    logger.info("Graph ingest worker started")
    while True:
        if stop_event.is_set() and graph_ingest_queue.empty():
            break
        try:
            job = await asyncio.wait_for(graph_ingest_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        try:
            await process_graph_job(job)
        finally:
            graph_ingest_queue.task_done()
    logger.info("Graph ingest worker stopped")
