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
from src.pam.graph.entity_sync import index_graph_entity_views
from src.pam.graph.graphiti_service import GraphitiService
from src.pam.ingestion.chunkers.hybrid_chunker import HybridChunker
from src.pam.ingestion.embedders.factory import build_embedding_provider
from src.pam.ingestion.stores.elasticsearch_store import ElasticsearchStore

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
    embedder = build_embedding_provider(settings)
    es_store = ElasticsearchStore()
    chunker = HybridChunker(
        max_tokens=settings.GRAPH_CHUNK_MAX_TOKENS,
        overlap_tokens=settings.GRAPH_CHUNK_OVERLAP_TOKENS,
        tokenizer_model=settings.CHUNK_TOKENIZER_MODEL,
        split_on_headings=True,
        split_on_paragraphs=False,
    )
    path = Path(job.folder_path) / job.source_id
    content = path.read_text(encoding="utf-8")
    chunks = chunker.chunk(content, metadata={"document_title": job.title})
    total_chunks = len(chunks)

    async with AsyncSessionLocal() as session:
        await session.execute(
            update(Document)
            .where(Document.id == job.document_id)
            .values(
                graph_status="processing",
                graph_total_chunks=total_chunks,
                graph_processed_chunks=0,
                graph_failed_chunks=0,
                graph_last_error=None,
            )
        )
        await session.commit()

    try:
        async def on_progress(payload: dict) -> None:
            async with AsyncSessionLocal() as session:
                await session.execute(
                    update(Document)
                    .where(Document.id == job.document_id)
                    .values(
                        graph_status="processing",
                        graph_total_chunks=payload["total"],
                        graph_processed_chunks=payload["completed"],
                        graph_failed_chunks=0,
                    )
                )
                await session.commit()

        graph_results = await graph_svc.sync_chunks(
            chunks=chunks,
            group_id=job.source_id,
            document_ref=str(job.document_id),
            progress_callback=on_progress,
        )
        await index_graph_entity_views(
            es_store=es_store,
            embedder=embedder,
            graph_results=graph_results,
            document_title=job.title,
            group_id=GraphitiService.normalize_group_id(job.source_id),
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
                    graph_total_chunks=total_chunks,
                    graph_failed_chunks=max(total_chunks, 1),
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
                graph_total_chunks=total_chunks,
                graph_processed_chunks=total_chunks,
                graph_failed_chunks=0,
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
