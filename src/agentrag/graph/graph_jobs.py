"""StructMem document ingest processing — called by the ARQ worker."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import update

from src.agentrag.config import settings
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document
from src.agentrag.graph.structmem_service import StructMemService
from src.agentrag.graph.structmem_sync import index_structmem_views
from src.agentrag.ingestion.chunkers.hybrid_chunker import HybridChunker
from src.agentrag.ingestion.embedders.factory import build_embedding_provider
from src.agentrag.ingestion.parsers.excel_parser import ExcelParser
from src.agentrag.ingestion.parsers.markitdown_parser import MarkItDownParser
from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore

if TYPE_CHECKING:
    from arq import ArqRedis

logger = logging.getLogger(__name__)

_structmem_service: StructMemService | None = None


@dataclass(frozen=True)
class GraphIngestJob:
    document_id: uuid.UUID
    folder_path: str
    source_id: str
    title: str
    parsed_cache_path: str | None = None


async def _get_structmem_service() -> StructMemService:
    global _structmem_service
    if _structmem_service is None:
        _structmem_service = StructMemService()
    return _structmem_service


async def process_graph_job(job: GraphIngestJob, arq_pool: ArqRedis | None = None) -> None:
    svc = await _get_structmem_service()
    embedder = build_embedding_provider(settings)
    es_store = ElasticsearchStore()
    chunker = HybridChunker(
        max_tokens=settings.STRUCTMEM_CHUNK_MAX_TOKENS,
        overlap_tokens=settings.STRUCTMEM_CHUNK_OVERLAP_TOKENS,
        tokenizer_model=settings.CHUNK_TOKENIZER_MODEL,
        split_on_headings=True,
        split_on_paragraphs=False,
    )
    # Use cached parsed content if available (set by pipeline.py to avoid re-parsing
    # and to survive temp-dir cleanup in the upload endpoint).
    cache_path = Path(job.parsed_cache_path) if job.parsed_cache_path else None
    if cache_path and cache_path.exists():
        content = cache_path.read_text(encoding="utf-8")
    else:
        path = Path(job.folder_path) / job.source_id
        suffix = path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            parse_result = ExcelParser().parse(str(path), mode=settings.EXCEL_INGEST_MODE)
            content = parse_result["parsed_content"]
        elif suffix == ".csv":
            parse_result = ExcelParser().parse(str(path), mode="markdown")
            content = parse_result["parsed_content"]
        elif suffix in (".pdf", ".docx", ".doc", ".pptx", ".ppt", ".html", ".htm"):
            parse_result = MarkItDownParser().parse(str(path))
            content = parse_result["parsed_content"]
        else:
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

        structmem_results = await svc.sync_chunks(
            chunks=chunks,
            group_id=job.source_id,
            document_ref=str(job.document_id),
            progress_callback=on_progress,
        )
        group_id = StructMemService.normalize_group_id(job.source_id)
        index_stats = await index_structmem_views(
            es_store=es_store,
            embedder=embedder,
            structmem_results=structmem_results,
            document_title=job.title,
            group_id=group_id,
        )
        logger.info(
            "StructMem indexed %s entries for document %s",
            index_stats.get("entries_indexed", 0),
            job.document_id,
        )

        if total_chunks >= settings.STRUCTMEM_CONSOLIDATION_THRESHOLD and arq_pool is not None:
            try:
                await arq_pool.enqueue_job(
                    "consolidate",
                    group_id=group_id,
                    document_id=str(job.document_id),
                    trigger_chunk_count=total_chunks,
                )
            except Exception as ce:
                logger.warning("Failed to enqueue consolidation job: %s", ce)

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

    if cache_path and cache_path.exists():
        try:
            cache_path.unlink()
        except Exception:
            pass
