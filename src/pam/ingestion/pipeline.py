# src/pam/ingestion/pipeline.py
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Literal

from sqlalchemy import update

from src.pam.database.models import Document
from src.pam.database import AsyncSessionLocal
from src.pam.config import settings

from .connectors.markdown import MarkdownConnector
from .parsers.docling_parser import DoclingParser
from .chunkers.hybrid_chunker import HybridChunker
from .embedders.hf_inference_embedder import HFInferenceEmbedder
from .stores.postgres_store import PostgresStore
from .stores.elasticsearch_store import ElasticsearchStore
from src.pam.graph.graphiti_service import GraphitiService
from src.pam.graph.graph_jobs import GraphIngestJob, graph_ingest_queue


async def ingest_folder(
    folder_path: str,
    graph_ingest_mode: Literal["sync", "async"] | None = None,
) -> dict[str, Any]:
    """
    Ingest markdown folder: Postgres + ES dùng chunk search; Graphiti dùng chunk graph (lớn hơn).
    graph_ingest_mode: override env GRAPH_INGEST_MODE (sync = chờ Graphiti xong; async = hàng đợi).
    """
    mode = graph_ingest_mode or settings.GRAPH_INGEST_MODE

    connector = MarkdownConnector(folder_path)
    documents = connector.list_documents()

    parser = DoclingParser()
    search_chunker = HybridChunker(max_tokens=settings.SEARCH_CHUNK_MAX_TOKENS)
    graph_chunker = HybridChunker(max_tokens=settings.GRAPH_CHUNK_MAX_TOKENS)
    embedder = HFInferenceEmbedder(model="intfloat/multilingual-e5-large-instruct")
    pg_store = PostgresStore()
    es_store = ElasticsearchStore()

    graph_service: GraphitiService | None = None
    if mode == "sync":
        graph_service = GraphitiService()
        await graph_service.build_indices()

    ingested_count = 0
    doc_reports: list[dict[str, Any]] = []
    totals_ms: dict[str, float] = {}

    async with AsyncSessionLocal() as session:
        for doc in documents:
            file_path = doc["file_path"]
            report: dict[str, Any] = {
                "source_id": doc["source_id"],
                "title": doc["title"],
            }
            timings: dict[str, float] = {}

            t0 = time.perf_counter()
            parser.parse(file_path)
            timings["parse_ms"] = (time.perf_counter() - t0) * 1000

            content = Path(file_path).read_text(encoding="utf-8")

            t0 = time.perf_counter()
            chunks_search = search_chunker.chunk(
                content, metadata={"document_title": doc["title"]}
            )
            timings["chunk_search_ms"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            chunks_graph = graph_chunker.chunk(
                content, metadata={"document_title": doc["title"]}
            )
            timings["chunk_graph_ms"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            texts = [c["content"] for c in chunks_search]
            embeddings = await embedder.embed(texts)
            for c, emb in zip(chunks_search, embeddings):
                c["embedding"] = emb
            timings["embed_ms"] = (time.perf_counter() - t0) * 1000

            doc_id, status = await pg_store.save_document_and_segments(
                session, doc, chunks_search
            )
            report["status"] = status

            if status == "skipped":
                report["timings_ms"] = timings
                for k, v in timings.items():
                    totals_ms[k] = totals_ms.get(k, 0.0) + v
                doc_reports.append(report)
                continue

            t0 = time.perf_counter()
            await es_store.index_segments(chunks_search, doc["title"])
            timings["elasticsearch_ms"] = (time.perf_counter() - t0) * 1000

            try:
                if mode == "sync":
                    assert graph_service is not None
                    t0 = time.perf_counter()
                    graph_results = await graph_service.sync_chunks(
                        chunks_graph, str(doc_id)
                    )
                    timings["graphiti_ms"] = (time.perf_counter() - t0) * 1000
                    await session.execute(
                        update(Document)
                        .where(Document.id == doc_id)
                        .values(
                            graph_synced=True,
                            graph_status="done",
                            graph_last_error=None,
                        )
                    )
                    report["graph_status"] = "done"
                    report["graph_chunks"] = len(chunks_graph)
                    report["graph_episodes"] = len(graph_results)
                else:
                    await session.execute(
                        update(Document)
                        .where(Document.id == doc_id)
                        .values(
                            graph_synced=False,
                            graph_status="queued",
                            graph_last_error=None,
                        )
                    )
                    report["graph_status"] = "queued"
                    report["graph_chunks"] = len(chunks_graph)
                    timings["graphiti_ms"] = 0.0

                await session.commit()

                if mode == "async":
                    await graph_ingest_queue.put(
                        GraphIngestJob(
                            document_id=doc_id,
                            folder_path=str(Path(folder_path).resolve()),
                            source_id=doc["source_id"],
                            title=doc["title"],
                        )
                    )
            except Exception as e:
                await session.execute(
                    update(Document)
                    .where(Document.id == doc_id)
                    .values(
                        graph_synced=False,
                        graph_status="failed",
                        graph_last_error=str(e)[:8000],
                    )
                )
                await session.commit()
                report["graph_status"] = "failed"
                report["graph_error"] = str(e)
                report["timings_ms"] = timings
                for k, v in timings.items():
                    totals_ms[k] = totals_ms.get(k, 0.0) + v
                doc_reports.append(report)
                continue

            ingested_count += 1
            report["timings_ms"] = timings
            for k, v in timings.items():
                totals_ms[k] = totals_ms.get(k, 0.0) + v
            doc_reports.append(report)

    return {
        "status": "success",
        "ingested": ingested_count,
        "total": len(documents),
        "graph_ingest_mode": mode,
        "chunking": {
            "search_max_tokens": settings.SEARCH_CHUNK_MAX_TOKENS,
            "graph_max_tokens": settings.GRAPH_CHUNK_MAX_TOKENS,
        },
        "timings_ms_totals": totals_ms,
        "documents": doc_reports,
    }
