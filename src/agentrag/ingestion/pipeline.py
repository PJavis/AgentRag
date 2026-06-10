# src/pam/ingestion/pipeline.py
from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any, Literal

from sqlalchemy import update

from src.agentrag.database.models import Document
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.config import settings

from .connectors.folder import FolderConnector
from .parsers.markitdown_parser import MarkItDownParser
from .parsers.excel_parser import ExcelParser
from .parsers.pdf_parser import PDFParser
from .parsers.image_parser import ImageParser
from .chunkers.hybrid_chunker import HybridChunker
from .embedders.factory import build_embedding_provider
from .stores.postgres_store import PostgresStore
from .stores.elasticsearch_store import ElasticsearchStore
from src.agentrag.graph.structmem_service import StructMemService
from src.agentrag.graph.structmem_sync import index_structmem_views
from src.agentrag.graph.graph_jobs import GraphIngestJob
from src.agentrag.worker.pool import get_pool

# Định dạng parse qua PyMuPDF (page-aware, trả về page_data cho image extraction)
_PYMUPDF_SOURCE_TYPES = {"pdf"}
# Định dạng parse qua MarkItDown (Word, PPTX, HTML)
_MARKITDOWN_SOURCE_TYPES = {"word"}
# Định dạng Excel/CSV
_EXCEL_SOURCE_TYPES = {"excel", "csv"}
# Standalone image files — described via Vision LLM
_IMAGE_SOURCE_TYPES = {"image"}
# Audio formats — Whisper transcription
_AUDIO_SOURCE_TYPES = {"audio"}


def _embed_input_for_chunk(chunk: dict[str, Any]) -> str:
    """Text to embed/BM25: contextualized when WS1 produced a context_text,
    else the raw content. The original `content` is always what gets cited."""
    ctx = chunk.get("context_text")
    if ctx:
        return f"{ctx}\n\n{chunk['content']}"
    return chunk["content"]


async def _build_and_index_raptor(builder: Any, es_store: Any, leaf_chunks: list[dict[str, Any]], document_title: str) -> int:
    """Build RAPTOR summary nodes from leaves and index them. Returns count."""
    summary_nodes = await builder.build(leaf_chunks, document_title)
    if not summary_nodes:
        return 0
    await es_store.index_segments(summary_nodes, document_title)
    return len(summary_nodes)


async def ingest_folder(
    folder_path: str,
    graph_ingest_mode: Literal["sync", "async"] | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Ingest thư mục: hỗ trợ .md, .pdf, .docx, .xlsx, .xls, .csv
    graph_ingest_mode: override env STRUCTMEM_INGEST_MODE (sync = chờ Graphiti xong; async = hàng đợi).
    """
    mode = graph_ingest_mode or settings.STRUCTMEM_INGEST_MODE

    connector = FolderConnector(folder_path)
    documents = connector.list_documents()

    pdf_parser = PDFParser()
    markitdown_parser = MarkItDownParser()
    excel_parser = ExcelParser()

    # Image parser — only instantiated when VISION_PROVIDER is configured
    image_parser: ImageParser | None = None
    if settings.VISION_PROVIDER is not None:
        from src.agentrag.services.llm_gateway import LLMGateway
        image_parser = ImageParser(LLMGateway())
    search_chunker = HybridChunker(
        max_tokens=settings.SEARCH_CHUNK_MAX_TOKENS,
        overlap_tokens=settings.SEARCH_CHUNK_OVERLAP_TOKENS,
        tokenizer_model=settings.CHUNK_TOKENIZER_MODEL,
        split_on_headings=True,
        split_on_paragraphs=settings.SEARCH_CHUNK_BY_PARAGRAPH,
    )
    graph_chunker = HybridChunker(
        max_tokens=settings.STRUCTMEM_CHUNK_MAX_TOKENS,
        overlap_tokens=settings.STRUCTMEM_CHUNK_OVERLAP_TOKENS,
        tokenizer_model=settings.CHUNK_TOKENIZER_MODEL,
        split_on_headings=True,
        split_on_paragraphs=False,
    )
    embedder = build_embedding_provider(settings)
    pg_store = PostgresStore()
    es_store = ElasticsearchStore()

    structmem_service: StructMemService | None = None
    if mode == "sync":
        structmem_service = StructMemService()

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
            _pdf_images: list[dict] = []

            # ── Parse theo source_type ─────────────────────────────────────
            source_type = doc.get("source_type", "markdown")
            t0 = time.perf_counter()

            if source_type in _IMAGE_SOURCE_TYPES:
                if image_parser is None:
                    report["status"] = "skipped"
                    report["skip_reason"] = "VISION_PROVIDER not configured"
                    doc_reports.append(report)
                    continue
                parse_result = await image_parser.parse(file_path, doc["title"])
                content = parse_result["parsed_content"]
                report["pages"] = 1
                # Inject image URL into metadata so it survives chunking → ES
                doc["_image_url"] = parse_result.get("image_url", "")
                doc["_image_path"] = parse_result.get("image_path", "")

            elif source_type in _PYMUPDF_SOURCE_TYPES:
                parse_result = pdf_parser.parse(file_path)
                content = parse_result["parsed_content"]
                report["pages"] = parse_result.get("pages", 1)
                if image_parser is not None:
                    _pdf_images = pdf_parser.extract_images(file_path, doc["title"])

            elif source_type in _MARKITDOWN_SOURCE_TYPES:
                parse_result = markitdown_parser.parse(file_path)
                content = parse_result["parsed_content"]
                report["pages"] = parse_result.get("pages", 1)

            elif source_type in _EXCEL_SOURCE_TYPES:
                parse_result = excel_parser.parse(file_path, mode=settings.EXCEL_INGEST_MODE)
                content = parse_result["parsed_content"]
                report["sheets"] = parse_result.get("sheets", [])
                report["total_rows"] = parse_result.get("total_rows", 0)

            elif source_type in _AUDIO_SOURCE_TYPES:
                from .parsers.audio_parser import AudioParser
                audio_parser = AudioParser()
                parse_result = audio_parser.parse(file_path, doc["title"])
                content = parse_result["content"]
                _md = parse_result.get("metadata", {}) or {}
                report["audio_duration_s"] = _md.get("duration_s")
                report["audio_language"] = _md.get("language")
                report["audio_segments"] = len(_md.get("segments", []) or [])

            else:
                # markdown — đọc trực tiếp
                content = Path(file_path).read_text(encoding="utf-8")

            timings["parse_ms"] = (time.perf_counter() - t0) * 1000
            # ──────────────────────────────────────────────────────────────

            t0 = time.perf_counter()
            chunks_search = search_chunker.chunk(
                content, metadata={"document_title": doc["title"]}
            )
            # Bỏ các chunk chỉ có heading hoặc quá ngắn để tránh nhiễu retrieval.
            # Threshold 80 chars loại bỏ "## API", "## Overview" nhưng giữ lại nội dung thực.
            chunks_search = [c for c in chunks_search if len(c["content"].strip()) >= 80]
            timings["chunk_search_ms"] = (time.perf_counter() - t0) * 1000

            # Enrich standalone image chunks with segment_type + image metadata
            if source_type in _IMAGE_SOURCE_TYPES:
                img_url = doc.get("_image_url", "")
                img_path = doc.get("_image_path", "")
                for c in chunks_search:
                    c["segment_type"] = "image"
                    c.setdefault("metadata", {})
                    c["metadata"]["image_url"] = img_url
                    c["metadata"]["image_path"] = img_path
                    if c.get("page_start") is None:
                        c["page_start"] = 1
                        c["page_end"] = 1

            # Tag spreadsheet chunks as tabular so retrieval + the structured
            # SQL pipeline's corpus-aware gate can recognise them as structured
            # data (the content already carries `### Sheet:` / ```csv markers).
            if source_type in _EXCEL_SOURCE_TYPES:
                for c in chunks_search:
                    c["segment_type"] = "table"

            # Add image chunks extracted from PDF pages.
            # In async mode: skip vision LLM describe inline; queue ARQ job sau
            # khi PG/ES text segments saved (search ready ngay).
            _pending_vision_records: list[dict] = []
            if _pdf_images:
                if settings.VISION_INGEST_MODE == "sync":
                    next_pos = len(chunks_search)
                    for img in _pdf_images:
                        description = await image_parser.describe(  # type: ignore[union-attr]
                            img["bytes"], img["mime"], context=doc["title"]
                        )
                        if not description or description.startswith("[image"):
                            continue
                        img_chunk: dict = {
                            "content": description,
                            "content_hash": hashlib.sha256(description.encode("utf-8")).hexdigest(),
                            "segment_type": "image",
                            "section_path": f"page_{img['page']}_image",
                            "position": next_pos,
                            "page_start": img["page"],
                            "page_end": img["page"],
                            "metadata": {
                                "document_title": doc["title"],
                                "image_url": img["url"],
                                "image_path": img["path"],
                            },
                        }
                        chunks_search.append(img_chunk)
                        next_pos += 1
                else:
                    # async: persist refs only (no bytes — worker reads from disk)
                    _pending_vision_records = [
                        {"page": img["page"], "path": img["path"],
                         "url": img["url"], "mime": img["mime"]}
                        for img in _pdf_images
                    ]
                report["pdf_images"] = len(_pdf_images)

            t0 = time.perf_counter()
            chunks_graph = graph_chunker.chunk(
                content, metadata={"document_title": doc["title"]}
            )
            timings["chunk_graph_ms"] = (time.perf_counter() - t0) * 1000

            # S5 — tag chunks with system_tag / specialty_tag / canonical_terms
            # via SectionTagger (lazy import to avoid hard dep when feature off).
            if settings.TAGGING_ENABLED:
                t0 = time.perf_counter()
                from .section_tagger import SectionTagger
                _tagger = SectionTagger()
                chunks_search = [await _tagger.tag_chunk(c) for c in chunks_search]
                timings["tagging_ms"] = (time.perf_counter() - t0) * 1000

            # WS1 — Contextual Retrieval: add a situating context sentence per
            # chunk BEFORE embedding so dense + BM25 see the contextualized text.
            if settings.CONTEXTUAL_RETRIEVAL_ENABLED:
                t0 = time.perf_counter()
                from src.agentrag.ingestion.contextualizer import Contextualizer
                from src.agentrag.services.llm_gateway import LLMGateway
                chunks_search = await Contextualizer(LLMGateway()).contextualize_chunks(
                    doc_text=content, chunks=chunks_search, document_title=doc["title"]
                )
                timings["contextualize_ms"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            texts = [_embed_input_for_chunk(c) for c in chunks_search]
            embeddings = await embedder.embed(texts)
            for c, emb in zip(chunks_search, embeddings):
                c["embedding"] = emb
            timings["embed_ms"] = (time.perf_counter() - t0) * 1000

            doc_id, status = await pg_store.save_document_and_segments(
                session, doc, chunks_search
            )
            report["document_id"] = str(doc_id)
            report["status"] = status

            if status == "skipped":
                report["timings_ms"] = timings
                for k, v in timings.items():
                    totals_ms[k] = totals_ms.get(k, 0.0) + v
                doc_reports.append(report)
                continue

            if status != "retry":
                t0 = time.perf_counter()
                await es_store.index_segments(chunks_search, doc["title"])
                timings["elasticsearch_ms"] = (time.perf_counter() - t0) * 1000
                if settings.RAPTOR_ENABLED:
                    t0 = time.perf_counter()
                    from src.agentrag.ingestion.raptor import RaptorBuilder
                    from src.agentrag.services.llm_gateway import LLMGateway
                    raptor_count = await _build_and_index_raptor(
                        RaptorBuilder(LLMGateway(), embedder),
                        es_store, chunks_search, doc["title"],
                    )
                    timings["raptor_ms"] = (time.perf_counter() - t0) * 1000
                    report["raptor_summary_nodes"] = raptor_count
            else:
                timings["elasticsearch_ms"] = 0.0

            try:
                if mode == "sync":
                    assert structmem_service is not None
                    t0 = time.perf_counter()
                    structmem_results = await structmem_service.sync_chunks(
                        chunks=chunks_graph,
                        group_id=doc["source_id"],
                        document_ref=str(doc_id),
                    )
                    group_id = StructMemService.normalize_group_id(doc["source_id"])
                    sm_stats = await index_structmem_views(
                        es_store=es_store,
                        embedder=embedder,
                        structmem_results=structmem_results,
                        document_title=doc["title"],
                        group_id=group_id,
                    )
                    timings["structmem_ms"] = (time.perf_counter() - t0) * 1000
                    await session.execute(
                        update(Document)
                        .where(Document.id == doc_id)
                        .values(
                            graph_synced=True,
                            graph_status="done",
                            graph_last_error=None,
                            graph_total_chunks=len(chunks_graph),
                            graph_processed_chunks=len(chunks_graph),
                            graph_failed_chunks=0,
                        )
                    )
                    report["graph_status"] = "done"
                    report["graph_chunks"] = len(chunks_graph)
                    report["entries_indexed"] = sm_stats["entries_indexed"]
                else:
                    # Text segments are in ES now → the doc is searchable. Mark
                    # it so immediately (chat works); the StructMem extract +
                    # any enrichment runs as the async tail (status → enriching
                    # → done in the worker).
                    _pages = int(report.get("pages") or 0)
                    await session.execute(
                        update(Document)
                        .where(Document.id == doc_id)
                        .values(
                            graph_synced=False,
                            graph_status="searchable",
                            graph_last_error=None,
                            graph_total_chunks=len(chunks_graph),
                            graph_processed_chunks=0,
                            graph_failed_chunks=0,
                            parse_total_pages=_pages,
                            parse_done_pages=_pages,
                        )
                    )
                    report["graph_status"] = "searchable"
                    report["graph_chunks"] = len(chunks_graph)
                    timings["structmem_ms"] = 0.0
                    try:
                        from src.agentrag.common.progress import publish_progress
                        await publish_progress(user_id, str(doc_id), "searchable")
                    except Exception:
                        pass

                await session.commit()

                if mode == "async":
                    # Cache parsed content so the ARQ worker can read it without
                    # re-parsing (also fixes upload endpoint where temp dir is
                    # deleted before the worker runs).
                    parsed_cache_path: str | None = None
                    try:
                        _cache_dir = Path(settings.STRUCTMEM_CACHE_DIR) / "parsed"
                        _cache_dir.mkdir(parents=True, exist_ok=True)
                        _cache_key = hashlib.sha256(content.encode()).hexdigest()
                        _cache_file = _cache_dir / f"{_cache_key}.txt"
                        if not _cache_file.exists():
                            _cache_file.write_text(content, encoding="utf-8")
                        parsed_cache_path = str(_cache_file)
                    except Exception:
                        pass  # fall back to re-parse in worker
                    await get_pool().enqueue_job(
                        "graph_ingest",
                        document_id=str(doc_id),
                        folder_path=str(Path(folder_path).resolve()),
                        source_id=doc["source_id"],
                        title=doc["title"],
                        parsed_cache_path=parsed_cache_path,
                    )

                # Queue vision describe job nếu async + có pending images
                if _pending_vision_records:
                    await get_pool().enqueue_job(
                        "vision_extract",
                        document_id=str(doc_id),
                        title=doc["title"],
                        image_records=_pending_vision_records,
                    )
                    report["vision_status"] = "queued"
                    report["pdf_images_pending"] = len(_pending_vision_records)
            except Exception as e:
                await session.execute(
                    update(Document)
                    .where(Document.id == doc_id)
                    .values(
                        graph_synced=False,
                        graph_status="failed",
                        graph_last_error=str(e)[:8000],
                        graph_failed_chunks=len(chunks_graph),
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
            "graph_max_tokens": settings.STRUCTMEM_CHUNK_MAX_TOKENS,
        },
        "timings_ms_totals": totals_ms,
        "documents": doc_reports,
    }
