"""Deferred enrichment — contextual retrieval + RAPTOR after a document is
already searchable.

Inline, both ran ahead of "searchable": one LLM call per chunk to
contextualize, then the RAPTOR summaries, so an upload sat unchattable for the
whole LLM pass. With INGEST_DEFER_ENRICHMENT the pipeline indexes plain
chunks, flips the document to searchable, and queues this job. It rewrites the
same ES segments in place (context_text + embedding of the contextualized
text) and builds the RAPTOR tree, so the end state matches an inline ingest.

A re-ingest can replace the document while this runs. The job re-checks the
Document row (same id, same content_hash) before each write and stops if it
was superseded; the newer ingest queued its own job.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agentrag.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrichJob:
    document_id: str
    title: str
    content_hash: str | None
    doc_key: str | None = None
    contextualize: bool = True
    raptor: bool = True
    parsed_cache_path: str | None = None


async def _still_current(job: EnrichJob) -> bool:
    from src.agentrag.database import AsyncSessionLocal
    from src.agentrag.database.models import Document

    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, uuid.UUID(job.document_id))
        return doc is not None and doc.content_hash == job.content_hash


def _doc_text(job: EnrichJob, leaves: list[dict[str, Any]]) -> tuple[str, bool]:
    """(document text, whether it is the real parse)."""
    if job.parsed_cache_path:
        try:
            return Path(job.parsed_cache_path).read_text(encoding="utf-8"), True
        except OSError:
            pass
    # Cache gone (worker restart, other host): the leaves in order are a close
    # stand-in.
    return "\n\n".join(str(leaf.get("content") or "") for leaf in leaves), False


async def process_enrich_job(
    job: EnrichJob,
    *,
    es_store: Any = None,
    embedder: Any = None,
    gateway: Any = None,
    is_current: Any = None,
) -> dict[str, Any]:
    """Returns a small report: status in {done, stale, empty, failed}."""
    from src.agentrag.ingestion.pipeline import _embed_input_for_chunk

    if es_store is None:
        from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore

        es_store = ElasticsearchStore()
    if embedder is None:
        from src.agentrag.ingestion.embedders.factory import build_embedding_provider

        embedder = build_embedding_provider(settings)
    if gateway is None:
        from src.agentrag.services.llm_gateway import LLMGateway

        gateway = LLMGateway()
    is_current = is_current or _still_current

    report: dict[str, Any] = {"document_id": job.document_id}

    def _out(status: str, **extra: Any) -> dict[str, Any]:
        report.update(status=status, **extra)
        return report

    try:
        if not await is_current(job):
            return _out("stale")
        leaves = await es_store.fetch_leaf_segments(job.document_id)
        if not leaves:
            return _out("empty")
        leaf_ids = [leaf["_id"] for leaf in leaves]

        if job.contextualize:
            from src.agentrag.ingestion.contextualizer import Contextualizer

            doc_text, real_parse = _doc_text(job, leaves)
            await Contextualizer(gateway).contextualize_chunks(
                doc_text=doc_text,
                chunks=leaves,
                document_title=job.title,
                # Blurbs from the stand-in text must not be cached under the
                # file's key, where a later real ingest would reuse them.
                doc_key=job.doc_key if real_parse else None,
            )
            changed = [leaf for leaf in leaves if leaf.get("context_text")]
            if not changed:
                # Every LLM call failed (the contextualizer swallows them).
                return _out("failed", error="no chunk contextualized", contextualized=0)
            if changed:
                vectors = await embedder.embed([_embed_input_for_chunk(c) for c in changed])
                for leaf, vec in zip(changed, vectors):
                    leaf["embedding"] = vec
                if not await is_current(job):
                    return _out("stale")
                missing = await es_store.update_segments(
                    [
                        (leaf["_id"], {"context_text": leaf["context_text"], "embedding": leaf["embedding"]})
                        for leaf in changed
                    ]
                )
                if missing:
                    # Purged under us by a re-ingest of the same document.
                    return _out("stale", missing=missing)
            report["contextualized"] = len(changed)

        if job.raptor:
            from src.agentrag.ingestion.raptor import RaptorBuilder

            for leaf in leaves:
                leaf.pop("_id", None)
            nodes = await RaptorBuilder(gateway, embedder).build(leaves, job.title)
            report["raptor_summary_nodes"] = len(nodes)
            if nodes:
                # The build takes minutes. Write only if the document is still
                # this generation (not deleted, not re-ingested): nodes under a
                # dead document_id are never purged and surface in answers.
                if not await is_current(job):
                    return _out("stale")
                current_ids = [s["_id"] for s in await es_store.fetch_leaf_segments(job.document_id)]
                if current_ids != leaf_ids:
                    return _out("stale")
                # Replace the tree (retry / duplicate job), never stack a
                # second one. An empty build keeps the old tree.
                await es_store.delete_summary_nodes(job.document_id)
                await es_store.index_segments(nodes, job.title, document_id=job.document_id)
        return _out("done")
    except Exception as exc:
        # The plain segments are indexed and searchable; only the enrichment
        # is missing. Do not fail the document over it.
        logger.exception("enrich_segments failed for document %s", job.document_id)
        return _out("failed", error=str(exc)[:500])
    finally:
        if job.parsed_cache_path:
            try:
                Path(job.parsed_cache_path).unlink(missing_ok=True)
            except Exception:
                pass
        logger.info("enrich_segments %s: %s", job.document_id, report)
