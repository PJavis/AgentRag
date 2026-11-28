"""Re-queue deferred enrichment that never landed.

An upload is searchable on plain chunks first; `enrich_segments` adds the
contextual-retrieval blurbs and the RAPTOR tree afterwards (enrich_jobs.py).
If that job is never queued (Valkey down) or fails (every LLM call failed:
DeepSeek outage, empty balance), nothing retries it and the document stays
un-enriched for good. This sweep finds those documents from the index itself
and queues the job again.

    # dry run: what would be queued, and roughly how many LLM calls
    docker compose exec -w /app -e PYTHONPATH=/app worker \\
        python -m src.agentrag.ingestion.enrich_sweep
    # queue them
    ... python -m src.agentrag.ingestion.enrich_sweep --apply [--limit N] [--only contextualize] [--max-calls N]

"Missing" is judged the way the job judges failure: no leaf has a
context_text (a partly contextualized document is left alone), or a document
big enough for RAPTOR has no summary node. Eligibility mirrors the ingest
pipeline (images, spreadsheets and over-cap documents never get these).

This is the only retry path: the job reports a failure instead of raising, so
ARQ's own retry never fires. A document whose RAPTOR build keeps producing no
node is flagged again on every sweep; --max-calls bounds what one run spends.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Any

from src.agentrag.config import settings

logger = logging.getLogger(__name__)

# Not searchable yet: the ingest still owns these. ("failed" is NOT skipped:
# graph_status also records the StructMem graph step, and most of a real
# corpus sits at "failed" there with every segment indexed.)
_SKIP_GRAPH_STATUS = {"pending", "parsing"}


def enrich_job_id(document_id: str, content_hash: str | None) -> str:
    """One ARQ job id per document generation, shared by the pipeline and the
    sweep, so neither queues a second copy while one is queued or running."""
    return f"enrich_segments:{document_id}:{content_hash or ''}"


async def enqueue_enrich(pool: Any, *, document_id: str, content_hash: str | None, **kwargs: Any) -> bool:
    """Queue `enrich_segments`; False when the same job is already queued or
    running. ARQ also refuses an id whose RESULT is still kept (keep_result,
    1 h) — a finished run, possibly failed, or one whose segments a re-ingest
    of the same row just replaced. Both callers only queue when the work is
    needed now, so that stale result is dropped first."""
    from arq.constants import result_key_prefix

    job_id = enrich_job_id(document_id, content_hash)
    await pool.delete(result_key_prefix + job_id)
    job = await pool.enqueue_job(
        "enrich_segments", document_id=document_id, content_hash=content_hash,
        _job_id=job_id, **kwargs,
    )
    return job is not None


@dataclass(frozen=True)
class Candidate:
    document_id: str
    title: str
    content_hash: str | None
    contextualize: bool
    raptor: bool
    leaves: int

    @property
    def est_llm_calls(self) -> int:
        calls = self.leaves if self.contextualize else 0
        if self.raptor:
            # One summary per cluster per level; the levels shrink by the
            # cluster size, so the first level dominates.
            calls += math.ceil(self.leaves / max(settings.RAPTOR_CLUSTER_SIZE, 1)) * 2
        return calls


def needs(source_type: str | None, counts: dict[str, int]) -> tuple[bool, bool]:
    """(contextualize, raptor) still missing for one document."""
    from src.agentrag.ingestion.pipeline import (
        _EXCEL_SOURCE_TYPES,
        _IMAGE_SOURCE_TYPES,
        _contextualize_skip_reason,
    )

    leaves = counts.get("leaves", 0)
    if not leaves or source_type in _IMAGE_SOURCE_TYPES:
        return False, False
    ctx = (
        settings.CONTEXTUAL_RETRIEVAL_ENABLED
        and counts.get("contextualized", 0) == 0
        # The ingest cap counted image chunks too.
        and _contextualize_skip_reason(source_type or "", leaves + counts.get("images", 0)) is None
    )
    raptor = (
        settings.RAPTOR_ENABLED
        and source_type not in _EXCEL_SOURCE_TYPES
        and leaves >= settings.RAPTOR_MIN_LEAVES
        and counts.get("summaries", 0) == 0
    )
    return ctx, raptor


async def _documents() -> list[dict[str, Any]]:
    from sqlalchemy import select

    from src.agentrag.database import AsyncSessionLocal
    from src.agentrag.database.models import Document

    async with AsyncSessionLocal() as session:
        rows = (await session.execute(select(
            Document.id, Document.title, Document.source_type,
            Document.content_hash, Document.graph_status,
        ).order_by(Document.created_at, Document.id))).all()
    return [
        {"id": str(r.id), "title": r.title, "source_type": r.source_type,
         "content_hash": r.content_hash, "graph_status": r.graph_status}
        for r in rows
    ]


async def find_candidates(*, documents: list[dict[str, Any]] | None = None,
                          es_store: Any = None, only: str | None = None) -> list[Candidate]:
    if documents is None:
        documents = await _documents()
    if es_store is None:
        from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore

        es_store = ElasticsearchStore()
    documents = [d for d in documents if d.get("graph_status") not in _SKIP_GRAPH_STATUS]
    counts: dict[str, dict[str, int]] = {}
    for i in range(0, len(documents), 500):
        counts.update(await es_store.enrichment_counts([d["id"] for d in documents[i:i + 500]]))

    out: list[Candidate] = []
    unindexed = 0
    for d in documents:
        c = counts.get(d["id"])
        if not c:
            # No segment carries this document_id (legacy title-keyed
            # segments: scripts/ops/backfill_document_ids.py).
            unindexed += 1
            continue
        ctx, raptor = needs(d.get("source_type"), c)
        if only == "contextualize":
            raptor = False
        elif only == "raptor":
            ctx = False
        if ctx or raptor:
            out.append(Candidate(d["id"], d["title"], d.get("content_hash"), ctx, raptor, c["leaves"]))
    if unindexed:
        logger.warning("%d document(s) have no segment with their document_id; not checked", unindexed)
    return out


async def enqueue(candidates: list[Candidate], pool: Any) -> tuple[int, int]:
    """Returns (queued, already queued or running)."""
    queued = dup = 0
    for c in candidates:
        # No parsed_cache_path / doc_key: the parse cache is long gone, so the
        # job situates each chunk against the joined leaf text instead of the
        # original parse (close, not identical, to an inline ingest).
        if await enqueue_enrich(
            pool, document_id=c.document_id, content_hash=c.content_hash,
            title=c.title, contextualize=c.contextualize, raptor=c.raptor,
        ):
            queued += 1
        else:
            dup += 1
    return queued, dup


async def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="queue the jobs (default: dry run)")
    ap.add_argument("--limit", type=int, default=0, help="queue at most N documents")
    ap.add_argument("--only", choices=["contextualize", "raptor"])
    ap.add_argument("--max-calls", type=int, default=500,
                    help="refuse to queue more than ~N LLM calls (one ingest pass once drained the DeepSeek balance)")
    args = ap.parse_args(argv)

    try:
        candidates = await find_candidates(only=args.only)
    finally:
        from src.agentrag.ingestion.stores.elasticsearch_store import close_shared_es_client

        await close_shared_es_client()
    if args.limit:
        candidates = candidates[: args.limit]
    for c in candidates:
        parts = [p for p, on in (("contextualize", c.contextualize), ("raptor", c.raptor)) if on]
        print(f"{c.document_id}  leaves={c.leaves:5d}  {'+'.join(parts):22s} {c.title[:60]}")
    total = sum(c.est_llm_calls for c in candidates)
    print(f"{len(candidates)} document(s), ~{total} LLM calls")
    if not args.apply or not candidates:
        if candidates:
            print("dry run: pass --apply to queue")
        return 0
    if total > args.max_calls:
        print(f"refusing: ~{total} LLM calls > --max-calls={args.max_calls} (narrow with --limit/--only or raise it)")
        return 2

    from src.agentrag.worker.pool import close_pool, init_pool

    pool = await init_pool(settings.REDIS_URL or "redis://127.0.0.1:6379/0")
    try:
        queued, dup = await enqueue(candidates, pool)
    finally:
        await close_pool()
    print(f"queued {queued}, skipped {dup} already queued or running")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    raise SystemExit(asyncio.run(_main()))
