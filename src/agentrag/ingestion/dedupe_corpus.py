"""Collapse documents that hold the same file.

Until 2026-11-28 nothing stopped one file from becoming several documents
(UI uploads land in data/originals as <uuid>.pdf; folder ingest deduped by
source_id only), and re-ingesting that folder turned 35 files into 122
documents. Retrieval keeps the top-k chunks, so 3-7 identical copies of a
passage crowd out everything else.

    # 1. fill documents.source_bytes_sha for rows that predate the column
    docker compose exec -T -w /app -e PYTHONPATH=/app worker \\
        python -m src.agentrag.ingestion.dedupe_corpus --backfill
    # 2. dry run: groups, which copy is kept, what goes
    ... python -m src.agentrag.ingestion.dedupe_corpus
    # 3. back up the extra copies (ES docs + PG segments), then delete them
    ... python -m src.agentrag.ingestion.dedupe_corpus --apply --backup-dir /app/data/dedupe_backup

Groups are (file bytes, owner): two users' copies of one file are separate
documents on purpose. The kept copy is the most enriched (has segments, most
contextualized leaves, has RAPTOR nodes), then the oldest. Before a copy is
deleted, the kept copy is linked into every notebook the copy was in. Deletes
go through document_service.delete_document (Postgres rows, ES segments,
StructMem group, images). Source files in data/originals are not touched.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agentrag.config import settings

logger = logging.getLogger(__name__)

_IN_FLIGHT = {"pending", "parsing"}


def _candidate_paths(root: Path, doc_id: str, source_id: str | None) -> list[Path]:
    paths: list[Path] = []
    if source_id:
        paths.append(root / source_id)  # folder ingest of the originals dir
        suffix = Path(source_id).suffix
        paths.append(root / f"{doc_id}{suffix}")  # upload: originals/<doc id>.<ext>
    return paths


def _file_is_what_was_ingested(path: Path, content_hash: str | None, ingested_at) -> bool:
    """Only trust today's file for a row whose ingest it provably fed: its
    ingest key still matches, or it has not been modified since the row was
    written. An edited file would otherwise group a stale document with a
    current copy, and the cleanup could keep the stale one."""
    from src.agentrag.ingestion.connectors.folder import content_cache_key

    if content_hash and content_cache_key(path.read_bytes(), path.suffix) == content_hash:
        return True
    if ingested_at is None:
        return False
    return path.stat().st_mtime <= ingested_at.timestamp()


async def backfill(root: Path | None = None) -> tuple[int, int]:
    """Set source_bytes_sha where it is NULL, from the original file, when the
    file is provably unchanged since ingest. Returns (filled, unresolved)."""
    from sqlalchemy import select, update

    from src.agentrag.database import AsyncSessionLocal
    from src.agentrag.database.models import Document

    root = root or Path(settings.ORIGINALS_DIR)
    filled = unresolved = 0
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(Document.id, Document.source_id, Document.content_hash, Document.created_at)
            .where(Document.source_bytes_sha.is_(None))
        )).all()
        for r in rows:
            found = next((p for p in _candidate_paths(root, str(r.id), r.source_id)
                          if p.is_file()), None)
            if found is None or not _file_is_what_was_ingested(found, r.content_hash, r.created_at):
                unresolved += 1
                continue
            sha = hashlib.sha256(found.read_bytes()).hexdigest()
            await session.execute(update(Document).where(Document.id == r.id).values(source_bytes_sha=sha))
            filled += 1
        await session.commit()
    return filled, unresolved


@dataclass
class Group:
    source_bytes_sha: str
    user_id: str | None
    keep: dict[str, Any]
    drop: list[dict[str, Any]] = field(default_factory=list)


def plan(documents: list[dict[str, Any]], counts: dict[str, dict[str, int]]) -> list[Group]:
    """Duplicate groups with the copy to keep. `documents` must be ordered
    oldest first. Groups with an in-flight row are left alone."""
    by_key: dict[tuple[str, str | None], list[dict[str, Any]]] = collections.defaultdict(list)
    for d in documents:
        if d.get("source_bytes_sha"):
            by_key[(d["source_bytes_sha"], d.get("user_id"))].append(d)
    groups: list[Group] = []
    for (sha, owner), docs in by_key.items():
        if len(docs) < 2 or any(d.get("graph_status") in _IN_FLIGHT for d in docs):
            continue

        def score(d: dict[str, Any]) -> tuple[bool, int, bool]:
            c = counts.get(d["id"], {})
            return (c.get("leaves", 0) > 0, c.get("contextualized", 0), c.get("summaries", 0) > 0)

        keep = max(docs, key=score)  # ties: max() keeps the first = oldest
        groups.append(Group(sha, owner, keep, [d for d in docs if d is not keep]))
    return groups


async def _documents() -> list[dict[str, Any]]:
    from sqlalchemy import select

    from src.agentrag.database import AsyncSessionLocal
    from src.agentrag.database.models import Document

    async with AsyncSessionLocal() as session:
        rows = (await session.execute(select(
            Document.id, Document.title, Document.source_id, Document.user_id,
            Document.source_bytes_sha, Document.graph_status,
        ).order_by(Document.created_at, Document.id))).all()
    return [
        {"id": str(r.id), "title": r.title, "source_id": r.source_id,
         "user_id": str(r.user_id) if r.user_id else None,
         "source_bytes_sha": r.source_bytes_sha, "graph_status": r.graph_status}
        for r in rows
    ]


async def _backup(es_store: Any, session: Any, document_id: str, out_dir: Path) -> int:
    """Write one document's ES docs (leaves, images, RAPTOR nodes, with
    embeddings; `{"_id", "_source"}` lines, re-indexable with a bulk `index`)
    and its Postgres segment rows. Raises when the capture is incomplete.
    Not covered: the StructMem graph group and the image dir."""
    from sqlalchemy import select

    from src.agentrag.database.models import Segment

    q = {"term": {"document_id": document_id}}
    expected = (await es_store.client.count(index=es_store.index_name, query=q))["count"]
    if expected > 10000:
        raise RuntimeError(f"{expected} ES docs: too many for a one-page backup")
    resp = await es_store.client.search(index=es_store.index_name, query=q, size=max(expected, 1))
    hits = resp.get("hits", {}).get("hits", [])
    if len(hits) != expected:
        raise RuntimeError(f"ES backup captured {len(hits)} of {expected} docs")
    with (out_dir / f"{document_id}.es.jsonl").open("w", encoding="utf-8") as f:
        for h in hits:
            f.write(json.dumps({"_id": h["_id"], "_source": h["_source"]}, ensure_ascii=False) + "\n")
    segs = (await session.execute(
        select(Segment).where(Segment.document_id == uuid.UUID(document_id))
    )).scalars().all()
    with (out_dir / f"{document_id}.pg_segments.jsonl").open("w", encoding="utf-8") as f:
        for seg in segs:
            f.write(json.dumps({c.name: getattr(seg, c.name) for c in Segment.__table__.columns},
                               ensure_ascii=False, default=str) + "\n")
    return len(hits)


async def apply(groups: list[Group], backup_dir: Path) -> dict[str, Any]:
    """Back up, relink, delete — one copy at a time. Stops at the first copy
    whose ES purge reports an error (its chunks would stay retrievable)."""
    from sqlalchemy import select

    from src.agentrag.adapter.db import adapter_notebook_sources
    from src.agentrag.database import AsyncSessionLocal
    from src.agentrag.database.models import Document
    from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore
    from src.agentrag.ingestion.upload_jobs import link_to_notebooks
    from src.agentrag.services.document_service import delete_document

    backup_dir.mkdir(parents=True, exist_ok=True)
    es = ElasticsearchStore()
    stats: dict[str, Any] = {"deleted": 0, "backed_up_es_docs": 0, "relinked": 0,
                             "skipped": [], "errors": []}
    manifest = backup_dir / "manifest.jsonl"
    try:
        for g in groups:
            for d in g.drop:
                try:
                    async with AsyncSessionLocal() as session:
                        row = await session.get(Document, uuid.UUID(d["id"]))
                        keeper = await session.get(Document, uuid.UUID(g.keep["id"]))
                        if row is None or keeper is None:
                            stats["skipped"].append((d["id"], "copy or keeper gone"))
                            continue
                        # The plan came from a dry run; things may have moved.
                        if row.graph_status in _IN_FLIGHT or keeper.graph_status in _IN_FLIGHT:
                            stats["skipped"].append((d["id"], "ingest in flight"))
                            continue
                        pg_row = {c.name: getattr(row, c.name) for c in Document.__table__.columns}
                        nbs = [str(n) for n in (await session.execute(
                            select(adapter_notebook_sources.c.notebook_id)
                            .where(adapter_notebook_sources.c.document_id == row.id)
                        )).scalars()]
                        n = await _backup(es, session, d["id"], backup_dir)
                        # Keep the content reachable from every notebook the
                        # copy was in (same owner: groups are per owner).
                        await link_to_notebooks(session, keeper.id, nbs)
                        await session.commit()
                    with manifest.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"document": pg_row, "notebooks": nbs, "kept": g.keep["id"],
                                            "es_docs": n}, ensure_ascii=False, default=str) + "\n")
                    result = await delete_document(uuid.UUID(d["id"]))
                    purge = (result or {}).get("purged")
                    stats["backed_up_es_docs"] += n
                    stats["relinked"] += len(nbs)
                    stats["deleted"] += 1
                    if isinstance(purge, str) and purge.startswith("error"):
                        stats["errors"].append((d["id"], f"ES purge: {purge}"))
                        return stats
                except Exception as exc:  # keep going; report it
                    logger.exception("dedupe: copy %s failed", d["id"])
                    stats["errors"].append((d["id"], f"{type(exc).__name__}: {exc}"[:300]))
        return stats
    finally:
        print(json.dumps(stats, default=str))


async def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backfill", action="store_true", help="fill source_bytes_sha from data/originals, then exit")
    ap.add_argument("--apply", action="store_true", help="back up and delete the extra copies (default: dry run)")
    ap.add_argument("--backup-dir", type=Path, help="required with --apply")
    args = ap.parse_args(argv)

    from src.agentrag.ingestion.stores.elasticsearch_store import (
        ElasticsearchStore,
        close_shared_es_client,
    )

    try:
        if args.backfill:
            filled, unresolved = await backfill()
            print(f"source_bytes_sha filled for {filled} document(s); {unresolved} left NULL "
                  "(no original found, or it changed since ingest)")
            return 0
        docs = await _documents()
        missing_sha = sum(1 for d in docs if not d["source_bytes_sha"])
        counts = await ElasticsearchStore().enrichment_counts([d["id"] for d in docs])
        groups = plan(docs, counts)
        drop = sum(len(g.drop) for g in groups)
        for g in groups:
            c = counts.get(g.keep["id"], {})
            print(f"{g.source_bytes_sha[:10]} owner={g.user_id or '-'} keep {g.keep['id']} "
                  f"(leaves={c.get('leaves', 0)} ctx={c.get('contextualized', 0)} raptor={c.get('summaries', 0)}) "
                  f"drop {len(g.drop)}")
        print(f"{len(docs)} documents, {len(groups)} duplicate group(s): keep {len(docs) - drop}, delete {drop}"
              + (f"; {missing_sha} without source_bytes_sha (run --backfill)" if missing_sha else ""))
        if not args.apply or not drop:
            if drop:
                print("dry run: pass --apply --backup-dir DIR to delete")
            return 0
        if not args.backup_dir:
            print("refusing: --apply needs --backup-dir")
            return 2
        stats = await apply(groups, args.backup_dir)
        return 1 if stats["errors"] else 0
    finally:
        await close_shared_es_client()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    raise SystemExit(asyncio.run(_main()))
