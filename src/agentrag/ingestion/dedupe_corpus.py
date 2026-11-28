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
    # 3. back up the extra copies' ES segments, then delete them
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


async def backfill(root: Path | None = None) -> tuple[int, int]:
    """Set source_bytes_sha where it is NULL, from the original file. Returns
    (filled, unresolved)."""
    from sqlalchemy import select, update

    from src.agentrag.database import AsyncSessionLocal
    from src.agentrag.database.models import Document

    root = root or Path(settings.ORIGINALS_DIR)
    filled = unresolved = 0
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(Document.id, Document.source_id).where(Document.source_bytes_sha.is_(None))
        )).all()
        for r in rows:
            found = next((p for p in _candidate_paths(root, str(r.id), r.source_id)
                          if p.is_file()), None)
            if found is None:
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


async def _backup_segments(es_store: Any, document_id: str, out_dir: Path) -> int:
    """Every ES doc of one document (leaves, images, RAPTOR nodes, with
    embeddings) as JSON lines `{"_id", "_source"}` — re-indexable with a bulk
    `index` call."""
    hits_out: list[dict[str, Any]] = []
    search_after = None
    while True:
        body: dict[str, Any] = {
            "size": 1000,
            "query": {"term": {"document_id": document_id}},
            "sort": [{"_doc": "asc"}],
        }
        if search_after is not None:
            body["search_after"] = search_after
        resp = await es_store.client.search(index=es_store.index_name, **body)
        hits = resp.get("hits", {}).get("hits", [])
        hits_out += [{"_id": h["_id"], "_source": h["_source"]} for h in hits]
        if len(hits) < body["size"]:
            break
        search_after = hits[-1]["sort"]
    with (out_dir / f"{document_id}.es.jsonl").open("w", encoding="utf-8") as f:
        for h in hits_out:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    return len(hits_out)


async def apply(groups: list[Group], backup_dir: Path) -> dict[str, int]:
    from sqlalchemy import select

    from src.agentrag.adapter.db import adapter_notebook_sources
    from src.agentrag.database import AsyncSessionLocal
    from src.agentrag.database.models import Document
    from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore
    from src.agentrag.ingestion.upload_jobs import link_to_notebooks
    from src.agentrag.services.document_service import delete_document

    backup_dir.mkdir(parents=True, exist_ok=True)
    es = ElasticsearchStore()
    stats = {"deleted": 0, "backed_up_segments": 0, "relinked": 0, "missing": 0}
    manifest = backup_dir / "manifest.jsonl"
    for g in groups:
        for d in g.drop:
            async with AsyncSessionLocal() as session:
                row = await session.get(Document, uuid.UUID(d["id"]))
                if row is None:
                    stats["missing"] += 1
                    continue
                pg_row = {c.name: getattr(row, c.name) for c in Document.__table__.columns}
                nbs = [str(n) for n in (await session.execute(
                    select(adapter_notebook_sources.c.notebook_id)
                    .where(adapter_notebook_sources.c.document_id == row.id)
                )).scalars()]
                # Keep the content reachable from every notebook the copy was in.
                await link_to_notebooks(session, uuid.UUID(g.keep["id"]), nbs)
                await session.commit()
                stats["relinked"] += len(nbs)
            n = await _backup_segments(es, d["id"], backup_dir)
            stats["backed_up_segments"] += n
            with manifest.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"document": pg_row, "notebooks": nbs, "kept": g.keep["id"],
                                    "es_segments": n}, ensure_ascii=False, default=str) + "\n")
            await delete_document(uuid.UUID(d["id"]))
            stats["deleted"] += 1
    return stats


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
            print(f"source_bytes_sha filled for {filled} document(s); {unresolved} without a findable original")
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
        print(json.dumps(stats))
        return 0
    finally:
        await close_shared_es_client()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    raise SystemExit(asyncio.run(_main()))
