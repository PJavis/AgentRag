"""Stamp `document_id` on Elasticsearch segments indexed before that field existed.

Chat scope and delete now match segments by Postgres Document.id. Segments
indexed earlier carry only `document_title`, so a scoped turn cannot see them
until they are stamped. No LLM calls, no re-embedding: one update_by_query per
document, matched by title.

A title carried by more than one document cannot be attributed this way; those
documents are listed and left alone (re-ingest them to get ids).

    uv run python scripts/ops/backfill_document_ids.py            # dry run
    uv run python scripts/ops/backfill_document_ids.py --apply
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _missing_id_query(title: str) -> dict:
    return {
        "bool": {
            "filter": [{"term": {"document_title.keyword": title}}],
            "must_not": [{"exists": {"field": "document_id"}}],
        }
    }


async def backfill(apply: bool) -> dict:
    from sqlalchemy import select

    from src.agentrag.database import AsyncSessionLocal
    from src.agentrag.database.models import Document
    from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore

    store = ElasticsearchStore()
    index = store.index_name
    if not await store.client.indices.exists(index=index):
        return {"index": index, "error": "index does not exist"}
    if apply:
        # Adds the keyword mapping on an existing index (no-op when present).
        await store._ensure_segment_fields()

    async with AsyncSessionLocal() as session:
        rows = (await session.execute(select(Document.id, Document.title))).all()
    by_title: dict[str, list[str]] = defaultdict(list)
    for doc_id, title in rows:
        if title:
            by_title[title].append(str(doc_id))

    report = {"index": index, "applied": apply, "stamped": 0, "documents": 0,
              "ambiguous_titles": {}, "unstamped_left": None}
    for title, ids in sorted(by_title.items()):
        pending = (await store.client.count(index=index, query=_missing_id_query(title)))["count"]
        if not pending:
            continue
        if len(ids) > 1:
            report["ambiguous_titles"][title] = {"document_ids": ids, "segments": pending}
            continue
        report["documents"] += 1
        if not apply:
            report["stamped"] += pending
            continue
        resp = await store.client.update_by_query(
            index=index,
            query=_missing_id_query(title),
            script={"source": "ctx._source.document_id = params.id",
                    "lang": "painless", "params": {"id": ids[0]}},
            conflicts="proceed",
            refresh=True,
        )
        report["stamped"] += resp.get("updated", 0)

    report["unstamped_left"] = (await store.client.count(
        index=index,
        query={"bool": {"must_not": [{"exists": {"field": "document_id"}}]}},
    ))["count"]
    await store.client.close()
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = ap.parse_args()
    report = asyncio.run(backfill(args.apply))
    import json

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
