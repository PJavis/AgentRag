"""Re-tag every existing segment in Elasticsearch with system_tag /
specialty_tag / canonical_terms derived from section_path + content.

Idempotent: re-running just refreshes tags.

Usage:
    PYTHONPATH=. uv run python scripts/backfill_tags.py
    PYTHONPATH=. uv run python scripts/backfill_tags.py --batch 200 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Any

from elasticsearch import AsyncElasticsearch

from src.agentrag.config import settings
from src.agentrag.ingestion.section_tagger import SectionTagger


async def _main(batch: int, dry_run: bool) -> None:
    es = AsyncElasticsearch(hosts=[settings.ELASTICSEARCH_URL])
    tagger = SectionTagger()
    try:
        if not await es.indices.exists(index=settings.ELASTICSEARCH_INDEX_NAME):
            print(
                f"Index '{settings.ELASTICSEARCH_INDEX_NAME}' does not exist yet — "
                "no segments to backfill. Skipping (this is normal on a fresh install)."
            )
            return
        scroll = await es.search(
            index=settings.ELASTICSEARCH_INDEX_NAME,
            body={"query": {"match_all": {}}, "size": batch},
            scroll="2m",
        )
        sid = scroll["_scroll_id"]
        total = scroll["hits"]["total"]["value"]
        print(f"Backfilling {total} segments (batch={batch}, dry_run={dry_run})")
        updated = 0
        while True:
            hits = scroll["hits"]["hits"]
            if not hits:
                break
            actions: list[dict[str, Any]] = []
            for hit in hits:
                src = hit["_source"]
                tagged = await tagger.tag_chunk(src)
                if dry_run:
                    continue
                actions.append(
                    {"update": {"_index": hit["_index"], "_id": hit["_id"]}}
                )
                actions.append(
                    {
                        "doc": {
                            "system_tag": tagged.get("system_tag"),
                            "specialty_tag": tagged.get("specialty_tag", []),
                            "canonical_terms": tagged.get("canonical_terms", []),
                        }
                    }
                )
                updated += 1
            if actions:
                await es.bulk(operations=actions, refresh=False)
            scroll = await es.scroll(scroll_id=sid, scroll="2m")
        print(f"Done. {updated} segments updated.")
    finally:
        await es.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=200)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    asyncio.run(_main(args.batch, args.dry_run))
