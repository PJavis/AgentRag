"""Reindex legacy StructMem indices into the unified R4 layout.

Before R4 there were four ES indices:
    agentrag_entries          → kind="entry"   (doc memory)
    agentrag_synthesis        → kind="synthesis" (doc memory)
    agentrag_chat_entries     → kind="entry"   (chat memory)
    agentrag_chat_synthesis   → kind="synthesis" (chat memory)

After R4 they collapse into two physical indices:
    agentrag_memory_doc       — kind ∈ {entry, synthesis}
    agentrag_memory_chat      — kind ∈ {entry, synthesis}

This script uses ES `_reindex` to copy every doc from the legacy indices
into the unified targets, stamping the correct `kind`. Run BEFORE deleting
the old indices. After verifying counts, remove the old indices manually.

Usage:
    uv run python scripts/migrate_structmem_indices.py
    # Optional:
    #   --dry-run      print counts only, no writes
    #   --drop-legacy  delete the old indices after successful reindex
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from elasticsearch import AsyncElasticsearch
from elasticsearch import NotFoundError as ESNotFoundError

from src.agentrag.config import settings

LEGACY_DOC = [
    ("agentrag_entries", "entry"),
    ("agentrag_synthesis", "synthesis"),
]
LEGACY_CHAT = [
    ("agentrag_chat_entries", "entry"),
    ("agentrag_chat_synthesis", "synthesis"),
]
TARGET_DOC = settings.STRUCTMEM_INDEX
TARGET_CHAT = settings.CHAT_MEMORY_INDEX


async def _count(es: AsyncElasticsearch, index: str) -> int:
    try:
        resp = await es.count(index=index)
        return int(resp.get("count", 0))
    except ESNotFoundError:
        return 0


async def _reindex_with_kind(
    es: AsyncElasticsearch, source: str, target: str, kind: str, dry_run: bool
) -> int:
    count = await _count(es, source)
    if count == 0:
        print(f"  {source:30s} → {target}: empty, skip")
        return 0
    print(f"  {source:30s} → {target}: {count} docs (kind={kind})")
    if dry_run:
        return count
    body = {
        "source": {"index": source},
        "dest": {"index": target},
        "script": {
            "source": f"ctx._source.kind = '{kind}'",
            "lang": "painless",
        },
    }
    resp = await es.reindex(body=body, refresh=True, wait_for_completion=True)
    moved = resp.get("total", 0)
    print(f"    moved {moved} docs")
    return moved


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--drop-legacy", action="store_true")
    args = parser.parse_args()

    es = AsyncElasticsearch([settings.ELASTICSEARCH_URL])

    try:
        print(f"== Doc memory → {TARGET_DOC} ==")
        for src, kind in LEGACY_DOC:
            await _reindex_with_kind(es, src, TARGET_DOC, kind, args.dry_run)

        print(f"\n== Chat memory → {TARGET_CHAT} ==")
        for src, kind in LEGACY_CHAT:
            await _reindex_with_kind(es, src, TARGET_CHAT, kind, args.dry_run)

        if args.drop_legacy and not args.dry_run:
            print("\n== Dropping legacy indices ==")
            for src, _ in LEGACY_DOC + LEGACY_CHAT:
                try:
                    await es.indices.delete(index=src)
                    print(f"  deleted {src}")
                except ESNotFoundError:
                    print(f"  {src} already absent")

        if args.dry_run:
            print("\n(dry-run — no writes performed)")
        else:
            print("\nDone. Verify counts on the unified indices, then re-run with --drop-legacy when satisfied.")
    finally:
        await es.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
