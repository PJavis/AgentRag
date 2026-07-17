"""Corpus fingerprint — bind an eval set to the corpus snapshot it was built from.

The 2026-07-13 landmine: `prod_corpus_evalset_v3.jsonl` was generated from the
2026-06 residue corpus and silently scored sys=0.00 on every question against the
real corpus. Rule: an eval set is only valid against the corpus snapshot it was
generated from. This module enforces it — `build_prod_evalset.py` stamps every
row with `corpus_fp`; `oracle_probe.py` recomputes the live fingerprint and
refuses to run on a mismatch.

Fingerprint = sha1 over the sorted (document_title, segment_count) pairs —
changes when documents are added/removed or a re-ingest changes segmentation,
stable across row order and unrelated DB churn."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def fingerprint_from_pairs(pairs: list[tuple[str, int]]) -> str:
    lines = sorted(f"{title}:{count}" for title, count in pairs)
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()[:12]


async def compute_corpus_fingerprint() -> str:
    """Live fingerprint from Postgres (same source build_prod_evalset samples)."""
    from sqlalchemy import func, select

    from src.agentrag.database import AsyncSessionLocal
    from src.agentrag.database.models import Document, Segment

    async with AsyncSessionLocal() as s:
        rows = (
            await s.execute(
                select(Document.title, func.count(Segment.id))
                .join(Segment, Segment.document_id == Document.id)
                .group_by(Document.title)
            )
        ).all()
    return fingerprint_from_pairs([(title or "", count) for title, count in rows])


def read_evalset_fingerprint(path: str) -> str | None:
    """corpus_fp of the first row of a JSONL eval set (all rows carry the same
    stamp); None for pre-fingerprint sets."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line).get("corpus_fp")
        except json.JSONDecodeError:
            return None
    return None


def check_fingerprint(
    *, evalset_fp: str | None, live_fp: str, allow_mismatch: bool
) -> tuple[bool, str]:
    """(ok, message). ok=False means: do not run, numbers would be garbage."""
    if evalset_fp is None:
        return True, ("eval set has NO corpus fingerprint (built pre-guard) — "
                      "cannot verify it matches the ingested corpus")
    if evalset_fp == live_fp:
        return True, ""
    msg = (f"corpus fingerprint MISMATCH: eval set was built from corpus {evalset_fp}, "
           f"live corpus is {live_fp} — questions won't match the ingested documents "
           f"(the v3-landmine failure mode). Rebuild the eval set or pass "
           f"--allow-corpus-mismatch to override.")
    if allow_mismatch:
        return True, f"OVERRIDDEN {msg}"
    return False, msg
