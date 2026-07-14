"""RMM-style citation-reward mining: the answer LLM's own inline [n] citations
label the rerank pool — cited passage = positive, retrieved-but-uncited =
hard negative. Output triplets feed scripts/finetune_reranker.py /
finetune_embedding.py unchanged. Pure functions; no I/O."""
from __future__ import annotations

from typing import Any


def mine_triplets(
    rows: list[dict[str, Any]],
    *,
    min_system_mean: float = 0.75,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if float(row.get("system_mean", 0.0)) < min_system_mean:
            continue
        packed = row.get("packed") or []
        cited = {n for n in (row.get("cited_sources") or []) if 1 <= n <= len(packed)}
        if not cited:
            continue
        positives = [packed[n - 1].get("content") or "" for n in sorted(cited)]
        negatives = sorted(
            (item for i, item in enumerate(packed, start=1) if i not in cited),
            key=lambda c: float(c.get("rerank_score") or 0.0),
            reverse=True,
        )
        negatives = [c.get("content") or "" for c in negatives if c.get("content")]
        if not negatives:
            continue
        query = row.get("question") or ""
        for i, pos in enumerate(p for p in positives if p):
            out.append({
                "query": query,
                "positive": pos,
                "negative": negatives[i % len(negatives)],
                "source": "citation",
            })
    return out
