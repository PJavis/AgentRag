"""RMM-style citation-reward mining: the answer LLM's own inline [n] citations
label the rerank pool — cited passage = positive, retrieved-but-uncited =
hard negative. Output triplets feed scripts/finetune_reranker.py /
finetune_embedding.py unchanged. Pure functions; no I/O."""
from __future__ import annotations

from typing import Any

from src.agentrag.eval.probe_rows import parse_inline_citations


def feedback_to_row(
    *,
    question: str,
    answer: str,
    citations: list[dict[str, Any]] | None,
    rating: int | None,
) -> dict[str, Any] | None:
    """Convert a rated prod chat turn (ChatMessage.content + .citations,
    AdapterChatFeedback.rating) into a probe-row-shaped dict for mine_triplets.

    Prod has no judge score — the thumbs rating stands in: +1 → system_mean 1.0
    (passes the default min_system_mean=0.75 quality filter), else 0.0. Citations
    carry no rerank_score; mine_triplets' stable sort then keeps them in packed
    (relevance) order, so 'hardest uncited' = highest-ranked uncited passage."""
    if not question or not answer or not citations:
        return None
    ordered = sorted(citations, key=lambda c: c.get("source") or 0)
    packed = [
        {"content": c.get("excerpt") or c.get("content") or "", "rerank_score": None}
        for c in ordered
    ]
    return {
        "question": question,
        "system_answer": answer,
        "system_mean": 1.0 if rating == 1 else 0.0,
        "cited_sources": parse_inline_citations(answer),
        "packed": packed,
    }


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
        pos_texts = {p for p in positives if p}
        negatives = sorted(
            (item for i, item in enumerate(packed, start=1) if i not in cited),
            key=lambda c: float(c.get("rerank_score") or 0.0),
            reverse=True,
        )
        # Drop any negative whose text equals a positive — hybrid+RRF can merge the
        # same chunk into packed twice (one cited, one not), which would emit a
        # degenerate (q, X, X) triplet and train the reranker on contradictory
        # labels for the same pair.
        negatives = [c.get("content") or "" for c in negatives
                     if c.get("content") and c.get("content") not in pos_texts]
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
