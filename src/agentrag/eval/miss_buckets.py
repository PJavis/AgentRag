"""Classify probe-row misses into actionable failure buckets.

Buckets:
    false_abstention — refused an answerable question (floor/gate territory)
    retrieval_miss   — gold passage never reached the answer LLM (retrieval/graph work)
    generation_miss  — gold was packed, answer still wrong (prompt/model work)

Pure functions over Task-1 probe rows; no I/O."""
from __future__ import annotations

from typing import Any

_REFUSED = ("abstained", "hedged_cited", "empty")
JUDGE_GAP = 0.4  # |system_mean - judge2_mean| at/above this → judge-disagreement flag


def _words(text: str) -> set[str]:
    return set((text or "").lower().split())


def gold_overlap(packed: list[dict[str, Any]], gold_contexts: list[str]) -> float:
    """Best Jaccard word overlap between any packed passage and any gold context.
    Proxy for 'did the gold chunk reach the answer LLM' — exact hashes are not
    available in probe rows (the eval set stores raw text, not segment ids)."""
    best = 0.0
    for item in packed or []:
        pw = _words(item.get("content") or "")
        if not pw:
            continue
        for gold in gold_contexts or []:
            gw = _words(gold)
            if not gw:
                continue
            inter = len(pw & gw)
            union = len(pw | gw)
            if union:
                best = max(best, inter / union)
    return best


def bucket_row(
    row: dict[str, Any],
    *,
    miss_threshold: float = 0.5,
    overlap_threshold: float = 0.35,
) -> str | None:
    if float(row.get("system_mean", 0.0)) >= miss_threshold:
        return None
    if row.get("refusal_class") in _REFUSED:
        return "false_abstention"
    if gold_overlap(row.get("packed") or [], row.get("gold_contexts") or []) < overlap_threshold:
        return "retrieval_miss"
    return "generation_miss"


def summarize_buckets(rows: list[dict[str, Any]], **kw) -> dict[str, Any]:
    buckets: dict[str, int] = {}
    misses = 0
    judge_gap_rows: list[str] = []
    for row in rows:
        b = bucket_row(row, **kw)
        if b:
            misses += 1
            buckets[b] = buckets.get(b, 0) + 1
        if abs(float(row.get("system_mean", 0.0)) - float(row.get("judge2_mean", 0.0))) >= JUDGE_GAP:
            judge_gap_rows.append(row.get("qid", "?"))
    return {"n": len(rows), "misses": misses, "buckets": buckets,
            "judge_gap_rows": judge_gap_rows}


def render_report(rows: list[dict[str, Any]], summary: dict[str, Any], label: str) -> str:
    lines = [
        f"# Miss buckets — {label}",
        "",
        f"- rows scored: {summary['n']}",
        f"- misses (system_mean < 0.5): {summary['misses']}",
    ]
    for name, count in sorted(summary["buckets"].items()):
        lines.append(f"- **{name}**: {count}")
    if summary["judge_gap_rows"]:
        lines.append(f"- judge-disagreement rows (|sys−judge2| ≥ {JUDGE_GAP}): "
                     + ", ".join(summary["judge_gap_rows"]))
    lines += ["", "## Miss detail", ""]
    for row in rows:
        b = bucket_row(row)
        if not b:
            continue
        best = gold_overlap(row.get("packed") or [], row.get("gold_contexts") or [])
        scores = [c.get("rerank_score") for c in (row.get("packed") or [])
                  if c.get("rerank_score") is not None]
        lines += [
            f"### {row['qid']} — `{b}`",
            "",
            f"- Q: {row.get('question', '')[:200]}",
            f"- sys={row.get('system_mean')} oracle={row.get('oracle_mean')} "
            f"judge2={row.get('judge2_mean')} refusal={row.get('refusal_class')}",
            f"- gold_overlap={best:.2f} max_rerank={max(scores) if scores else None} "
            f"cited={row.get('cited_sources')}",
            f"- tool_queries: {row.get('tool_queries')}",
            f"- answer: {row.get('system_answer', '')[:300]}",
            "",
        ]
    return "\n".join(lines) + "\n"
