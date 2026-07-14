"""Per-question probe rows — the raw material for miss bucketing and
citation mining. Pure functions only: no I/O, no LLM calls."""
from __future__ import annotations

import re
from typing import Any

from src.agentrag.eval.refusal import classify_refusal

# Inline citation markers the answer prompt mandates: "… [1]." / "… [1][2]."
# \[(\d{1,2})\] deliberately excludes markdown links ([text](url) has no digit-only body).
_CITE_RE = re.compile(r"\[(\d{1,2})\]")

_PACKED_FIELDS = ("content", "rerank_score", "document_title", "section_path")


def parse_inline_citations(answer: str | None) -> list[int]:
    """Source numbers the answer actually cited — the RMM 'used it' signal."""
    if not answer:
        return []
    return sorted({int(m) for m in _CITE_RE.findall(answer)})


def build_probe_row(
    *,
    qid: str,
    question: str,
    chat_out: dict[str, Any],
    oracle_answer: str,
    system_mean: float,
    oracle_mean: float,
    judge2_mean: float,
    gold_contexts: list[str],
) -> dict[str, Any]:
    answer = chat_out.get("answer") or ""
    citations = chat_out.get("citations") or []
    packed = [
        {k: item.get(k) for k in _PACKED_FIELDS}
        for item in (chat_out.get("context") or [])
    ]
    tool_queries = [
        (step.get("tool_input") or {}).get("query")
        for step in (chat_out.get("tool_trace") or [])
        if (step.get("tool_input") or {}).get("query")
    ]
    return {
        "qid": qid,
        "question": question,
        "system_answer": answer,
        "oracle_answer": oracle_answer,
        "system_mean": system_mean,
        "oracle_mean": oracle_mean,
        "judge2_mean": judge2_mean,
        "refusal_class": classify_refusal(answer, citations),
        "cited_sources": parse_inline_citations(answer),
        "packed": packed,
        "gold_contexts": list(gold_contexts),
        "tool_queries": tool_queries,
        "citations_count": len(citations),
    }
