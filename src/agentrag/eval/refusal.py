from __future__ import annotations

from typing import Any

from src.agentrag.agent.service import _has_uncertainty


def group_contexts(contexts: list[str], group_size: int) -> list[str]:
    """Merge gold contexts into multi-passage documents so ingest produces
    multi-chunk docs (activates RAPTOR + cross-passage distractors). group_size
    <= 0 keeps the legacy one-context-per-doc behavior."""
    if group_size <= 0:
        return list(contexts)
    out: list[str] = []
    for i in range(0, len(contexts), group_size):
        bucket = contexts[i:i + group_size]
        out.append("\n\n---\n\n".join(bucket))
    return out


def is_abstention(answer: str, citations: list[Any] | None) -> bool:
    """True when the system cleanly abstained: it signalled uncertainty AND
    produced no citations. A confident claim with no citations is a
    hallucination, not an abstention, so it returns False."""
    return _has_uncertainty(answer or "") and not (citations or [])


def is_hallucination(answer: str) -> bool:
    """The DANGEROUS failure on an out-of-corpus question: a non-empty, confident
    answer with NO uncertainty hedge. (Hedging but citing a tangential distractor
    is a soft issue, not a hallucination — see classify_refusal.)"""
    a = (answer or "").strip()
    return bool(a) and not _has_uncertainty(a)


def classify_refusal(answer: str, citations: list[Any] | None) -> str:
    """Three-way classify an out-of-corpus response:
    - "abstained"     — hedged + no citation (ideal: safe + clean)
    - "hedged_cited"  — hedged but cited a (likely distractor) source (safe but messy)
    - "hallucinated"  — confident answer, no hedge (DANGEROUS)
    - "empty"         — no answer / error
    """
    a = (answer or "").strip()
    if not a:
        return "empty"
    if _has_uncertainty(a):
        return "abstained" if not (citations or []) else "hedged_cited"
    return "hallucinated"
