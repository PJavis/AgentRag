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
    """True when the system correctly abstained: it signalled uncertainty AND
    produced no citations. A confident claim with no citations is a
    hallucination, not an abstention, so it returns False."""
    return _has_uncertainty(answer or "") and not (citations or [])
