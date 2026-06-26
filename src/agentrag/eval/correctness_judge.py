"""Reference-based, phrasing-robust correctness judge.

Two scorers, ensembled:
  - nugget-recall: decompose GOLD into atomic must-have facts, score how many the
    answer covers, penalize only contradictions (extra true info is free).
  - reference-guided rubric: one anchored 0-1 judgement given Q + gold + gold context.

The pure aggregation functions below carry no LLM dependency and are unit-tested
directly. Async orchestration (Task 2) injects an LLM gateway.
"""
from __future__ import annotations

from dataclasses import dataclass


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def parse_nuggets(raw: dict) -> list[str]:
    """Extract a clean list of non-empty nugget strings from a judge payload."""
    items = raw.get("nuggets") or []
    return [n.strip() for n in items if isinstance(n, str) and n.strip()]


@dataclass
class NuggetScore:
    n_total: int
    n_covered: int
    n_contradicted: int
    recall: float
    contradiction_penalty: float
    score: float


def aggregate_nugget_labels(labels: list[str]) -> NuggetScore:
    """labels are per-nugget verdicts in {"covered","contradicted","absent"}.

    recall = covered/total ; penalty = contradicted/total ;
    score = max(0, recall - penalty).
    """
    total = len(labels)
    if total == 0:
        return NuggetScore(0, 0, 0, 0.0, 0.0, 0.0)
    covered = sum(1 for l in labels if l == "covered")
    contradicted = sum(1 for l in labels if l == "contradicted")
    recall = covered / total
    penalty = contradicted / total
    return NuggetScore(total, covered, contradicted, recall, penalty, clamp01(recall - penalty))


@dataclass
class EnsembleScore:
    nugget: float
    rubric: float
    mean: float
    abs_delta: float
    low_confidence: bool


def ensemble(nugget: float, rubric: float, *, delta_threshold: float = 0.2) -> EnsembleScore:
    nugget = clamp01(nugget)
    rubric = clamp01(rubric)
    abs_delta = abs(nugget - rubric)
    return EnsembleScore(
        nugget=nugget,
        rubric=rubric,
        mean=(nugget + rubric) / 2,
        abs_delta=abs_delta,
        low_confidence=abs_delta > delta_threshold,
    )
