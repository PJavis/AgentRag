"""Reference-based, phrasing-robust correctness judge.

Two scorers, ensembled:
  - nugget-recall: decompose GOLD into atomic must-have facts, score how many the
    answer covers, penalize only contradictions (extra true info is free).
  - reference-guided rubric: one anchored 0-1 judgement given Q + gold + gold context.

The pure aggregation functions below carry no LLM dependency and are unit-tested
directly. Async orchestration (Task 2) injects an LLM gateway.
"""
from __future__ import annotations

import json
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


_NUGGET_EXTRACT_SYSTEM = (
    "You extract atomic factual claims from a reference answer. Break the "
    "reference into the smallest standalone must-have facts a correct answer "
    "must convey. Do not invent facts not present. Same language as the input. "
    'Return strict JSON: {"nuggets": ["fact 1", "fact 2", ...]}'
)

_NUGGET_SCORE_SYSTEM = (
    "You compare a candidate answer against a list of required facts (nuggets). "
    "For each nugget, output exactly one label:\n"
    "  covered      - the answer states this fact (any phrasing).\n"
    "  contradicted - the answer states something that conflicts with this fact.\n"
    "  absent       - the answer neither states nor contradicts it.\n"
    "Extra correct information in the answer is fine and must NOT be penalized. "
    'Return strict JSON: {"labels": ["covered", "absent", ...]} in nugget order.'
)

_RUBRIC_SYSTEM = (
    "You are a strict grader. Given a question, a reference (gold) answer, the "
    "gold context, and a candidate answer, rate how correctly and completely the "
    "candidate answers the question. Credit valid paraphrase and extra correct "
    "facts; penalize only wrong, missing, or contradictory content.\n"
    "Anchors: 1.0 fully correct & complete; 0.7 correct, minor gap; 0.4 partially "
    "correct; 0.0 wrong or contradicts the gold.\n"
    'Return strict JSON: {"score": <float 0-1>, "reason": "<one sentence>"}'
)


async def extract_nuggets(gold: str, gateway, *, task: str = "eval_judge") -> list[str]:
    raw, _ = await gateway.json_response(
        system_prompt=_NUGGET_EXTRACT_SYSTEM,
        user_prompt=gold,
        task=task,
    )
    return parse_nuggets(raw)


async def score_nuggets(question: str, answer: str, nuggets: list[str], gateway, *, task: str = "eval_judge") -> NuggetScore:
    if not nuggets:
        return NuggetScore(0, 0, 0, 0.0, 0.0, 0.0)
    user = json.dumps(
        {"question": question, "answer": answer, "nuggets": nuggets},
        ensure_ascii=False,
    )
    raw, _ = await gateway.json_response(
        system_prompt=_NUGGET_SCORE_SYSTEM, user_prompt=user, task=task
    )
    labels = [l for l in (raw.get("labels") or []) if isinstance(l, str)]
    return aggregate_nugget_labels(labels)


async def score_rubric(question: str, answer: str, gold: str, gold_context: str, gateway, *, task: str = "eval_judge") -> float:
    user = json.dumps(
        {"question": question, "gold_answer": gold, "gold_context": gold_context[:4000], "answer": answer},
        ensure_ascii=False,
    )
    raw, _ = await gateway.json_response(
        system_prompt=_RUBRIC_SYSTEM, user_prompt=user, task=task
    )
    return clamp01(raw.get("score") or 0.0)


async def score_correctness(question: str, answer: str, gold: str, gold_context: str, gateway, *, task: str = "eval_judge") -> EnsembleScore:
    nuggets = await extract_nuggets(gold, gateway, task=task)
    ns = await score_nuggets(question, answer, nuggets, gateway, task=task)
    rubric = await score_rubric(question, answer, gold, gold_context, gateway, task=task)
    return ensemble(ns.score, rubric)
