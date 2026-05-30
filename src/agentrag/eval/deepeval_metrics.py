"""DeepEval metric harness — the 5 LLM-judged metrics from the eval spec.

| # | Metric              | DeepEval class             | Target |
|---|---------------------|----------------------------|--------|
| 1 | Retrieval recall@k  | ContextualRecallMetric     | 0.70   |
| 2 | Context precision   | ContextualPrecisionMetric  | 0.70   |
| 3 | Faithfulness        | FaithfulnessMetric         | 0.80   |
| 4 | Answer correctness  | GEval vs expected_output   | 0.70   |
| 5 | Citation accuracy   | GEval citations vs context | 0.70   |

Judge defaults to Gemini 2.5 Flash (cheap, multilingual — good for VN). Swap to
DeepSeek later via build_judge(provider="deepseek").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Targets from the spec table (metric name → threshold).
METRIC_TARGETS: dict[str, float] = {
    "contextual_recall": 0.70,
    "contextual_precision": 0.70,
    "faithfulness": 0.80,
    "answer_correctness": 0.70,
    "citation_accuracy": 0.70,
}


@dataclass
class CaseInput:
    """Everything needed to build one DeepEval LLMTestCase."""
    question: str
    actual_output: str
    expected_output: str
    retrieval_context: list[str]


def build_judge(provider: str = "gemini", model: str | None = None, api_key: str | None = None):
    """Return a DeepEval judge model. Gemini now; DeepSeek/OpenAI selectable."""
    if provider == "gemini":
        from deepeval.models import GeminiModel

        return GeminiModel(model=model or "gemini-2.5-flash", api_key=api_key, temperature=0.0)
    if provider == "deepseek":
        from deepeval.models import DeepSeekModel

        return DeepSeekModel(model=model or "deepseek-chat", api_key=api_key, temperature=0.0)
    if provider == "openai":
        from deepeval.models import GPTModel

        return GPTModel(model=model or "gpt-4o-mini")
    raise ValueError(f"unknown judge provider: {provider}")


def build_metrics(judge) -> list[tuple[str, Any]]:
    """Instantiate the 5 metrics bound to the judge. Returns (name, metric)."""
    from deepeval.metrics import (
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.test_case import LLMTestCaseParams

    correctness = GEval(
        name="answer_correctness",
        criteria=(
            "Determine whether the actual output is factually and substantively "
            "correct compared to the expected output. Penalize missing key facts, "
            "added wrong facts, and contradictions. Minor wording differences are fine."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=judge,
        threshold=METRIC_TARGETS["answer_correctness"],
    )
    citation = GEval(
        name="citation_accuracy",
        criteria=(
            "Every claim in the actual output that carries a citation marker like "
            "[1], [2] must be supported by the corresponding passage in the "
            "retrieval context. Penalize citations that point to passages which do "
            "not support the cited claim, and unsupported claims presented as cited."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        model=judge,
        threshold=METRIC_TARGETS["citation_accuracy"],
    )
    return [
        ("contextual_recall", ContextualRecallMetric(threshold=METRIC_TARGETS["contextual_recall"], model=judge)),
        ("contextual_precision", ContextualPrecisionMetric(threshold=METRIC_TARGETS["contextual_precision"], model=judge)),
        ("faithfulness", FaithfulnessMetric(threshold=METRIC_TARGETS["faithfulness"], model=judge)),
        ("answer_correctness", correctness),
        ("citation_accuracy", citation),
    ]


def score_cases(cases: list[CaseInput], judge) -> dict[str, Any]:
    """Run all metrics over all cases. Returns per-metric mean + pass-rate.

    A metric that errors on a case (judge JSON failure, etc.) contributes None
    and is excluded from that metric's mean — never crashes the whole run.
    """
    from deepeval.test_case import LLMTestCase

    metrics = build_metrics(judge)
    per_metric_scores: dict[str, list[float]] = {name: [] for name, _ in metrics}
    per_case: list[dict[str, Any]] = []

    for c in cases:
        tc = LLMTestCase(
            input=c.question,
            actual_output=c.actual_output or " ",
            expected_output=c.expected_output,
            retrieval_context=c.retrieval_context or [" "],
        )
        row: dict[str, Any] = {"question": c.question[:80]}
        for name, metric in metrics:
            try:
                metric.measure(tc)
                score = float(metric.score)
                per_metric_scores[name].append(score)
                row[name] = round(score, 3)
            except Exception as exc:  # judge failure on this case
                row[name] = f"ERR:{type(exc).__name__}"
        per_case.append(row)

    summary: dict[str, Any] = {}
    for name, scores in per_metric_scores.items():
        target = METRIC_TARGETS[name]
        if scores:
            mean = sum(scores) / len(scores)
            pass_rate = sum(1 for s in scores if s >= target) / len(scores)
            summary[name] = {
                "mean": round(mean, 3),
                "pass_rate": round(pass_rate, 3),
                "target": target,
                "n": len(scores),
                "meets_target": mean >= target,
            }
        else:
            summary[name] = {"mean": None, "target": target, "n": 0, "meets_target": False}
    return {"summary": summary, "per_case": per_case}
