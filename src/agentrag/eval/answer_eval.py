"""
Answer quality evaluation — LLM-as-judge.

Đánh giá theo 4 chiều:
  faithfulness      (0-1) Các claim trong answer có được context hỗ trợ không?
                    Phát hiện hallucination.
  answer_relevance  (0-1) Answer có trả lời đúng câu hỏi không?
  context_precision (0-1) Context được retrieve có thực sự hữu ích cho câu trả lời không?
  correctness       (0-1) Answer có khớp với expected_answer không?
                    (chỉ tính khi có expected_answer)

LLM judge trả về JSON:
{
  "faithfulness": 0.9,
  "faithfulness_reason": "...",
  "answer_relevance": 0.8,
  "answer_relevance_reason": "...",
  "context_precision": 0.7,
  "context_precision_reason": "...",
  "correctness": 0.85,
  "correctness_reason": "..."
}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = """You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Score each dimension from 0.0 to 1.0.
Be strict: 1.0 means perfect, 0.0 means completely wrong/irrelevant.
Always return valid JSON with the exact keys specified."""

_JUDGE_TEMPLATE = """Evaluate this RAG response:

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

SYSTEM ANSWER: {answer}

EXPECTED ANSWER (ground truth, may be empty): {expected_answer}

Score each dimension (0.0 to 1.0):

1. faithfulness: Are ALL claims in the answer directly supported by the context?
   Penalize any claim not found in context (hallucination).

2. answer_relevance: Does the answer actually address what was asked?
   Penalize off-topic or evasive answers.

3. context_precision: Was the retrieved context actually useful for answering?
   Penalize if context is mostly irrelevant noise.

4. correctness: Does the answer match the expected answer semantically?
   Skip (set to null) if expected_answer is empty.

Return ONLY this JSON:
{{
  "faithfulness": <float 0-1>,
  "faithfulness_reason": "<one sentence>",
  "answer_relevance": <float 0-1>,
  "answer_relevance_reason": "<one sentence>",
  "context_precision": <float 0-1>,
  "context_precision_reason": "<one sentence>",
  "correctness": <float 0-1 or null>,
  "correctness_reason": "<one sentence or empty>"
}}"""


@dataclass
class AnswerScore:
    question_id: str
    question: str
    answer: str
    faithfulness: float
    faithfulness_reason: str
    answer_relevance: float
    answer_relevance_reason: str
    context_precision: float
    context_precision_reason: str
    correctness: float | None
    correctness_reason: str
    judge_latency_ms: float

    @property
    def avg_score(self) -> float:
        scores = [self.faithfulness, self.answer_relevance, self.context_precision]
        if self.correctness is not None:
            scores.append(self.correctness)
        return sum(scores) / len(scores)

    def as_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question[:100],
            "answer_preview": self.answer[:200],
            "faithfulness": round(self.faithfulness, 3),
            "faithfulness_reason": self.faithfulness_reason,
            "answer_relevance": round(self.answer_relevance, 3),
            "answer_relevance_reason": self.answer_relevance_reason,
            "context_precision": round(self.context_precision, 3),
            "context_precision_reason": self.context_precision_reason,
            "correctness": round(self.correctness, 3) if self.correctness is not None else None,
            "correctness_reason": self.correctness_reason,
            "avg_score": round(self.avg_score, 3),
            "judge_latency_ms": round(self.judge_latency_ms, 1),
        }


@dataclass
class AnswerEvalReport:
    n_questions: int
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_context_precision: float
    avg_correctness: float | None
    overall_avg: float
    hallucination_rate: float          # % with faithfulness < 0.5
    low_relevance_rate: float          # % with answer_relevance < 0.5
    scores: list[AnswerScore]

    def as_dict(self) -> dict:
        return {
            "n_questions": self.n_questions,
            "avg_faithfulness": round(self.avg_faithfulness, 3),
            "avg_answer_relevance": round(self.avg_answer_relevance, 3),
            "avg_context_precision": round(self.avg_context_precision, 3),
            "avg_correctness": round(self.avg_correctness, 3) if self.avg_correctness is not None else None,
            "overall_avg": round(self.overall_avg, 3),
            "hallucination_rate": round(self.hallucination_rate, 3),
            "low_relevance_rate": round(self.low_relevance_rate, 3),
            "per_question": [s.as_dict() for s in self.scores],
        }


def _format_context(packed_context: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(packed_context[:6]):  # cap ở 6 chunks
        excerpt = (chunk.get("excerpt") or chunk.get("content") or "")[:1500]
        section = chunk.get("section_path", "")
        parts.append(f"[{i+1}] ({section}): {excerpt}")
    return "\n".join(parts)


async def evaluate_answer(
    question_id: str,
    question: str,
    answer: str,
    packed_context: list[dict],
    expected_answer: str,
    llm_gateway,
) -> AnswerScore:
    import time
    context_text = _format_context(packed_context)
    user_prompt = _JUDGE_TEMPLATE.format(
        question=question,
        context=context_text,
        answer=answer,
        expected_answer=expected_answer or "(not provided)",
    )
    t0 = time.perf_counter()
    raw, _lat = await llm_gateway.json_response(
        system_prompt=_JUDGE_SYSTEM,
        user_prompt=user_prompt,
        task="eval_judge",
    )
    judge_latency = (time.perf_counter() - t0) * 1000

    faithfulness      = float(raw.get("faithfulness") or 0.0)
    answer_relevance  = float(raw.get("answer_relevance") or 0.0)
    context_precision = float(raw.get("context_precision") or 0.0)
    correctness_raw   = raw.get("correctness")
    correctness       = float(correctness_raw) if correctness_raw is not None else None

    return AnswerScore(
        question_id=question_id,
        question=question,
        answer=answer,
        faithfulness=min(max(faithfulness, 0.0), 1.0),
        faithfulness_reason=raw.get("faithfulness_reason", ""),
        answer_relevance=min(max(answer_relevance, 0.0), 1.0),
        answer_relevance_reason=raw.get("answer_relevance_reason", ""),
        context_precision=min(max(context_precision, 0.0), 1.0),
        context_precision_reason=raw.get("context_precision_reason", ""),
        correctness=correctness,
        correctness_reason=raw.get("correctness_reason", ""),
        judge_latency_ms=judge_latency,
    )


def aggregate_answer_scores(scores: list[AnswerScore]) -> AnswerEvalReport:
    n = len(scores)
    if n == 0:
        return AnswerEvalReport(
            n_questions=0, avg_faithfulness=0.0, avg_answer_relevance=0.0,
            avg_context_precision=0.0, avg_correctness=None, overall_avg=0.0,
            hallucination_rate=0.0, low_relevance_rate=0.0, scores=[],
        )

    avg_f  = sum(s.faithfulness for s in scores) / n
    avg_ar = sum(s.answer_relevance for s in scores) / n
    avg_cp = sum(s.context_precision for s in scores) / n

    correctness_scores = [s.correctness for s in scores if s.correctness is not None]
    avg_c = sum(correctness_scores) / len(correctness_scores) if correctness_scores else None

    overall = sum(s.avg_score for s in scores) / n
    hallucination_rate = sum(1 for s in scores if s.faithfulness < 0.5) / n
    low_rel_rate       = sum(1 for s in scores if s.answer_relevance < 0.5) / n

    return AnswerEvalReport(
        n_questions=n,
        avg_faithfulness=avg_f,
        avg_answer_relevance=avg_ar,
        avg_context_precision=avg_cp,
        avg_correctness=avg_c,
        overall_avg=overall,
        hallucination_rate=hallucination_rate,
        low_relevance_rate=low_rel_rate,
        scores=scores,
    )
