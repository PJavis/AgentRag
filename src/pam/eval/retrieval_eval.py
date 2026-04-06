"""
Retrieval quality evaluation.

Metrics tính cho từng mode (sparse / dense / hybrid / hybrid_kg):
  recall@K        có ít nhất 1 relevant chunk trong top-K (K=1,3,5,10)
  mrr             Mean Reciprocal Rank
  ndcg@K          Normalized Discounted Cumulative Gain
  avg_latency_ms
  p95_latency_ms

Relevance được xác định theo 2 tiêu chí (OR logic):
  1. section_path khớp với relevant_sections của golden question
  2. content chứa bất kỳ relevant_keyword nào
"""
from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field

from .dataset import GoldenQuestion


@dataclass
class QueryResult:
    question_id: str
    question: str
    mode: str
    latency_ms: float
    results: list[dict]
    relevant_ranks: list[int]         # ranks của các kết quả relevant (1-indexed)
    first_relevant_rank: int | None

    def hit_at(self, k: int) -> bool:
        return any(r <= k for r in self.relevant_ranks)

    def reciprocal_rank(self) -> float:
        if self.first_relevant_rank is None:
            return 0.0
        return 1.0 / self.first_relevant_rank

    def ndcg_at(self, k: int) -> float:
        """Binary relevance NDCG@K."""
        gains = [1.0 if (i + 1) in self.relevant_ranks else 0.0 for i in range(min(k, len(self.results)))]
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        # Ideal: all relevant at top
        n_rel = len(self.relevant_ranks)
        ideal_gains = [1.0] * min(n_rel, k)
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
        return dcg / idcg if idcg > 0 else 0.0


@dataclass
class RetrievalModeReport:
    mode: str
    top_k: int
    n_questions: int
    recall_at: dict[int, float]       # {1: 0.6, 3: 0.8, 5: 0.9, 10: 1.0}
    mrr: float
    ndcg_at: dict[int, float]         # {5: 0.72, 10: 0.81}
    avg_latency_ms: float
    p95_latency_ms: float
    query_results: list[QueryResult] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "n_questions": self.n_questions,
            "recall": {f"@{k}": round(v, 4) for k, v in self.recall_at.items()},
            "mrr": round(self.mrr, 4),
            "ndcg": {f"@{k}": round(v, 4) for k, v in self.ndcg_at.items()},
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "per_query": [
                {
                    "id": r.question_id,
                    "question": r.question[:80],
                    "first_rank": r.first_relevant_rank,
                    "hit@5": r.hit_at(5),
                    "latency_ms": round(r.latency_ms, 1),
                }
                for r in self.query_results
            ],
        }


def _is_relevant(result: dict, question: GoldenQuestion) -> bool:
    """Kiểm tra kết quả retrieval có relevant với câu hỏi không."""
    section = (result.get("section_path") or "").lower()
    content = (result.get("content") or "").lower()

    # Khớp section — normalize spaces/underscores so "data models" matches "data_models"
    for s in question.relevant_sections:
        s_lo = s.lower()
        s_underscore = s_lo.replace(" ", "_")
        s_space = s_lo.replace("_", " ")
        if s_lo in section or s_underscore in section or s_space in section:
            return True

    # Khớp keyword
    for kw in question.relevant_keywords:
        if kw.lower() in content:
            return True

    return False


async def evaluate_retrieval_mode(
    retriever,
    questions: list[GoldenQuestion],
    mode: str,
    top_k: int = 10,
    document_title: str | None = None,
) -> RetrievalModeReport:
    k_values = [k for k in [1, 3, 5, 10] if k <= top_k]
    latencies: list[float] = []
    query_results: list[QueryResult] = []

    for q in questions:
        t0 = time.perf_counter()
        response = await retriever.search(
            query=q.question,
            mode=mode,
            top_k=top_k,
            document_title=document_title,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        results = response.get("results", [])
        relevant_ranks = [
            r["rank"] for r in results if _is_relevant(r, q)
        ]
        first_rank = min(relevant_ranks) if relevant_ranks else None

        query_results.append(
            QueryResult(
                question_id=q.id,
                question=q.question,
                mode=mode,
                latency_ms=latency_ms,
                results=results,
                relevant_ranks=relevant_ranks,
                first_relevant_rank=first_rank,
            )
        )

    n = len(questions)

    recall_at = {
        k: sum(1 for r in query_results if r.hit_at(k)) / n
        for k in k_values
    }
    mrr = sum(r.reciprocal_rank() for r in query_results) / n if n else 0.0
    ndcg_at = {k: sum(r.ndcg_at(k) for r in query_results) / n for k in k_values}

    sorted_lat = sorted(latencies)
    p95_idx = max(0, int(len(sorted_lat) * 0.95) - 1)

    return RetrievalModeReport(
        mode=mode,
        top_k=top_k,
        n_questions=n,
        recall_at=recall_at,
        mrr=mrr,
        ndcg_at=ndcg_at,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
        p95_latency_ms=float(sorted_lat[p95_idx]) if sorted_lat else 0.0,
        query_results=query_results,
    )
