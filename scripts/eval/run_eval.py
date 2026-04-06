"""
Chạy full evaluation pipeline: chunking → retrieval → answer quality.

Usage:
  # Đánh giá tất cả stages cho document achievement-system
  python scripts/eval/run_eval.py achievement-system

  # Chỉ đánh giá retrieval (nhanh hơn, không gọi LLM cho answer)
  python scripts/eval/run_eval.py achievement-system --stages chunking retrieval

  # Đánh giá với top-k khác
  python scripts/eval/run_eval.py achievement-system --top-k 10

  # Chỉ eval 5 câu hỏi đầu (debug nhanh)
  python scripts/eval/run_eval.py achievement-system --limit 5

Output: data/eval/<title>_eval_report.json + console table
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pam.config import settings
from src.pam.config_validation import validate_settings
from src.pam.eval.chunking_eval import evaluate_chunking
from src.pam.eval.dataset import GoldenDataset
from src.pam.eval.retrieval_eval import evaluate_retrieval_mode
from src.pam.eval.answer_eval import evaluate_answer, aggregate_answer_scores
from src.pam.retrieval.elasticsearch_retriever import ElasticsearchRetriever
from src.pam.agent.service import AgentService
from src.pam.services.llm_gateway import LLMGateway
from elasticsearch import AsyncElasticsearch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PAM evaluation suite")
    p.add_argument("document_title", help="Document title to evaluate (must be ingested)")
    p.add_argument(
        "--stages",
        nargs="+",
        choices=["chunking", "retrieval", "answer"],
        default=["chunking", "retrieval", "answer"],
        help="Stages to run (default: all)",
    )
    p.add_argument(
        "--dataset",
        help="Path to golden dataset JSON (default: data/eval/<title>.json)",
    )
    p.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval (default: 10)")
    p.add_argument(
        "--modes",
        nargs="+",
        default=["dense", "hybrid", "hybrid_kg"],
        help="Retrieval modes to test",
    )
    p.add_argument("--limit", type=int, help="Max questions to evaluate (for quick tests)")
    p.add_argument("--out", help="Output report path")
    return p.parse_args()


def _bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def _grade(score: float) -> str:
    if score >= 0.85:   return "✅ GOOD"
    if score >= 0.65:   return "⚠️  OK"
    return "❌ POOR"


def print_chunking_report(report) -> None:
    print("\n" + "="*60)
    print("  STAGE 1: CHUNKING QUALITY")
    print("="*60)
    d = report.as_dict()
    print(f"  Total chunks:       {d['total_chunks']}")
    print(f"  Short chunks (<{d['min_chars_threshold']}c): {d['short_chunk_count']} ({d['short_chunk_rate']:.1%})  {_grade(1 - d['short_chunk_rate'])}")
    print(f"  Avg length:         {d['avg_length_chars']:.0f} chars")
    print(f"  Median length:      {d['median_length_chars']:.0f} chars")
    print(f"  P95 length:         {d['p95_length_chars']:.0f} chars")
    print(f"  Unique sections:    {d['unique_sections']}")
    print(f"  Duplicate chunks:   {d['duplicate_chunks']} ({d['dedup_rate']:.1%})  {_grade(1 - d['dedup_rate'])}")


def print_retrieval_report(reports: list) -> None:
    print("\n" + "="*60)
    print("  STAGE 2: RETRIEVAL QUALITY")
    print("="*60)
    header = f"  {'Mode':<14} {'R@1':>5} {'R@3':>5} {'R@5':>5} {'MRR':>6} {'NDCG@5':>7} {'Lat(ms)':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in reports:
        d = r.as_dict()
        r1  = d["recall"].get("@1", 0)
        r3  = d["recall"].get("@3", 0)
        r5  = d["recall"].get("@5", 0)
        mrr = d["mrr"]
        ndcg5 = d["ndcg"].get("@5", 0)
        lat = d["avg_latency_ms"]
        grade = _grade(r5)
        print(f"  {d['mode']:<14} {r1:>5.2f} {r3:>5.2f} {r5:>5.2f} {mrr:>6.3f} {ndcg5:>7.3f} {lat:>7.1f}ms  {grade}")

    # Câu hỏi miss nhiều nhất
    if reports:
        worst = reports[0]  # dùng mode đầu tiên
        misses = [r for r in worst.query_results if not r.hit_at(5)]
        if misses:
            print(f"\n  ⚠️  Missed queries (mode={worst.mode}, not found in top-5):")
            for m in misses[:5]:
                print(f"     [{m.question_id}] {m.question[:70]}")


def print_answer_report(report) -> None:
    print("\n" + "="*60)
    print("  STAGE 3: ANSWER QUALITY (LLM-as-judge)")
    print("="*60)
    d = report.as_dict()
    print(f"  Questions evaluated: {d['n_questions']}")
    print(f"  Faithfulness:        {d['avg_faithfulness']:.3f}  {_bar(d['avg_faithfulness'])}  {_grade(d['avg_faithfulness'])}")
    print(f"  Answer Relevance:    {d['avg_answer_relevance']:.3f}  {_bar(d['avg_answer_relevance'])}  {_grade(d['avg_answer_relevance'])}")
    print(f"  Context Precision:   {d['avg_context_precision']:.3f}  {_bar(d['avg_context_precision'])}  {_grade(d['avg_context_precision'])}")
    if d["avg_correctness"] is not None:
        print(f"  Correctness:         {d['avg_correctness']:.3f}  {_bar(d['avg_correctness'])}  {_grade(d['avg_correctness'])}")
    print(f"  Overall:             {d['overall_avg']:.3f}  {_bar(d['overall_avg'])}")
    print(f"  Hallucination rate:  {d['hallucination_rate']:.1%}  {_grade(1 - d['hallucination_rate'])}")
    print(f"  Low relevance rate:  {d['low_relevance_rate']:.1%}  {_grade(1 - d['low_relevance_rate'])}")

    # Worst performing questions
    worst = sorted(d["per_question"], key=lambda x: x["avg_score"])[:3]
    if worst:
        print("\n  ⚠️  Lowest scoring questions:")
        for q in worst:
            print(f"     [{q['question_id']}] score={q['avg_score']:.2f} | {q['question'][:60]}")
            print(f"       → faith={q['faithfulness']:.2f} rel={q['answer_relevance']:.2f} ctx={q['context_precision']:.2f}")


async def main() -> None:
    args = parse_args()
    validate_settings(settings)

    doc = args.document_title
    stages = set(args.stages)
    report: dict = {
        "document_title": doc,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "embedding_model": settings.EMBEDDING_MODEL,
            "top_k": args.top_k,
        },
    }

    # ── Stage 1: Chunking ────────────────────────────────────────────────────
    if "chunking" in stages:
        es = AsyncElasticsearch([settings.ELASTICSEARCH_URL])
        try:
            chunking_result = await evaluate_chunking(
                es_client=es,
                index_name=settings.ELASTICSEARCH_INDEX_NAME,
                document_title=doc,
            )
            print_chunking_report(chunking_result)
            report["chunking"] = chunking_result.as_dict()
        finally:
            await es.close()

    # ── Load golden dataset ──────────────────────────────────────────────────
    dataset_path = Path(args.dataset) if args.dataset else ROOT / "data" / "eval" / f"{doc}.json"
    if not dataset_path.exists():
        if stages & {"retrieval", "answer"}:
            print(f"\n[INFO] Golden dataset not found at {dataset_path}")
            print(f"  Run first: python scripts/eval/generate_dataset.py {doc}")
            if "chunking" not in stages:
                sys.exit(1)
        dataset = None
    else:
        dataset = GoldenDataset.load(dataset_path)
        questions = dataset.questions
        if args.limit:
            questions = questions[: args.limit]
        print(f"\n[INFO] Loaded {len(questions)} golden questions from {dataset_path}")

    # ── Stage 2: Retrieval ───────────────────────────────────────────────────
    if "retrieval" in stages and dataset:
        retriever = ElasticsearchRetriever()
        try:
            retrieval_reports = []
            for mode in args.modes:
                print(f"  Evaluating retrieval mode: {mode}...")
                mode_report = await evaluate_retrieval_mode(
                    retriever=retriever,
                    questions=questions,
                    mode=mode,
                    top_k=args.top_k,
                    document_title=doc,
                )
                retrieval_reports.append(mode_report)

            print_retrieval_report(retrieval_reports)
            report["retrieval"] = [r.as_dict() for r in retrieval_reports]
        finally:
            await retriever.store.client.close()

    # ── Stage 3: Answer quality ──────────────────────────────────────────────
    if "answer" in stages and dataset:
        print(f"\n  Evaluating answer quality ({len(questions)} questions)...")
        agent = AgentService()
        llm = LLMGateway()
        answer_scores = []
        try:
            for i, q in enumerate(questions):
                print(f"  [{i+1}/{len(questions)}] {q.question[:60]}...", end="", flush=True)
                try:
                    result = await agent.chat(
                        question=q.question,
                        document_title=doc,
                    )
                    score = await evaluate_answer(
                        question_id=q.id,
                        question=q.question,
                        answer=result.get("answer", ""),
                        packed_context=result.get("context", []),
                        expected_answer=q.expected_answer,
                        llm_gateway=llm,
                    )
                    answer_scores.append(score)
                    print(f" → avg={score.avg_score:.2f}")
                except Exception as exc:
                    print(f" → ERROR: {exc}")
        finally:
            await agent.knowledge._tools.retriever.store.client.close()
            await agent.knowledge._tools.es_store.client.close()

        answer_report = aggregate_answer_scores(answer_scores)
        print_answer_report(answer_report)
        report["answer_quality"] = answer_report.as_dict()

    # ── Save report ──────────────────────────────────────────────────────────
    out_path = Path(args.out) if args.out else ROOT / "data" / "eval" / f"{doc}_eval_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "="*60)
    print(f"  Report saved → {out_path}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
