"""Full RAG benchmark — the 9-metric eval over HF datasets, against OUR system.

Pipeline per suite:
  1. Load N examples (sailor2 Vietnamese_RAG / galileo-ai ragbench).
  2. Ingest the gold contexts into the index so our retrieval can find them.
  3. Run the agent on each question → answer + retrieved context, timing each.
  4. Score DeepEval metrics 1-5 (recall, precision, faithfulness, correctness, citation).
  5. Compute latency p50/p95/p99, cost/query (ledger), failure rate.
  6. Run the freshness check once.
  7. Emit a gated report (targets from the spec table).

Requires:  uv sync --extra deepeval
Env:       GEMINI_API_KEY, LLM_COST_TRACKING_ENABLED=true, an isolated
           ELASTICSEARCH_INDEX_NAME for the benchmark, live ES/PG/Valkey.

Usage:
  python scripts/eval/run_benchmark.py --suite vn --n 30
  python scripts/eval/run_benchmark.py --suite both --n 30 --out data/eval/benchmark.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentrag.config import settings
from src.agentrag.config_validation import validate_settings
from src.agentrag.eval.benchmark_datasets import load_suite
from src.agentrag.eval.deepeval_metrics import CaseInput, METRIC_TARGETS, build_judge, score_cases
from src.agentrag.eval.ragas_eval import extract_context_texts
from src.agentrag.eval.freshness import run_freshness_check
from src.agentrag.agent.factory import get_agent_service
from src.agentrag.observability.cost import recent_calls, reset_ledger

# Report-only / gated thresholds for the computed (non-LLM) metrics.
FAILURE_RATE_TARGET = 0.05


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return round(s[k], 1)


async def _ingest_gold(examples, group_size: int = 0) -> None:
    """Write gold contexts to a temp corpus and ingest them.

    group_size <= 0 writes one unique context per .md (legacy: 1 chunk/doc, so
    RAPTOR never builds and there are no cross-passage distractors). group_size
    > 0 merges that many unique contexts per doc, producing multi-chunk docs
    that activate RAPTOR and introduce distractors.
    """
    from src.agentrag.ingestion.pipeline import ingest_folder
    from src.agentrag.eval.refusal import group_contexts

    tmp = Path(tempfile.mkdtemp(prefix="bench_corpus_"))
    try:
        seen: set[str] = set()
        unique_contexts: list[str] = []
        for ex in examples:
            for ctx in ex.gold_contexts:
                h = hashlib.sha1(ctx.encode("utf-8")).hexdigest()[:16]
                if h in seen:
                    continue
                seen.add(h)
                unique_contexts.append(ctx)
        docs = group_contexts(unique_contexts, group_size)
        for i, doc in enumerate(docs):
            (tmp / f"{i}.md").write_text(doc, encoding="utf-8")
        print(
            f"[INFO] ingesting {len(docs)} docs from {len(unique_contexts)} unique "
            f"gold contexts (group_size={group_size})..."
        )
        await ingest_folder(str(tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def _run_refusal_eval(path: str) -> dict:
    """Out-of-corpus refusal eval: ask questions whose answers are NOT in the
    corpus and measure whether the system abstains (good) or hallucinates (bad).

    Assumes the normal corpus is already ingested so the agent retrieves
    *something* and must correctly judge it irrelevant. Returns the refusal_rate
    (fraction of cases the system correctly abstained on) plus per-case detail.
    """
    from src.agentrag.eval.refusal import classify_refusal

    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    agent = get_agent_service()
    per_case: list[dict] = []
    counts = {"abstained": 0, "hedged_cited": 0, "hallucinated": 0, "empty": 0}
    for c in cases:
        qid = str(c.get("id", ""))
        question = c.get("question", "") or ""
        try:
            out = await agent.chat(
                question=question, document_title=None, conversation_id=f"refusal-{qid}"
            )
        except Exception as exc:  # an error is not a refusal — count as empty
            per_case.append(
                {"id": qid, "question": question, "answer": f"ERR:{type(exc).__name__}",
                 "verdict": "empty"}
            )
            counts["empty"] += 1
            continue
        answer = out.get("answer", "") or ""
        verdict = classify_refusal(answer, out.get("citations"))
        counts[verdict] += 1
        per_case.append(
            {"id": qid, "question": question, "answer": answer[:200], "verdict": verdict}
        )

    n = len(cases) or 1
    # refusal_rate = cleanly abstained; hallucination_rate = the DANGEROUS failure
    # (confident answer to an out-of-corpus question). hedged_cited = safe-but-messy.
    return {
        "n": len(cases),
        "refusal_rate": round(counts["abstained"] / n, 3),
        "hedged_cited_rate": round(counts["hedged_cited"] / n, 3),
        "hallucination_rate": round(counts["hallucinated"] / n, 3),
        "counts": counts,
        "per_case": per_case,
    }


def _print_refusal_table(refusal: dict) -> None:
    print("\n" + "=" * 66)
    print(f"  OUT-OF-CORPUS REFUSAL EVAL — n={refusal['n']}")
    print("=" * 66)
    print(f"  refusal_rate       (clean abstain, ideal)     = {refusal['refusal_rate']:.3f}")
    print(f"  hedged_cited_rate  (hedged but cited, soft)   = {refusal['hedged_cited_rate']:.3f}")
    print(f"  hallucination_rate (confident, DANGEROUS)     = {refusal['hallucination_rate']:.3f}  ← keep ~0")
    for c in refusal["per_case"]:
        print(f"  [{c['verdict']:<12}] {c['question'][:46]:<46} → {c['answer'][:36]}")


async def main() -> None:
    p = argparse.ArgumentParser(description="9-metric RAG benchmark")
    p.add_argument("--suite", default="vn", choices=["vn", "en", "both"])
    p.add_argument("--n", type=int, default=30, help="examples per dataset")
    p.add_argument("--judge-provider", default="gemini", choices=["gemini", "deepseek", "openai"])
    p.add_argument("--judge-model", default=None)
    p.add_argument("--skip-ingest", action="store_true", help="reuse already-ingested corpus")
    p.add_argument("--skip-freshness", action="store_true")
    p.add_argument("--group-size", type=int, default=0, help="merge N gold contexts per ingested doc (0=one per doc; >0 activates RAPTOR/distractors)")
    p.add_argument("--refusal-set", help="path to a JSON list of out-of-corpus questions; runs an abstention eval after the main run")
    p.add_argument("--out", help="report path (default data/eval/benchmark_<suite>.json)")
    args = p.parse_args()
    validate_settings(settings)

    if not settings.LLM_COST_TRACKING_ENABLED:
        print("[WARN] LLM_COST_TRACKING_ENABLED is false → cost/query will be 0.")

    examples = load_suite(args.suite, n=args.n)
    print(f"[INFO] loaded {len(examples)} examples (suite={args.suite}, n={args.n})")

    if not args.skip_ingest:
        await _ingest_gold(examples, group_size=args.group_size)

    # ── Run the agent over every question ──
    reset_ledger()
    agent = get_agent_service()
    records = []
    failures = 0
    for i, ex in enumerate(examples):
        print(f"  [{i+1}/{len(examples)}] {ex.question[:55]}...", end="", flush=True)
        t0 = time.perf_counter()
        try:
            out = await agent.chat(question=ex.question, document_title=None, conversation_id=f"bench-{ex.id}")
            latency_ms = (time.perf_counter() - t0) * 1000
            answer = out.get("answer", "") or ""
            # packed_context items store text under content/excerpt/text — use the
            # robust extractor so retrieval_context is never silently empty.
            ctx = extract_context_texts(out.get("context"))
            failed = not answer.strip()
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            answer, ctx, failed = "", [], True
            print(f" ERR:{type(exc).__name__}", end="")
        if failed:
            failures += 1
        records.append({"ex": ex, "answer": answer, "ctx": ctx, "latency_ms": latency_ms, "failed": failed})
        print(f" {latency_ms:.0f}ms")

    # ── Computed metrics (6,7,8) ──
    latencies = [r["latency_ms"] for r in records]
    calls = recent_calls(limit=100000)
    total_usd = sum(float(c.get("usd", 0.0)) for c in calls)
    n = len(records) or 1
    latency = {"p50": _pct(latencies, 50), "p95": _pct(latencies, 95), "p99": _pct(latencies, 99)}
    cost_per_query = round(total_usd / n, 6)
    failure_rate = round(failures / n, 3)

    # ── LLM-judged metrics (1-5) ──
    print(f"\n[INFO] scoring {len(records)} cases with DeepEval (judge={args.judge_provider})...")
    # Pick the judge's API key by provider (deepseek falls back to OPENAI_API_KEY,
    # matching the runtime DeepSeek wiring). OpenAI judge reads OPENAI_API_KEY itself.
    judge_key = {
        "gemini": settings.GEMINI_API_KEY,
        "deepseek": settings.DEEPSEEK_API_KEY or settings.OPENAI_API_KEY,
        "openai": settings.OPENAI_API_KEY,
    }.get(args.judge_provider, settings.GEMINI_API_KEY)
    judge = build_judge(provider=args.judge_provider, model=args.judge_model, api_key=judge_key)
    cases = [
        CaseInput(
            question=r["ex"].question,
            actual_output=r["answer"],
            expected_output=r["ex"].reference_answer,
            retrieval_context=r["ctx"],
        )
        for r in records
    ]
    judged = score_cases(cases, judge)

    # ── Freshness (9) ──
    freshness = {"skipped": True}
    if not args.skip_freshness:
        print("[INFO] running freshness check...")
        try:
            freshness = await run_freshness_check()
        except Exception as exc:
            freshness = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}

    report = {
        "suite": args.suite,
        "n_examples": len(records),
        "judge": {"provider": args.judge_provider, "model": args.judge_model or "default"},
        "metrics": {
            **judged["summary"],
            "latency_ms": latency,
            "cost_per_query_usd": cost_per_query,
            "failure_rate": {"value": failure_rate, "target": FAILURE_RATE_TARGET, "meets_target": failure_rate < FAILURE_RATE_TARGET},
            "freshness": freshness,
        },
        "per_case": judged["per_case"],
    }

    # ── Print gated table ──
    print("\n" + "=" * 66)
    print(f"  RAG BENCHMARK — suite={args.suite}  n={len(records)}")
    print("=" * 66)
    for name in METRIC_TARGETS:
        m = judged["summary"][name]
        if m["mean"] is None:
            print(f"  {name:<22} n/a (all errored)")
        else:
            flag = "PASS" if m["meets_target"] else "FAIL"
            print(f"  {name:<22} {m['mean']:.3f}  (target ≥{m['target']:.2f})  {flag}  pass_rate={m['pass_rate']:.2f}")
    print(f"  {'latency p50/p95/p99':<22} {latency['p50']:.0f}/{latency['p95']:.0f}/{latency['p99']:.0f} ms  (report)")
    print(f"  {'cost_per_query_usd':<22} ${cost_per_query:.6f}  (report)")
    fr = report["metrics"]["failure_rate"]
    print(f"  {'failure_rate':<22} {fr['value']:.3f}  (target <{fr['target']:.2f})  {'PASS' if fr['meets_target'] else 'FAIL'}")
    print(f"  {'freshness':<22} {'PASS' if freshness.get('pass') else ('SKIP' if freshness.get('skipped') else 'FAIL')}  {freshness.get('detail','')}")

    # ── Out-of-corpus refusal eval (optional) ──
    # The suite corpus is already ingested above, so the agent retrieves
    # *something* and must correctly find it irrelevant for these questions.
    if args.refusal_set:
        print(f"\n[INFO] running refusal eval from {args.refusal_set}...")
        try:
            refusal = await _run_refusal_eval(args.refusal_set)
            report["refusal"] = refusal
            _print_refusal_table(refusal)
        except Exception as exc:
            report["refusal"] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[WARN] refusal eval failed: {type(exc).__name__}: {exc}")

    out_path = Path(args.out) if args.out else ROOT / "data" / "eval" / f"benchmark_{args.suite}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # EvalExample objects aren't JSON-serializable; per_case already plain.
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] report → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
