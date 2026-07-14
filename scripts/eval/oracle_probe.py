#!/usr/bin/env python
"""Phase-1 oracle probe — prove the correctness cap is the ruler, not the system.

For each vn example: score the live system answer AND an oracle answer
(strong model + GOLD context = perfect retrieval) against gold, through the new
ensemble judge. Also re-score system answers with a SECOND judge model to get the
judge-noise floor. If oracle ≈ system, the cap is the gold/metric.

Run (no ingest — vn suite must already be indexed):
    uv run python scripts/eval/oracle_probe.py --n 20 --out docs/eval/eval_fidelity_probe_2026-06-26.md
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentrag.eval.benchmark_datasets import load_suite
from src.agentrag.eval.correctness_judge import score_correctness
from src.agentrag.eval.probe_rows import build_probe_row


_ORACLE_SYSTEM = (
    "Answer the question using ONLY the provided context. Be complete and precise. "
    "Do not add facts beyond the context. Answer in the context's language."
)


async def generate_oracle_answer(question: str, gold_context: str, gateway) -> str:
    """Strong-model answer given gold context — the achievable correctness ceiling."""
    user = f"CONTEXT:\n{gold_context}\n\nQUESTION: {question}"
    # own task slot so the operator can map oracle→strong model (LLM_TASK_MODEL_MAP) independently of the judge
    return await gateway.text_response(
        system_prompt=_ORACLE_SYSTEM, user_prompt=user, task="oracle_gen"
    )


@dataclass
class ProbeRow:
    qid: str
    system_mean: float
    oracle_mean: float
    judge2_mean: float
    detail: dict | None = None


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def summarize_probe(rows: list[ProbeRow]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "system_avg": 0.0, "oracle_avg": 0.0,
                "oracle_minus_system": 0.0, "judge_noise_pearson": 0.0}
    sys_avg = sum(r.system_mean for r in rows) / n
    ora_avg = sum(r.oracle_mean for r in rows) / n
    noise = pearson([r.system_mean for r in rows], [r.judge2_mean for r in rows])
    return {
        "n": n,
        "system_avg": sys_avg,
        "oracle_avg": ora_avg,
        "oracle_minus_system": ora_avg - sys_avg,
        "judge_noise_pearson": noise,
    }


def _render(summary: dict, label: str, n: int) -> str:
    return (
        f"# Eval-fidelity probe — {label} n={n}\n\n"
        f"- examples: {summary['n']}\n"
        f"- system avg (ensemble): {summary['system_avg']:.3f}\n"
        f"- oracle avg (strong model + gold context): {summary['oracle_avg']:.3f}\n"
        f"- **oracle − system: {summary['oracle_minus_system']:+.3f}**\n"
        f"- judge-noise floor (pearson, judge1 vs judge2): {summary['judge_noise_pearson']:.3f}\n\n"
        "## Read\n\n"
        "If oracle − system is small (< ~0.05), perfect retrieval + a strong generator "
        "barely beats the live system — the cap is the gold/metric, not the system. "
        "A low judge-noise pearson means the judge itself is unreliable and must be fixed "
        "before any correctness number is trusted.\n"
    )


async def main(args: argparse.Namespace) -> None:
    from src.agentrag.agent.factory import get_agent_service  # live system answers (same as run_benchmark)
    from src.agentrag.services.llm_gateway import LLMGateway

    gateway = LLMGateway()
    agent = get_agent_service()
    if args.eval_set:
        from src.agentrag.eval.benchmark_datasets import load_local_jsonl
        examples = load_local_jsonl(args.eval_set, n=args.n)
        print(f"[probe] {len(examples)} examples (eval-set={args.eval_set}, n={args.n})")
    else:
        examples = load_suite(args.suite, n=args.n)
        print(f"[probe] {len(examples)} examples (suite={args.suite}, n={args.n})")

    async def _score_one(ex) -> ProbeRow:
        gold_ctx = "\n".join(ex.gold_contexts)
        out = await agent.chat(question=ex.question, document_title=None, conversation_id=f"probe-{ex.id}")
        if out.get("timed_out"):
            # graceful agent timeout (total-budget) is not a scorable answer — retry/skip it
            raise RuntimeError("agent timed_out")
        system_ans = out.get("answer", "") or ""
        oracle_ans = await generate_oracle_answer(ex.question, gold_ctx, gateway)
        sys_e = await score_correctness(ex.question, system_ans, ex.reference_answer, gold_ctx, gateway)
        ora_e = await score_correctness(ex.question, oracle_ans, ex.reference_answer, gold_ctx, gateway)
        # judge2 routes through the eval_judge2 task slot — map LLM_TASK_MODEL_MAP.eval_judge2
        # to a DIFFERENT model to get a real cross-model noise floor; if it is unmapped,
        # judge2 falls back to the same model and the pearson is ~1.0 (self-consistency only).
        j2_e = await score_correctness(ex.question, system_ans, ex.reference_answer, gold_ctx, gateway, task="eval_judge2")
        detail = build_probe_row(
            qid=ex.id, question=ex.question, chat_out=out,
            oracle_answer=oracle_ans,
            system_mean=sys_e.mean, oracle_mean=ora_e.mean, judge2_mean=j2_e.mean,
            gold_contexts=ex.gold_contexts,
        )
        return ProbeRow(ex.id, sys_e.mean, ora_e.mean, j2_e.mean, detail)

    rows: list[ProbeRow] = []
    failures = 0
    for i, ex in enumerate(examples):
        row: ProbeRow | None = None
        for attempt in range(args.retries + 1):
            try:
                row = await _score_one(ex)
                break
            except Exception as exc:
                # Transient provider errors (gemini 503 high-demand, agent timeout) shouldn't
                # skip a question on first failure — retry with exponential backoff so a clean
                # high-n run isn't decimated by overload spikes. Skip only after exhausting.
                if attempt < args.retries:
                    await asyncio.sleep(args.retry_sleep * (2 ** attempt))
                    continue
                failures += 1
                print(f"  [{i+1}/{len(examples)}] ERR {type(exc).__name__}: skipped after {args.retries + 1} tries", flush=True)
        if row is None:
            continue
        rows.append(row)
        print(f"  [{i+1}/{len(examples)}] sys={row.system_mean:.2f} oracle={row.oracle_mean:.2f}", flush=True)

    if failures:
        print(f"[probe] {failures}/{len(examples)} questions skipped on provider errors", flush=True)

    summary = summarize_probe(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    label = args.eval_set if args.eval_set else args.suite
    out_path.write_text(_render(summary, label, args.n), encoding="utf-8")
    print(f"[probe] wrote {out_path}")
    if args.rows_out:
        import json
        rows_path = Path(args.rows_out)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        with rows_path.open("w", encoding="utf-8") as f:
            for r in rows:
                if r.detail:
                    f.write(json.dumps(r.detail, ensure_ascii=False) + "\n")
        print(f"[probe] wrote {len([r for r in rows if r.detail])} rows → {rows_path}")
    print(f"[probe] oracle−system = {summary['oracle_minus_system']:+.3f}, "
          f"judge-noise = {summary['judge_noise_pearson']:.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="vn", choices=["vn", "en", "both"])
    p.add_argument("--eval-set", default=None,
                   help="path to a local JSONL eval set (EvalExample shape); overrides --suite")
    p.add_argument("--n", type=int, default=20, help="examples per dataset")
    p.add_argument("--out", default="docs/eval/eval_fidelity_probe.md")
    p.add_argument("--retries", type=int, default=2,
                   help="retries per question on transient provider error / agent timeout")
    p.add_argument("--retry-sleep", type=float, default=8.0,
                   help="base backoff seconds between retries (exponential)")
    p.add_argument("--rows-out", default=None,
                   help="ALSO dump one JSON row per scored question (miss bucketing / citation mining input)")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
