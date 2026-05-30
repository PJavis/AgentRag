"""RAGAS eval — STEP 2 of 2: score rows (runs in an ISOLATED venv).

Self-contained: imports only ragas + langchain-openai (no `src` import), so it
runs in an isolated environment that pulls RAGAS's old langchain stack
(`langchain-core <0.3`) without colliding with the app's `langchain-core 1.x`.

Run it via an ephemeral env:
  GEMINI_API_KEY=... uv run --no-project \
      --with "ragas>=0.2,<0.3" --with "langchain-openai<0.3" \
      python scripts/eval/score_ragas.py data/eval/<title>_ragas_rows.json

Judge model defaults to Gemini Flash; override with --judge-model / env keys.
Metrics default to faithfulness + context_precision + context_recall (LLM-only).
Add --with-relevancy to also run answer_relevancy — needs an embeddings model.
NOTE: Gemini's OpenAI-compatible /embeddings endpoint returns 501 UNIMPLEMENTED,
so answer_relevancy needs `--embedding-provider openai` (OpenAI embeddings).
Failed metrics are reported as FAILED and omitted from the JSON report.

Output: <rows>.scored.json + console table.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _grade(score: float) -> str:
    if score >= 0.85:
        return "GOOD"
    if score >= 0.65:
        return "OK"
    return "POOR"


def _build_judge(provider: str, model: str):
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("GEMINI_API_KEY env var required for --judge-provider=gemini")
        chat = ChatOpenAI(model=model, api_key=key, base_url=_GEMINI_OPENAI_BASE, temperature=0.0)
    else:  # openai
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("OPENAI_API_KEY env var required for --judge-provider=openai")
        chat = ChatOpenAI(model=model, api_key=key, temperature=0.0)
    return LangchainLLMWrapper(chat)


def _build_embeddings(provider: str, model: str):
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        emb = OpenAIEmbeddings(model=model, api_key=key, base_url=_GEMINI_OPENAI_BASE)
    else:
        key = os.environ.get("OPENAI_API_KEY")
        emb = OpenAIEmbeddings(model=model, api_key=key)
    return LangchainEmbeddingsWrapper(emb)


def main() -> None:
    p = argparse.ArgumentParser(description="RAGAS step 2 — score eval rows")
    p.add_argument("rows", help="Path to *_ragas_rows.json from run_ragas.py")
    p.add_argument("--judge-provider", default="gemini", choices=["gemini", "openai"])
    p.add_argument("--judge-model", default="gemini-2.5-flash")
    p.add_argument("--embedding-provider", default="gemini", choices=["gemini", "openai"])
    p.add_argument("--embedding-model", default="gemini-embedding-001")
    p.add_argument("--with-relevancy", action="store_true", help="Also run answer_relevancy (uses embeddings)")
    p.add_argument("--out", help="Output report path (default: <rows>.scored.json)")
    args = p.parse_args()

    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    rows = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    if not rows:
        sys.exit("No rows in input.")

    metrics = [faithfulness, context_precision, context_recall]
    embeddings = None
    if args.with_relevancy:
        metrics.append(answer_relevancy)
        embeddings = _build_embeddings(args.embedding_provider, args.embedding_model)

    judge = _build_judge(args.judge_provider, args.judge_model)
    print(f"[INFO] Scoring {len(rows)} rows | judge={args.judge_model} | metrics={[m.name for m in metrics]}")

    result = evaluate(
        dataset=EvaluationDataset.from_list(rows),
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
    )
    df = result.to_pandas()
    scores: dict[str, float] = {}
    failed: list[str] = []
    for m in metrics:
        if m.name not in df.columns:
            continue
        val = float(df[m.name].mean())
        if math.isnan(val):  # all rows errored (e.g. embeddings endpoint 501) → keep JSON valid
            failed.append(m.name)
            continue
        scores[m.name] = val

    print("\n" + "=" * 50)
    print("  RAGAS ANSWER QUALITY")
    print("=" * 50)
    for name, value in scores.items():
        print(f"  {name:<22} {value:.3f}  {_grade(value)}")
    for name in failed:
        print(f"  {name:<22} FAILED (all rows errored — see exceptions above; metric skipped)")

    report = {"judge_model": args.judge_model, "n_rows": len(rows), "scores": scores}
    out_path = Path(args.out) if args.out else Path(str(args.rows).replace(".json", "") + ".scored.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] Report saved → {out_path}")


if __name__ == "__main__":
    main()
