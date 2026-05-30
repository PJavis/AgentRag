"""RAGAS eval — STEP 1 of 2: build evaluation rows (runs in the app venv).

Runs the agent over a golden dataset and dumps RAGAS rows
(question / answer / retrieved_contexts / ground_truth) to JSON. Scoring is a
separate isolated step because RAGAS requires `langchain-core <0.3`, which
conflicts with this project's `langchain-core 1.x` (via langgraph) — the two
cannot share one venv.

Usage:
  # Step 1 (this venv): produce rows
  python scripts/eval/run_ragas.py achievement-system --limit 5
  #   → writes data/eval/achievement-system_ragas_rows.json

  # Step 2 (isolated venv): score the rows
  GEMINI_API_KEY=... uv run --no-project \
      --with "ragas>=0.2,<0.3" --with "langchain-openai<0.3" \
      python scripts/eval/score_ragas.py data/eval/achievement-system_ragas_rows.json

Output: data/eval/<title>_ragas_rows.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentrag.config import settings
from src.agentrag.config_validation import validate_settings
from src.agentrag.eval.dataset import GoldenDataset
from src.agentrag.eval.ragas_eval import build_ragas_row
from src.agentrag.agent.factory import get_agent_service


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAGAS step 1 — build eval rows from the agent")
    p.add_argument("document_title", help="Document title to evaluate (must be ingested)")
    p.add_argument("--dataset", help="Golden dataset JSON (default: data/eval/<title>.json)")
    p.add_argument("--limit", type=int, help="Max questions to evaluate (quick tests)")
    p.add_argument("--out", help="Output rows path (default: data/eval/<title>_ragas_rows.json)")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    validate_settings(settings)

    doc = args.document_title
    dataset_path = Path(args.dataset) if args.dataset else ROOT / "data" / "eval" / f"{doc}.json"
    if not dataset_path.exists():
        print(f"[ERROR] Golden dataset not found at {dataset_path}")
        print(f"  Generate first: python scripts/eval/generate_dataset.py {doc}")
        sys.exit(1)

    dataset = GoldenDataset.load(dataset_path)
    questions = dataset.questions
    if args.limit:
        questions = questions[: args.limit]
    print(f"[INFO] Loaded {len(questions)} golden questions from {dataset_path}")

    agent = get_agent_service()
    rows: list[dict] = []
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q.question[:60]}...", end="", flush=True)
        try:
            result = await agent.chat(question=q.question, document_title=doc)
            rows.append(
                build_ragas_row(
                    question=q.question,
                    answer=result.get("answer", ""),
                    context_items=result.get("context", []),
                    ground_truth=q.expected_answer,
                )
            )
            print(" ok")
        except Exception as exc:
            print(f" ERROR: {exc}")

    if not rows:
        print("[ERROR] No rows collected; aborting.")
        sys.exit(1)

    out_path = Path(args.out) if args.out else ROOT / "data" / "eval" / f"{doc}_ragas_rows.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] Wrote {len(rows)} rows → {out_path}")
    print("\nNext (isolated venv — RAGAS needs old langchain):")
    print(
        f'  GEMINI_API_KEY=$GEMINI_API_KEY uv run --no-project \\\n'
        f'      --with "ragas>=0.2,<0.3" --with "langchain-openai<0.3" \\\n'
        f"      python scripts/eval/score_ragas.py {out_path}"
    )


if __name__ == "__main__":
    asyncio.run(main())
