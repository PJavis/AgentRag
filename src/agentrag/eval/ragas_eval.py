"""RAGAS sample mappers (app-venv safe).

Turns an agent `chat()` result + a golden question into a RAGAS sample dict.
These are pure functions with no ragas/langchain dependency — they run in the
main app environment and are unit-tested directly.

IMPORTANT — why scoring lives elsewhere: RAGAS hard-requires the old langchain
stack (`langchain-core <0.3`), which conflicts with this project's
`langchain-core 1.x` (pulled by `langgraph`). The two cannot share one venv. So
the eval is a two-step, decoupled flow:

  1. `scripts/eval/run_ragas.py` (this venv) — run the agent, build rows with
     `build_ragas_row`, dump them to `<title>_ragas_rows.json`.
  2. `scripts/eval/score_ragas.py` (isolated venv) — read the rows JSON and
     score them with ragas + a dedicated judge. Self-contained; no src import.
"""
from __future__ import annotations

from typing import Any


def extract_context_texts(context_items: list[Any] | None) -> list[str]:
    """Flatten packed-context items into plain passage strings for RAGAS.

    Accepts the shapes the agent emits: dicts with `content`/`excerpt`/`text`,
    or bare strings. Drops empties.
    """
    texts: list[str] = []
    for item in context_items or []:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = item.get("content") or item.get("excerpt") or item.get("text") or ""
        else:
            text = ""
        text = (text or "").strip()
        if text:
            texts.append(text)
    return texts


def build_ragas_row(
    *,
    question: str,
    answer: str,
    context_items: list[Any] | None,
    ground_truth: str = "",
) -> dict[str, Any]:
    """Build a single RAGAS sample using its canonical field names.

    RAGAS `EvaluationDataset` expects: user_input, response,
    retrieved_contexts, reference.
    """
    return {
        "user_input": question,
        "response": answer or "",
        "retrieved_contexts": extract_context_texts(context_items),
        "reference": ground_truth or "",
    }
