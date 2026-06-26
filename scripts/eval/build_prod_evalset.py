#!/usr/bin/env python
"""Build a small prod-corpus eval set from the already-indexed corpus.

For N sampled corpus chunks (Segment.content): generate a question the chunk
answers, then a rich gold answer grounded ONLY in that chunk. Emit EvalExample-
shaped JSONL: {id, question, reference_answer, gold_contexts:[chunk], lang, source}.

No ingest — the docs are already in ES; this only reads Segment.content from PG.

Usage:
    uv run python scripts/eval/build_prod_evalset.py --n 30 \\
        --out data/eval/prod_corpus_evalset.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


_SYNTH_SYSTEM = (
    "You are a search-query generator. Given a passage, write ONE short, specific "
    "question that this passage directly answers. Same language as the passage. "
    "No yes/no question, no external knowledge. Return strict JSON: {\"questions\": [\"...\"]}"
)

_GOLD_SYSTEM = (
    "Answer the question using ONLY the provided context. Be complete, precise, and "
    "self-contained — a strong reference answer. Do not add facts beyond the context. "
    "Answer in the context's language."
)


def detect_lang(text: str) -> str:
    """Cheap heuristic: Vietnamese (and other diacritic-heavy text) trips the
    non-ASCII ratio; default to English otherwise."""
    if not text:
        return "en"
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return "vi" if non_ascii / max(len(text), 1) > 0.02 else "en"


def build_eval_row(idx: int, question: str, gold_answer: str, chunk: str,
                   source: str = "prod_corpus") -> dict:
    return {
        "id": f"{source}-{idx}",
        "question": question.strip(),
        "reference_answer": gold_answer.strip(),
        "gold_contexts": [chunk],
        "lang": detect_lang(chunk),
        "source": source,
    }


async def synth_question(chunk: str, gateway) -> str | None:
    payload, _ = await gateway.json_response(
        system_prompt=_SYNTH_SYSTEM,
        user_prompt=json.dumps({"passage": chunk[:2000]}, ensure_ascii=False),
        task="schema_discovery",
    )
    qs = payload.get("questions") or []
    for q in qs:
        if isinstance(q, str) and len(q.strip()) >= 5:
            return q.strip()
    return None


async def gen_gold_answer(question: str, chunk: str, gateway) -> str:
    user = f"CONTEXT:\n{chunk}\n\nQUESTION: {question}"
    return (await gateway.text_response(
        system_prompt=_GOLD_SYSTEM, user_prompt=user, task="gold_gen"
    )).strip()


async def _sample_chunks(n: int) -> list[str]:
    from sqlalchemy import select, func
    from src.agentrag.database import AsyncSessionLocal
    from src.agentrag.database.models import Segment

    async with AsyncSessionLocal() as s:
        rows = (
            await s.execute(
                select(Segment.content)
                .where(Segment.content.isnot(None))
                .order_by(func.random())
                .limit(n * 4)  # over-sample, then filter+shuffle
            )
        ).scalars().all()
    pool = [c for c in rows if c and len(c) >= 200]
    random.shuffle(pool)
    return pool[:n]


async def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.agentrag.services.llm_gateway import LLMGateway

    gateway = LLMGateway()
    chunks = await _sample_chunks(args.n)
    logger.info("sampled %d chunks (target %d)", len(chunks), args.n)

    rows: list[dict] = []
    sem = asyncio.Semaphore(4)

    async def _one(idx: int, chunk: str) -> None:
        async with sem:
            try:
                q = await synth_question(chunk, gateway)
                if not q:
                    return
                gold = await gen_gold_answer(q, chunk, gateway)
                if gold:
                    rows.append(build_eval_row(idx, q, gold, chunk))
            except Exception as exc:
                logger.warning("row %d failed: %s", idx, exc)

    await asyncio.gather(*[_one(i, c) for i, c in enumerate(chunks)])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("wrote %d eval rows → %s", len(rows), out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=30, help="number of chunks to sample")
    p.add_argument("--out", default="data/eval/prod_corpus_evalset.jsonl")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
