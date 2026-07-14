#!/usr/bin/env python
"""Mine citation-reward triplets from PRODUCTION chat traffic.

Same RMM signal as mine_citation_pairs.py, but the source is rated prod turns
instead of probe rows: for every thumbs-up assistant turn, the passages its
answer actually cited (inline [n]) become positives; retrieved-but-uncited
passages become hard negatives. Thumbs rating stands in for the judge score,
so only +1 turns are mined.

Run (needs Postgres up; no LLM calls):
    uv run python scripts/eval/mine_citation_pairs_prod.py \
        --out data/finetune/citation_pairs.jsonl --append
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

from src.agentrag.eval.citation_mining import feedback_to_row, mine_triplets


async def _load_rated_turns() -> list[dict]:
    from sqlalchemy import select

    from src.agentrag.adapter.db import AdapterChatFeedback
    from src.agentrag.database import AsyncSessionLocal
    from src.agentrag.database.models import ChatMessage

    async with AsyncSessionLocal() as s:
        rows = (
            await s.execute(
                select(AdapterChatFeedback, ChatMessage)
                .join(ChatMessage,
                      AdapterChatFeedback.turn_id == ChatMessage.id.cast(
                          type(AdapterChatFeedback.turn_id.type)))
                .where(AdapterChatFeedback.rating == 1)
            )
        ).all()
    out: list[dict] = []
    for fb, msg in rows:
        row = feedback_to_row(
            question=fb.question or "",
            answer=fb.answer or msg.content or "",
            citations=msg.citations or [],
            rating=fb.rating,
        )
        if row:
            out.append(row)
    return out


async def main(args: argparse.Namespace) -> None:
    rows = await _load_rated_turns()
    trips = mine_triplets(rows)  # thumbs-up rows carry system_mean=1.0 → pass filter
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a" if args.append else "w", encoding="utf-8") as f:
        for t in trips:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[mine-prod] {len(trips)} triplets from {len(rows)} thumbs-up turns → {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/finetune/citation_pairs.jsonl")
    p.add_argument("--append", action="store_true",
                   help="append to --out instead of overwriting (accumulate across runs)")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
