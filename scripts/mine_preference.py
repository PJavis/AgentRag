#!/usr/bin/env python
"""Mine preference-tuning data from thumbs feedback (adapter_chat_feedback).

KTO (default): one {prompt, completion, label} per rated turn — the correct fit
for binary thumbs. DPO (--format dpo): {prompt, chosen, rejected} pairs where the
SAME question was rated both up and down.

Usage:
    uv run python scripts/mine_preference.py --out data/finetune/preference.jsonl --format kto
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from sqlalchemy import select

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_ANSWER_CHARS = 30
MAX_PAIRS_PER_GROUP = 8
_WS = re.compile(r"\s+")


def _normalize_q(question: str | None) -> str:
    return _WS.sub(" ", (question or "").strip().lower())


def _valid(row: dict[str, Any]) -> tuple[str, str, int] | None:
    q = (row.get("question") or "").strip()
    a = (row.get("answer") or "").strip()
    rating = row.get("rating")
    if not q or len(a) < MIN_ANSWER_CHARS or rating not in (1, -1):
        return None
    return q, a, int(rating)


def build_kto_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        v = _valid(row)
        if v is None:
            continue
        q, a, rating = v
        out.append({"prompt": q, "completion": a, "label": rating == 1})
    return out


def build_dpo_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        v = _valid(row)
        if v is None:
            continue
        q, a, rating = v
        g = groups.setdefault(_normalize_q(q), {"prompt": q, "chosen": [], "rejected": []})
        bucket = g["chosen"] if rating == 1 else g["rejected"]
        if a not in bucket:
            bucket.append(a)
    pairs: list[dict[str, Any]] = []
    for g in groups.values():
        n = 0
        for chosen in g["chosen"]:
            for rejected in g["rejected"]:
                if chosen == rejected:
                    continue
                pairs.append({"prompt": g["prompt"], "chosen": chosen, "rejected": rejected})
                n += 1
                if n >= MAX_PAIRS_PER_GROUP:
                    break
            if n >= MAX_PAIRS_PER_GROUP:
                break
    return pairs


async def load_rows(limit: int) -> list[dict[str, Any]]:
    from src.agentrag.adapter.db import AdapterChatFeedback
    from src.agentrag.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(
                AdapterChatFeedback.question,
                AdapterChatFeedback.answer,
                AdapterChatFeedback.rating,
            )
            .where(AdapterChatFeedback.question.isnot(None))
            .where(AdapterChatFeedback.answer.isnot(None))
            .limit(limit)
        )
    return [{"question": q, "answer": a, "rating": r} for (q, a, r) in result.all()]


async def main(args: argparse.Namespace) -> None:
    rows = await load_rows(args.limit)
    records = build_kto_records(rows) if args.format == "kto" else build_dpo_pairs(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("wrote %d %s records → %s", len(records), args.format.upper(), out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/finetune/preference.jsonl")
    p.add_argument("--format", choices=["kto", "dpo"], default="kto")
    p.add_argument("--limit", type=int, default=5000)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
