#!/usr/bin/env python
"""Mine SFT samples for LLM finetune from logged assistant turns.

For every assistant message with rating != -1 (i.e. accepted or unrated),
build a (system, user, output) tuple where:
    system = stock agent system prompt
    user   = JSON({"question": ..., "context": top_chunks})
    output = the assistant's actual answer

This treats your existing chat history (with Gemini-quality answers) as a
distillation target for the local Qwen LoRA.

Usage:
    uv run python scripts/mine_sft.py --out data/finetune/llm_sft.jsonl --limit 1000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from sqlalchemy import select

from src.agentrag.adapter.db import AdapterChatFeedback
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import ChatMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "Answer ONLY from the provided context. Be precise. "
    "Match the user's language. Surface concrete details (names, numbers, "
    "examples). Do not invent facts."
)


async def main(args: argparse.Namespace) -> None:
    samples: list[dict[str, str]] = []
    rejected: set[str] = set()

    async with AsyncSessionLocal() as s:
        # Collect turn ids the user rated -1 so we exclude them.
        neg_rows = (
            await s.execute(
                select(AdapterChatFeedback.turn_id).where(AdapterChatFeedback.rating == -1)
            )
        ).scalars().all()
        rejected.update(str(t) for t in neg_rows)

        # Walk assistant messages, build SFT rows in chronological order.
        msgs = (
            await s.execute(
                select(ChatMessage)
                .where(ChatMessage.role == "assistant")
                .order_by(ChatMessage.created_at.desc())
                .limit(args.limit * 3)  # over-fetch in case of filters
            )
        ).scalars().all()

        for m in msgs:
            if str(m.id) in rejected:
                continue
            if not m.content or len(m.content) < 30:
                continue
            # Find the user message immediately preceding this assistant turn.
            prev_user = (
                await s.execute(
                    select(ChatMessage)
                    .where(
                        ChatMessage.conversation_id == m.conversation_id,
                        ChatMessage.role == "user",
                        ChatMessage.created_at < m.created_at,
                    )
                    .order_by(ChatMessage.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if prev_user is None:
                continue
            # Reconstruct the context the agent actually saw.
            ctx_chunks = []
            for trace in (m.tool_trace or []):
                tool_out = trace.get("tool_output") or {}
                results = tool_out.get("results") or tool_out.get("hybrid", {}).get("results") or []
                for hit in results[:8]:
                    if hit.get("content"):
                        ctx_chunks.append({
                            "section": hit.get("section_path"),
                            "page": hit.get("page_start"),
                            "content": (hit.get("content") or "")[:1200],
                        })
                    if len(ctx_chunks) >= 8:
                        break
                if len(ctx_chunks) >= 8:
                    break
            user_payload = {"question": prev_user.content, "context": ctx_chunks}
            samples.append({
                "system": _SYSTEM_PROMPT,
                "user": json.dumps(user_payload, ensure_ascii=False),
                "output": m.content,
            })
            if len(samples) >= args.limit:
                break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info("wrote %d SFT samples → %s", len(samples), out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/finetune/llm_sft.jsonl")
    p.add_argument("--limit", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
