#!/usr/bin/env python
"""Mine (query, positive_chunk, negative_chunk) triplets for embedding +
reranker finetuning.

Three sources, blended into one JSONL:

1. **Thumbs-up feedback** (`adapter_chat_feedback.rating = 1`)
   → join with `chat_messages.tool_trace` of the rated assistant turn
   → every chunk in the trace becomes a positive for that question.

2. **Synthetic Q-gen** from corpus chunks
   → sample `--synth-chunks` chunks evenly across documents
   → ask a local Ollama (or Gemini) to produce N questions per chunk
   → (synthetic_q, source_chunk) becomes a positive.

3. **Hard negatives**
   → for each positive (q, pos), run current retriever (BM25 + dense)
   → take top-`--hard-neg-pool` hits, drop the positive, pick K random survivors.

Output: one JSONL row per triplet
  {"query": str, "positive": str, "negative": str, "source": "feedback|synth"}

Usage:
    uv run python scripts/mine_finetune_pairs.py \\
        --out data/finetune/embed_triplets.jsonl \\
        --synth-chunks 800 \\
        --questions-per-chunk 3 \\
        --hard-negs-per-positive 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any

from sqlalchemy import select

from src.agentrag.adapter.db import AdapterChatFeedback
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import ChatMessage, Segment
from src.agentrag.retrieval.elasticsearch_retriever import ElasticsearchRetriever
from src.agentrag.services.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Source 1: thumbs-up feedback ─────────────────────────────────────────────


async def _mine_feedback_positives() -> list[dict[str, str]]:
    """Each +1 feedback row → (question, chunks_used) positives.

    Tool trace stores the retrieval hits — those chunks were what the user
    rated positively, so they're known relevant.
    """
    out: list[dict[str, str]] = []
    async with AsyncSessionLocal() as s:
        rows = (
            await s.execute(
                select(AdapterChatFeedback, ChatMessage)
                .join(ChatMessage, AdapterChatFeedback.turn_id == ChatMessage.id.cast(type(AdapterChatFeedback.turn_id.type)))
                .where(AdapterChatFeedback.rating == 1)
            )
        ).all()

        for fb, msg in rows:
            question = fb.question or ""
            if not question:
                continue
            for trace_step in (msg.tool_trace or []):
                tool_out = trace_step.get("tool_output") or {}
                results = (
                    tool_out.get("results")
                    or tool_out.get("hybrid", {}).get("results")
                    or []
                )
                for hit in results[:5]:  # top 5 per trace step
                    content = hit.get("content")
                    if content and len(content) > 80:
                        out.append({
                            "query": question,
                            "positive": content,
                            "source": "feedback",
                        })
    logger.info("feedback positives: %d", len(out))
    return out


# ── Source 2: synthetic Q-gen ────────────────────────────────────────────────


_SYNTH_SYSTEM = (
    "You are a search-query generator. Given a passage, write N short, "
    "specific questions that this passage directly answers. Questions must be "
    "in the SAME LANGUAGE as the passage. No yes/no questions. No questions "
    "that require external knowledge. Return strict JSON: {\"questions\": [...]}"
)


async def _mine_synthetic_positives(
    n_chunks: int,
    n_per_chunk: int,
) -> list[dict[str, str]]:
    gateway = LLMGateway()
    out: list[dict[str, str]] = []

    async with AsyncSessionLocal() as s:
        # Even sample across documents: ORDER BY random() with LIMIT.
        rows = (
            await s.execute(
                select(Segment.content)
                .where(Segment.content.isnot(None))
                .order_by(Segment.id)  # stable scan; randomise after
                .limit(n_chunks * 4)   # over-sample to cover deletes
            )
        ).scalars().all()

    pool = [c for c in rows if c and len(c) >= 200]
    random.shuffle(pool)
    pool = pool[:n_chunks]
    logger.info("synth: sampled %d chunks (target %d)", len(pool), n_chunks)

    sem = asyncio.Semaphore(4)

    async def _one(chunk: str) -> None:
        async with sem:
            try:
                payload, _ = await gateway.json_response(
                    system_prompt=_SYNTH_SYSTEM,
                    user_prompt=json.dumps({"n": n_per_chunk, "passage": chunk[:2000]}, ensure_ascii=False),
                    task="schema_discovery",
                )
            except Exception as exc:
                logger.warning("synth Q-gen failed: %s", exc)
                return
            qs = payload.get("questions") or []
            if not isinstance(qs, list):
                return
            for q in qs:
                if isinstance(q, str) and len(q) >= 5:
                    out.append({"query": q.strip(), "positive": chunk, "source": "synth"})

    await asyncio.gather(*[_one(c) for c in pool])
    logger.info("synthetic positives: %d", len(out))
    return out


# ── Source 3: hard negatives ─────────────────────────────────────────────────


async def _mine_hard_negatives(
    positives: list[dict[str, str]],
    k: int,
    pool: int,
) -> list[dict[str, str]]:
    retriever = ElasticsearchRetriever()
    triplets: list[dict[str, str]] = []
    sem = asyncio.Semaphore(4)

    async def _one(item: dict[str, str]) -> None:
        async with sem:
            q, pos = item["query"], item["positive"]
            try:
                hits = (await retriever.search(query=q, mode="hybrid", top_k=pool)).get("results", [])
            except Exception as exc:
                logger.warning("retriever fail for q=%r: %s", q[:60], exc)
                return
            candidates = [
                h.get("content")
                for h in hits
                if h.get("content") and h.get("content") != pos and len(h.get("content", "")) > 80
            ]
            random.shuffle(candidates)
            for neg in candidates[:k]:
                triplets.append({
                    "query": q,
                    "positive": pos,
                    "negative": neg,
                    "source": item["source"],
                })

    await asyncio.gather(*[_one(p) for p in positives])
    logger.info("hard-neg triplets: %d (from %d positives)", len(triplets), len(positives))
    return triplets


# ── Main ─────────────────────────────────────────────────────────────────────


async def main(args: argparse.Namespace) -> None:
    positives: list[dict[str, str]] = []

    if not args.skip_feedback:
        positives.extend(await _mine_feedback_positives())

    if not args.skip_synth and args.synth_chunks > 0:
        positives.extend(
            await _mine_synthetic_positives(args.synth_chunks, args.questions_per_chunk)
        )

    if not positives:
        logger.error("no positives mined — abort")
        return

    # Dedup by (query, positive)
    seen: set[tuple[str, str]] = set()
    dedup: list[dict[str, str]] = []
    for p in positives:
        key = (p["query"].strip(), p["positive"][:120])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    logger.info("positives after dedup: %d", len(dedup))

    triplets = await _mine_hard_negatives(
        dedup, k=args.hard_negs_per_positive, pool=args.hard_neg_pool
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    logger.info("wrote %d triplets → %s", len(triplets), out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/finetune/embed_triplets.jsonl")
    p.add_argument("--skip-feedback", action="store_true",
                   help="Don't pull rating=+1 rows (use when DB has no feedback yet)")
    p.add_argument("--skip-synth", action="store_true",
                   help="Don't generate synthetic Q-A pairs")
    p.add_argument("--synth-chunks", type=int, default=500,
                   help="How many corpus chunks to sample for synth Q-gen")
    p.add_argument("--questions-per-chunk", type=int, default=3)
    p.add_argument("--hard-negs-per-positive", type=int, default=4)
    p.add_argument("--hard-neg-pool", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
