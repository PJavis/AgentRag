#!/usr/bin/env python
"""Finetune a cross-encoder reranker on AgentRag triplets.

Each triplet (query, positive, negative) becomes two pairs:
    (query, positive)  → label 1.0
    (query, negative)  → label 0.0

Trains with BinaryCrossEntropyLoss. After training drop the path into .env:

    RETRIEVAL_RERANK_ENABLED=true
    RETRIEVAL_RERANK_BACKEND=local_cross_encoder
    RETRIEVAL_RERANK_MODEL=./models/agentrag-rerank-v1

Usage:
    uv run python scripts/finetune_reranker.py \\
        --base BAAI/bge-reranker-v2-m3 \\
        --train data/finetune/embed_triplets.jsonl \\
        --out models/agentrag-rerank-v1 \\
        --epochs 2 --batch-size 8

Hardware: ~5 GB VRAM on bge-reranker-v2-m3 @ bs=8.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    examples: list[InputExample] = []
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q, pos, neg = r.get("query"), r.get("positive"), r.get("negative")
            if not (q and pos and neg):
                continue
            examples.append(InputExample(texts=[q, pos], label=1.0))
            examples.append(InputExample(texts=[q, neg], label=0.0))
    if not examples:
        raise SystemExit("no usable rows in training file")
    logger.info("loaded %d labelled pairs", len(examples))

    ce = CrossEncoder(args.base, num_labels=1, max_length=512)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ce.fit(
        train_dataloader=DataLoader(examples, batch_size=args.batch_size, shuffle=True),
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(examples) / args.batch_size),
        output_path=str(out),
        show_progress_bar=True,
        use_amp=True,
    )
    logger.info("saved → %s", out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--train", default="data/finetune/embed_triplets.jsonl")
    p.add_argument("--out", default="models/agentrag-rerank-v1")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
