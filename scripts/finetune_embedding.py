#!/usr/bin/env python
"""Finetune a sentence-transformers embedding model on AgentRag triplets.

Reads `data/finetune/embed_triplets.jsonl` (output of mine_finetune_pairs.py),
trains MultipleNegativesRankingLoss with in-batch negatives + the explicit
hard negatives from the file, and writes the result to `models/<name>/`.

After training:
    docker compose -f deploy/tei.compose.yml up -d
    # then in .env:
    EMBEDDING_PROVIDER=openai
    EMBEDDING_MODEL=agentrag-embed-v1
    EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1/
    OPENAI_API_KEY=tei-dummy

Usage:
    uv run python scripts/finetune_embedding.py \\
        --base intfloat/multilingual-e5-base \\
        --train data/finetune/embed_triplets.jsonl \\
        --out models/agentrag-embed-v1 \\
        --epochs 2 --batch-size 16

Hardware: ~5 GB VRAM on multilingual-e5-base @ bs=16, ~10 GB on bge-m3.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    # Import deferred so the script imports cleanly even without sentence-transformers installed.
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    model = SentenceTransformer(args.base)
    logger.info("loaded base: %s | dim=%d", args.base, model.get_sentence_embedding_dimension())

    # E5 family expects "query: " / "passage: " prefixes for best results.
    needs_e5_prefix = "e5" in args.base.lower()
    def q(text: str) -> str: return f"query: {text}" if needs_e5_prefix else text
    def p(text: str) -> str: return f"passage: {text}" if needs_e5_prefix else text

    examples: list[InputExample] = []
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not (r.get("query") and r.get("positive") and r.get("negative")):
                continue
            examples.append(InputExample(texts=[
                q(r["query"]),
                p(r["positive"]),
                p(r["negative"]),
            ]))
    if not examples:
        raise SystemExit("no usable rows in training file")
    logger.info("loaded %d triplets", len(examples))

    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(loader)),
        output_path=str(out),
        show_progress_bar=True,
        use_amp=True,
    )
    logger.info("saved → %s", out)
    logger.info("Next: run `make eval-embed` to gate the swap, then `make serve-embed`.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="intfloat/multilingual-e5-base",
                   help="HF model id (e5 family auto-applies query/passage prefix)")
    p.add_argument("--train", default="data/finetune/embed_triplets.jsonl")
    p.add_argument("--out", default="models/agentrag-embed-v1")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
