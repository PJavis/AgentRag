#!/usr/bin/env python
"""Split mined triplets into train/test JSONL.

Usage:
    uv run python scripts/split_pairs.py \\
        --input data/finetune/embed_triplets.jsonl \\
        --train data/finetune/embed_train.jsonl \\
        --test data/finetune/embed_test.jsonl \\
        --test-frac 0.1
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    random.seed(args.seed)
    random.shuffle(rows)
    n_test = max(1, int(len(rows) * args.test_frac))
    test, train = rows[:n_test], rows[n_test:]

    for path, data in ((args.train, train), (args.test, test)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train: {len(train)} → {args.train}")
    print(f"test:  {len(test)} → {args.test}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/finetune/embed_triplets.jsonl")
    p.add_argument("--train", default="data/finetune/embed_train.jsonl")
    p.add_argument("--test", default="data/finetune/embed_test.jsonl")
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
