#!/usr/bin/env python
"""Mine reranker/embedding triplets from oracle_probe --rows-out JSONL.

Cited-in-answer passages become positives; retrieved-but-uncited become hard
negatives (RMM citation reward). Appends cleanly to the same training file
format as scripts/mine_finetune_pairs.py.

Run:
    uv run python scripts/eval/mine_citation_pairs.py \
        --rows docs/eval/rows_c2_n40.jsonl \
        --out data/finetune/citation_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentrag.eval.citation_mining import mine_triplets


def main(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.rows).read_text(encoding="utf-8").splitlines() if line.strip()]
    trips = mine_triplets(rows, min_system_mean=args.min_score)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a" if args.append else "w", encoding="utf-8") as f:
        for t in trips:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[mine] {len(trips)} triplets from {len(rows)} rows → {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", required=True)
    p.add_argument("--out", default="data/finetune/citation_pairs.jsonl")
    p.add_argument("--min-score", type=float, default=0.75,
                   help="mine only rows with system_mean >= this (trustworthy positives)")
    p.add_argument("--append", action="store_true",
                   help="append to --out instead of overwriting (accumulate across runs)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
