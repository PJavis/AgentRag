#!/usr/bin/env python
"""Render a miss-bucket report from an oracle_probe --rows-out JSONL.

Run:
    uv run python scripts/eval/report_miss_buckets.py \
        --rows docs/eval/rows_c2_n40.jsonl \
        --out docs/eval/miss_buckets_2026-07-14.md --label c2_evalset_n40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentrag.eval.miss_buckets import render_report, summarize_buckets


def main(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.rows).read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = summarize_buckets(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(rows, summary, args.label), encoding="utf-8")
    print(f"[buckets] {summary['misses']}/{summary['n']} misses → {dict(summary['buckets'])}")
    print(f"[buckets] wrote {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="eval-set")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
