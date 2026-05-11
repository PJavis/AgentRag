#!/usr/bin/env python
"""Eval candidate embedding (or reranker) vs baseline on a held-out test set.

Expects a JSONL file where each row has at least:
    {"query": str, "positive": str}

Re-uses your existing PostgreSQL chunks as the candidate pool. For every test
query:
    1. Embed query with BOTH models.
    2. Score against all unique positives (and a random sample of negatives) to
       form a ranking.
    3. Compute Recall@K and MRR.

Promotion gates (configurable via --gate-recall / --gate-mrr):
    recall@10 lift ≥ 5 pts AND mrr@10 lift ≥ 3 pts → exit 0 (promote)
    else → exit 1 (do NOT promote)

Usage:
    uv run python scripts/eval_retrieval.py \\
        --baseline intfloat/multilingual-e5-base \\
        --candidate models/agentrag-embed-v1 \\
        --test data/finetune/embed_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _recall_at_k(ranked_ids: list[int], gold_id: int, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def _mrr(ranked_ids: list[int], gold_id: int) -> float:
    try:
        i = ranked_ids.index(gold_id)
        return 1.0 / (i + 1)
    except ValueError:
        return 0.0


def evaluate(model_id: str, test_rows: list[dict], passages: list[str], ks=(5, 10)) -> dict:
    """Returns {recall@5, recall@10, mrr@10} for one model on the held-out set."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(model_id)
    needs_e5 = "e5" in model_id.lower()
    def fmt_q(t): return f"query: {t}" if needs_e5 else t
    def fmt_p(t): return f"passage: {t}" if needs_e5 else t

    logger.info("encoding %d passages with %s ...", len(passages), model_id)
    P = model.encode([fmt_p(p) for p in passages], batch_size=64, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=True)

    recalls = {k: [] for k in ks}
    mrrs: list[float] = []
    for row in test_rows:
        q = row["query"]
        gold = row["positive"]
        try:
            gold_id = passages.index(gold)
        except ValueError:
            continue
        qv = model.encode(fmt_q(q), normalize_embeddings=True)
        scores = P @ qv
        ranked = np.argsort(-scores).tolist()
        for k in ks:
            recalls[k].append(_recall_at_k(ranked, gold_id, k))
        mrrs.append(_mrr(ranked, gold_id))

    return {
        **{f"recall@{k}": (sum(v) / len(v) if v else 0.0) for k, v in recalls.items()},
        "mrr@10": sum(mrrs) / len(mrrs) if mrrs else 0.0,
        "n_queries": len(mrrs),
    }


def main(args: argparse.Namespace) -> int:
    rows = []
    with open(args.test, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("query") and r.get("positive"):
                rows.append({"query": r["query"], "positive": r["positive"]})
    if not rows:
        raise SystemExit("test file empty")
    logger.info("test set: %d queries", len(rows))

    # Build passage pool: positives + random sample of negatives from the file.
    pool_set = {r["positive"] for r in rows}
    with open(args.test, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("negative"):
                pool_set.add(r["negative"])
    passages = list(pool_set)
    random.shuffle(passages)
    logger.info("candidate pool: %d unique passages", len(passages))

    base = evaluate(args.baseline, rows, passages)
    cand = evaluate(args.candidate, rows, passages)

    print()
    print(f"{'metric':<14} {'baseline':>10} {'candidate':>11} {'delta':>10}")
    print("-" * 50)
    deltas = {}
    for m in ("recall@5", "recall@10", "mrr@10"):
        b, c = base[m], cand[m]
        d = c - b
        deltas[m] = d
        print(f"{m:<14} {b:>10.4f} {c:>11.4f} {d:>+10.4f}")
    print()

    # Gates
    ok_recall = deltas["recall@10"] >= args.gate_recall
    ok_mrr = deltas["mrr@10"] >= args.gate_mrr
    no_regression = all(d > -args.regress_tol for d in deltas.values())

    if ok_recall and ok_mrr and no_regression:
        print(f"PROMOTE: yes (recall@10 +{deltas['recall@10']:.3f} ≥ {args.gate_recall},"
              f" mrr@10 +{deltas['mrr@10']:.3f} ≥ {args.gate_mrr})")
        return 0
    print(f"PROMOTE: no")
    if not ok_recall:
        print(f"  recall@10 gate failed: +{deltas['recall@10']:.3f} < {args.gate_recall}")
    if not ok_mrr:
        print(f"  mrr@10 gate failed: +{deltas['mrr@10']:.3f} < {args.gate_mrr}")
    if not no_regression:
        print(f"  regression detected (any metric drop > {args.regress_tol})")
    return 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", default="intfloat/multilingual-e5-base")
    p.add_argument("--candidate", default="models/agentrag-embed-v1")
    p.add_argument("--test", default="data/finetune/embed_test.jsonl")
    p.add_argument("--gate-recall", type=float, default=0.05, help="min recall@10 lift to promote")
    p.add_argument("--gate-mrr", type=float, default=0.03, help="min mrr@10 lift to promote")
    p.add_argument("--regress-tol", type=float, default=0.02, help="max allowed drop on any metric")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
