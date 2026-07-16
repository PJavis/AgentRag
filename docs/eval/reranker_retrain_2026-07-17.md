# Reranker retrain (flywheel) — NOT deployed, 2026-07-17

Retrained the cross-encoder reranker on the flywheel to test whether it lifts the
system beyond the rerank-before-trim win. Pre-registered per the campaign rule
(no ship without a measured win).

## Setup
- Base: `BAAI/bge-reranker-v2-m3` (the deployed reranker).
- Train: `data/finetune/rerank_blend.jsonl` = 5888 embed_triplets + 107 citation
  triplets (flywheel) → 11 990 labelled pairs. `finetune_reranker.py`, 2 epochs,
  bs=8, **fp32** (amp triggers "CUDA device not ready" on this WSL GPU).
- Output: `models/agentrag-rerank-v1` (kept on disk, NOT wired into config).

## Result — LOSES, kept stock bge

Eval on `c2_evalset_n40_clean_v2.jsonl` (41 rows, CRAG off, breadth +
rerank-before-trim active):

| reranker | system avg | misses |
|---|---|---|
| stock bge-reranker-v2-m3 (baseline) | **0.904** | 23, multihop-6 |
| agentrag-rerank-v1 (retrained) | 0.888 | 23, **25**, multihop-6 |

- **Δ = −0.016** (< the +0.01 deploy bar) and **1 pass→miss regression**
  (prod_corpus-25). Fails the deploy rule on both counts.
- judge-noise pearson 0.793 this run → the −0.016 is within noise, i.e. the
  retrain is lateral-to-slightly-worse, not an improvement.

## Decision — KEEP stock bge-reranker-v2-m3
The reranker was never the bottleneck: the earlier agent-path dig showed it already
scores gold ~0.67–0.72 **when gold reaches it**, and the shipped rerank-before-trim
change is what surfaces gold. Retraining on the thin flywheel (107 citation pairs)
adds no signal the base model lacks. Consistent with the 2026-06-28 finetune-lever
result (C2 end-to-end FT lost 0.764 vs 0.813). Revisit only when the citation
flywheel is far larger.

`models/agentrag-rerank-v1` left on disk for a future re-eval; delete if space is
needed. No config change.
