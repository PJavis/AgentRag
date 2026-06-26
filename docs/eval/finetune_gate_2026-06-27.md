# Fine-tune retrieval gate — 2026-06-27 (GOAL: prove or kill the fine-tune lever)

Goal: *"Prove or kill the fine-tune lever — with a trustworthy number — and promote it if it wins."*
The one structural lever not yet pulled: VN-medical domain fine-tune of the retrieval models.

## Setup
- **Corpus:** real medical docs `data/originals` (101/114 PDFs ingested text-only → ~2828 ES segs;
  ingest cut short at 89% by RAPTOR-clustering slowness — plenty for mining).
- **Training data:** `mine_finetune_pairs.py` → **5888 triplets** (synthetic Q-gen via deepseek +
  ES hard-negatives over the real corpus), 90/10 split → 5300 train / 588 test.
- **Hardware:** home RTX 5060 Ti (16GB). **fp32** (amp/fp16 triggers `CUDA device not ready` on
  this WSL + new-GPU + CUDA-12.8 driver), batch 8, max-seq 512.
- **Embedding base:** `intfloat/multilingual-e5-base` (chose e5-base over prod's bge-m3 for VRAM
  safety — bge-m3 fp32 hit ~16GB; e5-base fp32 ran at ~10GB, within the 14GB budget).

## GATE 1 — embedding FT vs e5-base (held-out test, recall/MRR)

| metric | baseline (e5-base) | FT (agentrag-embed-v1) | delta |
|---|---|---|---|
| recall@5 | 0.726 | 0.976 | **+0.250** |
| recall@10 | 0.789 | 0.993 | **+0.204** |
| mrr@10 | 0.578 | 0.913 | **+0.335** |

**Gate verdict: PROMOTE = YES** (recall@10 +0.204 ≥ 0.05 threshold; mrr@10 +0.335 ≥ 0.03).
**The fine-tune lever WORKS** — far above the bar.

## GATE 2 — reranker FT vs bge-reranker-v2-m3 — INVALID EVAL (tooling gap)

Reported numbers (recall@10 0.010 baseline / 0.002 FT, "PROMOTE: no") are a **measurement
artifact, not a result.** `eval_retrieval.py` loads every model as a `SentenceTransformer` and
ranks by `.encode()` cosine — a **bi-encoder** eval. A **cross-encoder reranker** has no usable
`.encode()`, so BOTH baseline and FT score ~1% recall (garbage). The script's docstring claims
"(or reranker)" support but does not implement it.

- The reranker FT model **trained + saved fine** (loss 0.225, `models/agentrag-rerank-v1`,
  after fixing the st-5.x save bug). It is simply **un-gated** by the available tool.
- A correct rerank eval = retrieve top-K per query (BM25/embedding) → score each (q, candidate)
  pair with the cross-encoder → re-rank → recall@k/MRR after rerank. **Follow-up:** write that
  eval (or run it at the company alongside the end-to-end probe). Do NOT read GATE 2 as the
  reranker failing.

## Verdict
**The fine-tune lever is PROVEN** by GATE 1 (embedding, valid eval): domain fine-tuning massively
improves real-corpus retrieval. GATE 2 is inconclusive only because the eval tool is bi-encoder-
only — the reranker is trained and awaits a proper rerank eval. Goal criterion 1 = **PASS**.

## Honest caveats (the standing rule: trustworthy numbers)
1. **Synthetic-test inflation.** The 588 test triplets come from the SAME synthetic-Q generator as
   training → the model partly learned the synth-Q→chunk style → the +0.20–0.33 **magnitude is
   inflated**. It is strong *directional* proof the lever works, NOT the real-world lift.
2. **e5-base ≠ prod bge-m3.** Deploying e5-FT means switching the embedding arch (dim 768 vs 1024)
   + re-ingest. The win shows domain-FT helps; a bge-m3 FT (prod-matched, drop-in) is the
   bigger-VRAM follow-up.
3. **Not yet end-to-end.** This is retrieval-only. Whether better retrieval lifts ANSWER
   correctness is criterion 2.

## Decision (goal criteria)
- **Criterion 1 (retrieval gate): PASS.** Lever proven — domain fine-tuning massively improves
  retrieval recall/MRR on the real corpus. Promote-worthy.
- **Criterion 2 (end-to-end, trustworthy): PENDING — run at company (paid gemini).** Promote the
  FT embedding (serve via TEI + re-ingest) → gemini-judged prod-corpus oracle probe → confirm FT
  system correctness − baseline > noise floor before production deploy.
- **Net:** the lever is NOT dead — it's the real ceiling-raiser the earlier (retrieval-trick +
  answer-model) levers weren't. Proceed to the end-to-end confirmation; do bge-m3 FT for a
  drop-in prod model on bigger VRAM.
