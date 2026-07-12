# Rerank-floor re-calibration on the e5-FT corpus (2026-07-13)

`RETRIEVAL_RELEVANCE_FLOOR=0.55` was calibrated on the **bge-m3** embedding
(2026-06-19: out-of-corpus ~0.50, in-corpus ~0.726). Prod now serves the
fine-tuned **e5** embedding (`agentrag-embed-v1`, 768-dim via TEI), so the
rerank-score distribution had to be re-measured on the real medical corpus.

## Method

The old probe scripts (`scripts/eval/probe_thin_*.py`) are stale two ways: their
questions come from the 2026-06 **news/vn residue** corpus (off the current
medical corpus → all score ~0.50), and they spy on `service._is_thin_context`,
which the live **13-node graph** path does not call. Re-probed on the live path
by wrapping `ContextAssembler.assemble` and reading the post-rerank `ranked`
list, feeding real medical questions from `data/eval/c2_evalset_n40.jsonl`
(in-corpus) plus fabricated drug/disease names (out-of-corpus).
Reranker = `local_cross_encoder` (`BAAI/bge-reranker-v2-m3`), sigmoid output.

## Result

| set | min | median | max | n |
|---|---|---|---|---|
| out-of-corpus (fabricated) | 0.5014 | 0.5045 | **0.5176** | 4 |
| in-corpus (real medical)   | 0.5268 | 0.7209 | 0.7310 | 12 |

- Still cleanly bimodal, matching bge-m3: out-of-corpus flat ~0.50–0.52,
  in-corpus bulk **0.66–0.73**.
- One in-corpus outlier at 0.5268 ("reproduce exam questions 8–12" — a meta/thin
  request, arguably genuinely low-relevance).

## Decision — keep `RETRIEVAL_RELEVANCE_FLOOR=0.55` (no change)

0.55 sits in the gap: out-of-corpus max **0.5176 < 0.55 <** in-corpus bulk 0.66+
→ off-corpus questions are gated, real questions pass. **0.55 beats 0.6 on this
corpus**: raising to 0.6 would clip legitimate in-corpus hits at 0.5694 and
0.5956. The FT embedding did not shift the reranker distribution enough to need
recalibration (reranker is a separate cross-encoder, unchanged). See
`memory/rerank-floor-calibration.md`.
