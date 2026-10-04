# Rerank pool size A/B on the GPU reranker — 2026-10-04

**Decision: keep `RETRIEVAL_RERANK_TOP_N=20`.** 64 gives no measurable gain.

## Why this was run

`.env` had `RETRIEVAL_RERANK_TOP_N=8`, a 2026-07-22 demo-latency tweak made while
the cross-encoder ran on CPU. `maybe_rerank` only scores `candidates[:TOP_N]`, so
rerank-before-trim (2026-07-16, +0.082) was scoring 8 chunks of a 50+ pool. With
the reranker on the GPU (`RETRIEVAL_RERANK_BACKEND=tei`, ~0.2 s for 20 chunks,
~0.4 s for 60) the latency reason is gone. 20 is the value the 0.884 result was
measured with; 64 was untested.

## Setup

- `scripts/eval/oracle_probe.py`, both arms in parallel, same live corpus, same code.
- Eval set: `data/eval/c2_evalset_n40_clean_v2.jsonl` minus 2 rows whose gold text
  is no longer in the corpus (prod_corpus-29, prod_corpus_multihop-2; 0% verbatim
  after the September re-chunk/OCR repair). 39 rows.
- The corpus fingerprint no longer matches the set (`4191c8…` vs `5b3f00…`), so
  run with `--allow-corpus-mismatch`. The other 39 rows' gold text is still ≥80%
  verbatim in the live segments. **Compare the arms with each other, not with 0.884.**
- Other settings: `RETRIEVAL_NUM_CANDIDATES=30` (the demo value, restored to 50
  after this run), CRAG off, double-sigmoid rerank scores (floor 0.55).

## Result

| | TOP_N=20 | TOP_N=64 |
|---|---|---|
| system avg | **0.868** | 0.850 |
| oracle avg | 0.904 | 0.893 |
| oracle − system | +0.036 | +0.044 |
| misses (< 0.5) | 5 | 5 |
| judge-noise pearson | 0.894 | 0.898 |

- Mean difference −0.018, inside noise: the per-row system difference has
  sd 0.227. The oracle, which gets identical gold context in both arms, still
  moves by −0.010 (sd 0.095).
- Only 3 rows swing ≥ 0.3. prod_corpus-11 (0 → 1) and prod_corpus-7 (0.88 → 0)
  flip in opposite directions with gold packed in both arms, which points to
  answer/judge variance (AGENT_TEMPERATURE=0.3), not retrieval.
  prod_corpus_multihop-10 drops 1.00 → 0.70.

## Cost

About 1 h per arm (2 min/row: agent + oracle + 3 judge calls). Both arms together
used $1.00 of DeepSeek credit (balance 9.57 → 8.57), plus Gemini 2.5-pro judge
calls (eval_judge), which were not metered here.
