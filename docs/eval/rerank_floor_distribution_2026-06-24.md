# Rerank-score distribution & relevance floor (2026-06-24)

P1 Task 7. Measures the cross-encoder (`dengcao/bge-reranker-v2-m3`, sigmoid→[0,1])
max-score distribution to validate / refine `RETRIEVAL_RELEVANCE_FLOOR`.

> **Prerequisite that bit us:** these scores only exist once the rerank id-fix
> (`af29043`) is in AND the backend is `local_cross_encoder` with model
> `dengcao/bge-reranker-v2-m3` (the committed default in `config.py`/`.env.example`).
> The live `.env` on this host overrides the backend to `llm_chat`, which produces no
> `rerank_score`, so any floor is inert. See `benchmark_answerability_ab_2026-06-24_vi.md`.

## Aggregate distribution

| Question class | max rerank_score | source |
|---|---|---|
| **Out-of-corpus** (irrelevant) | **≈ 0.500** (0.500–0.502, very tight) | measured today, n=15 refusal set, local cross-encoder |
| **In-corpus** (relevant) | **≈ 0.73** | 19/06 report (`benchmark_abstain_ab_2026-06-19_vi.md`) |

The out-of-corpus cluster is strikingly flat at ~0.50 — the cross-encoder floors
irrelevant passages at sigmoid(0)≈0.5 regardless of content. In-corpus relevant
passages sit ~0.73. The two classes are cleanly separable.

## Floor placement

`RETRIEVAL_RELEVANCE_FLOOR = 0.6` sits squarely in the 0.50–0.73 gap → correctly
classifies both today's out-of-corpus (all `< 0.6` → thin-context fires) and the
report's in-corpus (`> 0.6` → kept). **The global floor 0.6 is validated.**

This also explains why the new gray-band answerability gate `[0.60, 0.73)` engaged on
nothing for the refusal set: out-of-corpus questions land at ~0.50, *below* the band,
already handled by the thin-context path.

## Per-specialty floors — deferred (YAGNI for now)

The plan proposed per-specialty floors. Two reasons to defer:

1. **No separation signal yet.** Out-of-corpus is a flat ~0.50 floor set by the
   cross-encoder's sigmoid(0), not by specialty; the global gap to in-corpus ~0.73 is
   wide and uniform. There is no evidence specialties separate at materially different
   thresholds.
2. **Insufficient tagged data.** The 15-question refusal set is not specialty-tagged,
   and the in-corpus number is a single aggregate from the report. Per-specialty floors
   need the grouped in-corpus benchmark run with specialty tags — a larger measurement
   than this task's budget.

**Decision:** keep the global floor `0.6`; do **not** add a
`RETRIEVAL_RELEVANCE_FLOOR_BY_DOMAIN` map (no measured gain → avoid complexity).
Revisit only if a future per-specialty in-corpus probe shows >0.05 threshold spread
between domains.

## Follow-ups

- Re-measure in-corpus per specialty as part of the next full grouped benchmark (the
  ablation harness already ingests gold per suite — extend it to log per-domain max
  rerank if per-specialty floors are ever pursued).
- The ~0.50 out-of-corpus floor is a useful invariant: anything at exactly ~0.50 with
  no spread is almost certainly irrelevant. A deterministic "all candidates ≈0.50 →
  refuse" short-circuit would complement the relevance-floor gate.
