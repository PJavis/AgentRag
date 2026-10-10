# RAPTOR summaries at query time: A/B — 2026-10-10

**Result: excluding RAPTOR summaries from search made no measurable quality
difference and cut answer context by 25%.** Recommendation below.

## Why

CR and RAPTOR have been ON since June with no measured gain
(`benchmark_ablation_2026-06-25-n40-gs8.md`: precision +0.014, faithfulness
−0.039, correctness +0.003, "recommend OFF", prod-corpus A/B deferred).
RAPTOR summaries are separate segments (`segment_type="raptor_summary"`, 195 of
4,101 indexed segments), so `RETRIEVAL_EXCLUDE_RAPTOR_SUMMARIES` can test them
without a re-ingest. Contextual retrieval cannot be tested this way: its context
is baked into the only stored embedding.

## Setup

- `scripts/eval/oracle_probe.py` on `data/eval/c3_evalset_2026-10-10.jsonl`
  (41 rows: 36 single-chunk + 5 multi-hop, built 2026-10-10 from the live
  corpus, fingerprint `5b3f0000c845` = live). Both arms ran in parallel on the
  same corpus and code (master with #27), `CHAT_STRUCTMEM_ENABLED=false`
  (single-turn eval; keeps the 14B extraction model unloaded).
- B: RAPTOR summaries searchable (current behaviour).
- C: `RETRIEVAL_EXCLUDE_RAPTOR_SUMMARIES=true`.

## Result

| | B: RAPTOR on | C: RAPTOR excluded |
|---|---|---|
| system avg | 0.951 | 0.933 |
| oracle avg | 0.930 | 0.904 |
| misses (< 0.5) | 2 | 2 |
| judge-noise pearson | 0.564 | 0.758 |
| RAPTOR summaries in packed context | 176 of 821 chunks, in 30/41 questions | 0 |
| packed context per question (chars, mean) | 18,628 | **13,987 (−25%)** |

- System difference C−B: −0.018, per-question sd 0.179. The oracle, which gets
  identical gold context in both arms, moved −0.026 (sd 0.172), so the gap is
  inside judge noise.
- Only two questions moved ≥ 0.3, in opposite directions, both multi-hop:
  `prod_corpus_multihop-11` 1.00 → 0.00 without RAPTOR, `prod_corpus_multihop-5`
  0.50 → 1.00 without RAPTOR.
- RAPTOR summaries do reach the answer: 21% of packed chunks, many with high
  rerank probability (0.85–1.00) and some near 0. They take context space
  without improving answers on this set.

## Caveats

- Judge agreement is low on this set (pearson 0.56–0.76), so only large effects
  are detectable. The claim is "no detectable harm", not "proven equal".
- Only 5 multi-hop questions, which is where RAPTOR is meant to help. The two
  swings cancel; a multi-hop-heavy set would be needed to rule out a small
  multi-hop effect.

## Recommendation

1. Set `RETRIEVAL_EXCLUDE_RAPTOR_SUMMARIES=true`: about 25% less answer context
   per question at no measurable quality cost.
2. If that holds up in use, set `RAPTOR_ENABLED=false` for future ingests to
   stop paying for summary generation (~0.25 LLM calls per chunk). Existing
   summaries can stay in the index; the query switch already hides them.

## Cost

Both arms together: about 1 h wall-clock (parallel, ~2 min/question). Earlier
attempts the same day were killed by Claude Code's background-shell memory
reaper on an 11–13 GiB WSL VM; this run used
`CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`.
