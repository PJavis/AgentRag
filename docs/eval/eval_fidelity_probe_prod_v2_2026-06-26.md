# Eval-fidelity probe v2 — prod corpus, post-fix (2026-06-26)

> **⚠️ SUPERSEDED for absolute numbers — see `eval_fidelity_probe_prod_v3_2026-06-26.md` (clean
> n=50, 0 skips).** This v2 run scored only 20/30 (10 dropped on gemini-503) → easy-Q selection
> bias inflated system avg to **0.950**; the trustworthy clean number is **0.888**. What still
> holds from v2: the abstain fix worked (0 hard misses, OOC safety) and oracle−system stays small
> (<0.05) → the eval-is-the-ceiling conclusion. Read v2 for the fix's relative impact, v3 for the
> number.

Re-run of the prod-corpus oracle probe AFTER the abstain-fix chain and the gold-prompt tighten.
Validates the aggregate impact of the fixes from the v1 investigation
(`eval_fidelity_probe_prod_2026-06-26.md`).

- eval set: `data/eval/prod_corpus_evalset_v2.jsonl` (gitignored) — rebuilt with the **gold-prompt
  v2** (no "based on the context" preamble, commit `872a863`).
- fixes in effect: floor `0.55` (`f5dfe76`) + deterministic `QueryRewriter` + raw-question retrieval
  injected into the rerank pool (`c706d6c`) + gateway 60s timeout.
- n=30 requested → **10 skipped on transient gemini 503** (heavy overload this run) → **20 scored**.

## Result — before vs after

| metric | v1 (pre-fix) | **v2 (post-fix)** | Δ |
|---|---|---|---|
| system avg (live agent, ensemble) | 0.842 | **0.950** | **+0.108** |
| oracle avg (pro + gold context) | 0.976 | 0.969 | ~flat |
| **oracle − system** | +0.134 | **+0.019** | **gap collapsed to noise** |
| judge-noise pearson (flash, pro) | 0.965 | 0.962 | trustworthy |
| **hard misses (sys 0.00/0.25)** | **3 / 26** | **0 / 20** | **eliminated** |

v2 system-score distribution (20 scored): `16× 1.00`, `0.92`, `0.85`, `0.70`, `0.53`. No refusals —
the lowest is a partial (0.53), not a 0.00 false-abstention.

## Read

1. **The false-abstentions are gone.** Zero hard misses (was 3). The floor-0.55 + raw-query-pool fixes
   closed the flaky false-abstention the v1 investigation root-caused.
2. **System now performs within noise of the oracle.** `oracle − system` collapsed from +0.134 to
   **+0.019** — the live system nearly matches perfect-retrieval + strong-generator. The v1 "+0.134
   headroom" was almost entirely the false-abstentions, not a retrieval/generation deficit.
3. **The ruler is still trustworthy** (judge pearson 0.962, ~unchanged) and still uncapped (oracle
   ~0.97). The 0.74-plateau conclusion stands: that was the old RAGAS claim-F1 metric.

## Caveats

- **n=20 effective** — 10/30 skipped on gemini 503 (heavy overload; the extra raw-query search adds
  some load too). **Directional, not a precision number** — a cleaner higher-n run (off-peak or with
  retry/backoff on 503) is needed to firm up the +0.108 / +0.019 figures.
- Single run; row-level flakiness is much reduced (0 hard misses) but not proven eliminated across
  many repeated runs.
- oracle 0.969, not 1.0 — gold-prompt v2 reduced but didn't fully remove the residual judge penalty
  on the LLM-written gold.
- Corpus is still the mixed-Vietnamese grab-bag (not medical); v1 single-chunk synth-Q still easy for
  the majority (16/20 = 1.0) — multi-chunk v1.1 would harden discrimination.

## Open

- **Total `agent.chat` hang/slowness**: the per-call 60s timeout doesn't bound the whole multi-rewrite
  loop; the high 503-skip rate + occasional >120s timeouts persist. A step/total budget is the fuller
  fix and remains open.
- Higher-n, lower-skip re-run to firm up the aggregate numbers.

## Files
`scripts/eval/build_prod_evalset.py`, `scripts/eval/oracle_probe.py`,
`src/agentrag/agent/context.py`, `src/agentrag/agent/service.py`, `src/agentrag/config.py`.
Eval set + raw log gitignored.
