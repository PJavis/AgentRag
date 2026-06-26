# RAG ablation benchmark — 2026-06-25-n40-gs8

- suite: `vn`  ·  n per dataset: `40`  ·  judge: `deepseek`
- generated: 2026-06-26T05:19:18

Each config ran in a **separate process** with all workstream flags forced to a
known baseline (off) then its own flag(s) set in the child env before `settings`
import. Configs are **grouped by index-shape** `(CR, RAPTOR)`: the first member of
each shape re-ingests (`STRUCTMEM_INGEST_MODE=sync`); query-time-only siblings
(CRAG/MULTIHOP/ADAPTIVE_ROUTING/SEMANTIC_CACHE) reuse that index via `--skip-ingest`.

## Acceptance gates

- `answer_correctness` baseline > 0.792 → target ≥ 0.85
- `contextual_precision` baseline > 0.819 → target ≥ 0.88
- `faithfulness` ≥ 0.80
- `latency_p50` < 10000 ms

Values below are metric **means** (judged 0–1); latency in ms, cost in USD.

| config | contextual_recall | contextual_precision | faithfulness | answer_correctness | citation_accuracy | latency_p50_ms | cost_per_query_usd | failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.854 | 0.780 | 0.943 | 0.723 | 0.844 | 19725.300 | 0.002 | 0.000 |
| cr_raptor | 0.858 | 0.794 | 0.904 | 0.726 | 0.846 | 20625.000 | 0.002 | 0.000 |

## Read — n=40 (80 questions/config) confirmation: CR+RAPTOR win does NOT hold

This run was the high-n confirmation of the CR+RAPTOR precision lift that the earlier
n=10 runs (20 questions each) suggested. **80 questions/config** roughly doubles the
statistical power (noise band ≈ ±0.03 vs ±0.06 at n=20), so it is the deciding sample.

**cr_raptor − baseline:** recall +0.004 · precision **+0.014** · faithfulness **−0.039**
· correctness +0.003 · citation +0.002 · latency +0.9 s.

| run | sample | precision Δ | faithfulness Δ |
|---|---|---|---|
| gs=0 matrix | n=10 (20 q) | +0.034 | −0.031 |
| gs=8 re-bench | n=10 (20 q) | +0.041 | +0.078 |
| **gs=8 n=40** | **n=80** | **+0.014** | **−0.039** |

**Conclusion — the +0.04 precision lift was a small-sample artifact.** At n=80 the
precision delta collapses to +0.014 (inside the noise band) while faithfulness is
**0.039 lower** than baseline (baseline grounds better — CR's contextualized chunks and
RAPTOR summary nodes add slightly less-grounded retrieval material). Recall, correctness,
citation, latency are all flat. So CR+RAPTOR delivers **no reliable quality gain** and
costs a heavy per-ingest tax (CR contextualization + RAPTOR tree build + StructMem
graph-extraction over more nodes; ~4 h/ingest here).

### Decision (supersedes the 06-25 gs=8 "keep ON" preliminary)

**Recommend CR+RAPTOR → OFF.** No measurable precision/recall/correctness benefit at the
trustworthy sample, slightly worse faithfulness (bad for a medical system), and a real
ingest cost. Turning both off makes ingestion cheaper/faster and restores the higher
baseline faithfulness with no quality loss. The live `.env` currently has them ON (a
leftover that the noisy n=10 result appeared to justify) — flip to OFF unless a different
corpus shape (real production docs, not the distractor-dense gs=8 eval set) later shows a
real gain. This does not change the query-time-flag verdict (all OFF).

Caveat: gs=8 is a distractor-dense synthetic eval corpus, harder than production. If the
real medical corpus (natural multi-chunk docs) is materially different, a one-off prod-shape
A/B could revisit — but on the evidence we have, OFF is the cost-justified default.

### Decision (2026-06-26): keep CR+RAPTOR ON pending a prod-corpus A/B (DEFERRED)

The synthetic-corpus evidence says OFF, but the result is on the public `vn_bkai`/`vn_legal`
gold-context sets, **not** the real medical corpus. Decision: **leave CR+RAPTOR ON in the
live `.env` for now**; do not flip on synthetic evidence alone. Settle it with a
**prod-corpus A/B** — **DEFERRED** (next-step, not yet built).

**Prod-corpus A/B plan (when picked up):**
1. **Eval set = synthetic Q-gen over the real corpus.** Run `mine_finetune_pairs.py`
   (synthetic Q from `data/originals` chunks, 114 PDFs) → `(question, gold source-chunk)`
   pairs. The source chunk is the gold context → enables precision/recall + judged
   faithfulness/correctness on the actual docs. ~100-200 Q.
2. **A/B (build needed — harness extension):** ingest `data/originals` with CR+RAPTOR
   **off** → score the synth set; re-ingest with CR+RAPTOR **on** → re-score; compare.
   The current `run_benchmark`/`run_ablation` ingest a suite's gold contexts via
   `load_suite`; this needs a path to ingest the real corpus + load the synth eval set
   (new dataset loader or a focused script). Each ingest of 114 real PDFs is heavy.
3. **Decide:** flip CR+RAPTOR OFF only if the real corpus also shows no precision gain
   (consistent with this n=80 result); keep ON if prod shape genuinely favors RAPTOR.

Until then the live default stays ON (status quo), and this n=80 synthetic result stands
as the documented evidence that the win is unproven at scale.
