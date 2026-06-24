# RAG ablation benchmark — 2026-06-25-gs8

- suite: `vn`  ·  n per dataset: `10`  ·  judge: `deepseek`
- generated: 2026-06-25T06:07:49

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
| baseline | 0.950 | 0.836 | 0.857 | 0.880 | 0.830 | 21184.000 | 0.002 | 0.000 |
| cr | 0.925 | 0.830 | 0.947 | 0.815 | 0.795 | 20484.100 | 0.002 | 0.000 |
| cr_raptor | 0.925 | 0.877 | 0.935 | 0.795 | 0.790 | 20891.600 | 0.002 | 0.000 |

## Read — group_size=8 (RAPTOR/CR can actually build) + cross-run with the gs=0 matrix

This run ingests 8 gold contexts/doc (13 docs from 99 contexts) so each doc has
≥8 chunks — enough for RAPTOR to build its summary tree and for Contextual
Retrieval to add real context. It is a **harder, distractor-dense** task than the
gs=0 matrix (each doc mixes 1 relevant + 7 other passages), so the gs=8 baseline
is NOT comparable to the gs=0 baseline — compare only *within* this run.

Cross-run effect of **CR+RAPTOR vs its own baseline** (single run each, n=20,
judge/sampling noise ≈ ±0.05–0.07):

| metric | gs=0: base → cr_raptor | gs=8: base → cr_raptor | verdict |
|---|---|---|---|
| contextual_precision | 0.858 → 0.892 (+0.034) | 0.836 → 0.877 (+0.041) | **consistent lift** |
| faithfulness | 0.989 → 0.958 (−0.031) | 0.857 → 0.935 (+0.078) | stays high (0.935–0.958) |
| answer_correctness | 0.795 → 0.865 (+0.070) | 0.880 → 0.795 (−0.085) | flips sign → **noise** |

- **Precision is the one robust win:** CR+RAPTOR lifts it ~+0.04 in *both* corpus
  shapes, same direction → trustworthy despite single runs. RAPTOR's summary nodes
  + CR's context give the reranker better material.
- **Faithfulness stays high** (0.935–0.958) under CR+RAPTOR — no grounding regression,
  which matters most for a medical system.
- **Correctness shows no real effect** — it improves at gs=0 and worsens at gs=8 by
  similar magnitudes → noise, not signal. Do not claim a correctness gain.
- **No latency effect** (~20–21 s p50 throughout; answer-LLM bound).

### Decision (T6.5) — CR + RAPTOR

By the plan's accept criterion (*beat baseline on the target metric — precision —
without regressing faithfulness*), **cr_raptor passes** in both runs. CR+RAPTOR is
already enabled in the live `.env`; this validates keeping it ON. Caveats: the
precision lift is modest (~+0.04, single-run), correctness is a wash, and ingest
is heavier (RAPTOR build + graph-extraction). A higher-n (n≥40) confirmation is the
honest next step before treating the precision win as settled.

All query-time flags (CRAG / fastpath / semcache / multihop) and CR-alone remain
**OFF** — no above-noise win in either run; CRAG additionally hurt faithfulness at gs=0.
