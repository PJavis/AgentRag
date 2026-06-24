# RAG ablation benchmark — 2026-06-24

- suite: `vn`  ·  n per dataset: `10`  ·  judge: `deepseek`
- generated: 2026-06-25T01:00:44

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
| baseline | 0.950 | 0.858 | 0.989 | 0.795 | 0.885 | 23134.100 | 0.002 | 0.000 |
| crag_only | 0.925 | 0.838 | 0.814 | 0.855 | 0.930 | 25196.400 | 0.002 | 0.000 |
| fastpath_only | 0.933 | 0.843 | 0.907 | 0.835 | 0.830 | 22327.800 | 0.002 | 0.000 |
| semcache_only | 0.933 | 0.869 | 0.875 | 0.870 | 0.900 | 22811.600 | 0.002 | 0.000 |
| multihop_only | 0.950 | 0.869 | 0.933 | 0.785 | 0.780 | 19664.900 | 0.002 | 0.000 |
| cr | 0.950 | 0.869 | 0.872 | 0.810 | 0.860 | 22418.100 | 0.002 | 0.000 |
| cr_raptor | 0.950 | 0.892 | 0.958 | 0.865 | 0.870 | 20176.800 | 0.002 | 0.000 |
| full | 0.950 | 0.927 | 0.908 | 0.900 | 0.845 | 20325.300 | 0.002 | 0.000 |

## Read (preliminary — pending group_size≥8 re-bench)

Single-run, n=20 questions/config. Judge + sampling variance ≈ **±0.05–0.07**, so
treat deltas inside that band as noise.

- **Query-time-only flags are noise.** `crag_only`/`fastpath_only`/`semcache_only`/
  `multihop_only` move precision/correctness within the noise band and **never cut
  latency** (p50 stays ~20–25 s; DeepSeek answer-gen ~18–20 s/Q dominates — the
  adaptive fast-path can't dent it). `semcache_only` is *untested for its real
  use-case* (20 unique queries, single pass → cache can't hit), not disproven.
- **`crag_only` hurts faithfulness** (0.989 → 0.814): CRAG's critique/rewrite step
  introduces ungrounded content. A real cost for a medical system.
- **`cr` alone ≈ baseline** (prec +0.011, corr +0.015 — noise). Expected: this matrix
  ingested at `group_size=0` (1 chunk/doc), which leaves Contextual Retrieval and
  especially **RAPTOR (needs ≥8 chunks/doc to build a tree) near-inert**.
- **`cr_raptor` and `full` are the only above-noise candidates**, and they stack
  monotonically: baseline (0.858/0.795) < cr_raptor (0.892/0.865) < full (0.927/0.900).
  `full` is the only config to reach **correctness 0.900** and the best precision
  (0.927) — but its faithfulness (0.908) sits 0.08 below baseline (the CRAG cost),
  whereas `cr_raptor` keeps faithfulness high (0.958).

### Provisional decision (T6.5)

- **Leave OFF:** CRAG, ADAPTIVE_ROUTING (fastpath), SEMANTIC_CACHE, AGENT_MULTIHOP,
  and CR-alone — none beat baseline above noise; CRAG/fastpath have real downsides
  (faithfulness / no latency win).
- **Hold for confirmation:** CR+RAPTOR (`cr_raptor`). Best faith-preserving lift, but
  crippled here by `group_size=0`. The **`group_size≥8` re-bench**
  (`benchmark_ablation_2026-06-25-gs8.md`) is the deciding test — enable
  `CONTEXTUAL_RETRIEVAL_ENABLED`+`RAPTOR_ENABLED` only if the lift holds (and
  ideally repeats) when RAPTOR can actually build.
- **Latency:** the <10 s target is unreachable by any flag here — it's answer-LLM
  bound. Belongs to a separate model/serving track, not the WS flags.
