# RAG ablation benchmark — 2026-06-10

- suite: `both`  ·  n per dataset: `10`  ·  judge: `deepseek`
- generated: 2026-06-10T22:39:33

Each config was run in a **separate process** with the workstream flags set in
that child's environment before `settings` import, and **re-ingested** so the
index reflects that config's ingest flags (`STRUCTMEM_INGEST_MODE=sync`).

## Acceptance gates

- `answer_correctness` baseline > 0.792 → target ≥ 0.85
- `contextual_precision` baseline > 0.819 → target ≥ 0.88
- `faithfulness` ≥ 0.80
- `latency_p50` < 10000 ms

Values below are metric **means** (judged 0–1); latency in ms, cost in USD.

| config | contextual_recall | contextual_precision | faithfulness | answer_correctness | citation_accuracy | latency_p50_ms | cost_per_query_usd | failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.925 | 0.843 | 0.897 | 0.755 | 0.925 | 15127.000 | 0.001 | 0.000 |
| full | 0.900 | 0.839 | 0.925 | 0.740 | 0.828 | 17545.300 | 0.001 | 0.000 |

## ⚠️ INVALID for Contextual-Retrieval / RAPTOR — do not trust the `full` row for them

Post-run verification (Elasticsearch) shows CR and RAPTOR **never applied** to the
`full` config:
- `0` segments carry the CR `context` field (`exists:context` → total 0).
- `segment_type` is only `text` (3947 docs); **no RAPTOR `summary` nodes**, `node_level` unused.

Root cause: the runner forces `STRUCTMEM_INGEST_MODE=sync` but does **not** disable
`UPLOAD_DEDUPE_BY_HASH`. The gold corpus was already ingested by a prior benchmark, so
each config's re-ingest **deduped to a no-op** — the index was never rebuilt with
CR/RAPTOR fields. The `full` row therefore tests only the **query-time** flags
(CRAG / adaptive-routing / semantic-cache / multi-hop) over the SAME flat index as
baseline.

Implications:
- The CR/RAPTOR comparison here is meaningless (both configs used identical flat chunks).
- The query-time flags, at **n=10** (high variance, ±0.1), show **no improvement** and a
  notable citation drop (0.925 → 0.828) + slower p50 — likely adaptive fast-path / CRAG
  altering which chunks get cited. Not conclusive at this n.

To produce a valid run: (1) fix the runner to set `UPLOAD_DEDUPE_BY_HASH=false` AND wipe
the `agentrag_segments`/`agentrag_memory_doc` indices per config (clean rebuild), and
(2) use n ≥ 20. Until then: **do not blanket-enable the flags in prod** — enable each only
after a valid ablation row beats baseline. (Per-feature wiring is confirmed working: a
direct smoke showed CRAG critique fires and the semantic cache hits on repeat queries.)
