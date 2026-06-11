# RAG ablation benchmark — 2026-06-11-n40

- suite: `both`  ·  n per dataset: `40`  ·  judge: `deepseek`
- generated: 2026-06-11T14:24:09

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
| baseline | 0.874 | 0.799 | 0.954 | 0.740 | 0.833 | 26420.400 | 0.001 | 0.006 |
| cr_raptor | 0.891 | 0.800 | 0.927 | 0.710 | 0.804 | 27035.000 | 0.001 | 0.000 |
