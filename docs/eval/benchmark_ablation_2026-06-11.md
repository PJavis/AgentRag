# RAG ablation benchmark — 2026-06-11

- suite: `both`  ·  n per dataset: `10`  ·  judge: `deepseek`
- generated: 2026-06-11T02:14:21

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
| baseline | 0.912 | 0.845 | 0.893 | 0.770 | 0.820 | 16311.400 | 0.001 | 0.000 |
| full | 0.912 | 0.840 | 0.895 | 0.767 | 0.855 | 15296.500 | 0.001 | 0.000 |
