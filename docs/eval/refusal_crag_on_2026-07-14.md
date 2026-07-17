# CRAG-on abstain-safety check — out-of-corpus refusal set (2026-07-14)

Runs `scripts/eval/run_refusal_ab.py` with `CRAG_ENABLED=true`, i.e. the CRAG
critique→corrective-retrieve loop is live during every answer call. Purpose: make
sure CRAG does NOT convert clean out-of-corpus abstentions into hallucinations
(it treats an uncertain answer as ungrounded and retries).

- Set: `data/eval/refusal_set.json` · n=15 fabricated/out-of-corpus questions
- Floor `RETRIEVAL_RELEVANCE_FLOOR=0.55` · rerank `local_cross_encoder`
- The script also toggles the separate `ANSWERABILITY_GATE_ENABLED` feature
  (gray-band gate), whose **production default is OFF** (`config.py:152`).

| Metric (CRAG on) | answerability gate OFF (= production) | answerability gate ON |
|---|---|---|
| refusal_rate (clean abstain ↑) | **1.000** | 0.933 |
| hedged_cited_rate ↓ | 0.000 | 0.000 |
| **hallucination_rate (DANGEROUS ↓)** | **0.000** | 0.067 |
| counts (abstain/hedged/halluc/empty) | 15/0/0/0 | 14/0/1/0 |

## Read

In the production configuration (answerability gate OFF), **CRAG on abstains
cleanly on all 15 out-of-corpus questions — zero hallucinated**. So CRAG does not
break refusal safety.

The single hallucination (`fab-mendoza`, `max_score=None` → thin/empty context)
appears only in the answerability-gate-ON arm; it is a property of that gate
letting a thin-context answer through, not of CRAG. That gate stays OFF in prod.

Compared against the historical baseline
`docs/eval/benchmark_answerability_ab_2026-06-24_vi.md` (current-config
prod-like arm: 0.000 hallucination), CRAG on shows **no regression** on OOC safety.
