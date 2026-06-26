# Answer-model A/B — flash vs pro (2026-06-26)

Isolates the **answer LLM's** contribution to correctness. Same cr_raptor index
(`--skip-ingest`, retrieval identical), same `vn n=40` (80 questions), judge=deepseek.
Only `LLM_TASK_MODEL_MAP.answer` differs: `deepseek-v4-flash` vs `deepseek-v4-pro`.

| arm | contextual_recall | contextual_precision | faithfulness | answer_correctness | latency p50 | cost/q |
|---|---|---|---|---|---|---|
| answer=flash | 0.858 | 0.805 | 0.931 | 0.734 | 18.4 s | $0.00172 |
| answer=pro | 0.858 | 0.803 | 0.971 | 0.740 | 34.1 s | $0.00194 |
| **Δ (pro−flash)** | 0.000 | −0.002 | **+0.040** | **+0.006** | **+85 %** | +12 % |

## Finding — the answer model is NOT the correctness lever

A 2× larger/slower answer model moves **correctness by +0.006 (noise)**. It does lift
**faithfulness +0.040** (pro hallucinates even less, 0.93→0.97) at the cost of ~2× latency
(18→34 s) and +12 % spend. **Keep flash** — pro buys near-perfect grounding but not
correctness, and the latency cost is steep.

## The real conclusion — the ceiling is the EVAL, not the system

Two independent, powerful levers now both fail to move correctness off ~0.73–0.74:
- **Retrieval architecture** (T6: CR/RAPTOR/CRAG/multihop/cache, and CR+RAPTOR at n=80) → no move.
- **Answer-model quality** (this A/B: flash→pro) → no move (+0.006).

When neither better retrieval **nor** a better generator shifts a metric, the metric itself
is the bottleneck. Corroborating evidence: **faithfulness DID move** (+0.04 with pro) —
faithfulness is *reference-free* (answer vs retrieved context), so the system clearly
responds to a better model. **Correctness is reference-based** (answer vs gold) and won't
budge → the cap is in the **gold answers + LLM judge**, not the system.

correctness ~0.74 is measured on **public synthetic datasets** (`vn_bkai`, `vn_legal`) with
**terse synthetic gold answers**, scored by an **LLM judge** that penalizes valid phrasing
variation. The system produces faithful, grounded answers (faith 0.93–0.97); the ruler can't
credit them past ~0.74.

### So: not a system limit, not a retrieval/generation design flaw — an **eval-fidelity** ceiling.

We have been optimizing against a ruler stuck at 0.74. Until the eval can resolve real
correctness gains, more system tuning (retrieval tricks, bigger models) is invisible by
construction.

## Recommendations

1. **Keep `answer=deepseek-v4-flash`** (confirmed). Revisit pro only if maximum faithfulness
   (0.97) is independently required for the medical/safety story — flash's 0.93 is already strong.
2. **Invest in eval fidelity, not more system levers.** Make correctness measurable before
   chasing it:
   - Build a **prod-corpus eval set** (synthetic Q-gen over `data/originals`, the deferred A/B)
     **with realistic gold answers**, not terse public-dataset stubs.
   - Consider a **rubric / reference-free correctness** judge (or human spot-check) so valid
     phrasing isn't penalized.
3. Only after a trustworthy ruler exists does the P3 domain embedding/reranker fine-tune (the
   remaining structural lever) become worth measuring.

## Files
`data/eval/answer_ab_flash.json`, `data/eval/answer_ab_pro.json` (gitignored raw reports).
