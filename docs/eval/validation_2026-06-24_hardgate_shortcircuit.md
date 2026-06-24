# Validation — hard floor-gate + short-circuit refusal (2026-06-24)

One `run_benchmark` pass: `--suite vn --n 5 --refusal-set refusal_set.json`,
`RETRIEVAL_RELEVANCE_GATE_ENABLED=true`, local cross-encoder rerank, judge=gemini.
Covers two questions: does the short-circuit refusal kill confident hallucination
(#1), and does the hard floor-gate hold in-corpus quality enough to default ON (#3).

## In-corpus (n=10, judged) — decides #3

| Metric | 19/06 baseline | hard-gate ON |
|---|---|---|
| contextual_recall | 0.873 | **0.550** ↓↓ |
| contextual_precision | 0.699 | 0.700 |
| faithfulness | 0.951 | 1.000 ↑ |
| answer_correctness | 0.721 | **0.590** ↓ |
| citation_accuracy | 0.788 | 0.900 ↑ |
| failure_rate | 0.000 | 0.100 (1× gemini 500, transient) |

**#3 decision: keep `RETRIEVAL_RELEVANCE_GATE_ENABLED=False` (do NOT default ON).**
The hard gate drops every candidate below floor 0.6 — but some *relevant* in-corpus
passages also score < 0.6, so recall falls 0.873→0.550 and correctness 0.721→0.590.
Faithfulness/precision/citation rise (fewer, cleaner sources) but the recall cost is
too high to bake into the default. The report's "relevance-gate counterproductive"
verdict was right for in-corpus quality; the nuance is it's a **safety↔recall
tradeoff**, not a pure win. (n=10 is noisy, but the recall drop is too large to be
noise.)

**Refinement (future):** use a *lower* drop-floor for the gate (~0.52, just above the
~0.50 distractor floor) so it removes clear distractors without nuking borderline-
relevant context — decoupled from the 0.6 abstain-decision floor.

## Out-of-corpus refusal (n=15) — #1

Raw eval printed `hallucination_rate=0.733`, but this was a **measurement artifact**:
the agent *did* short-circuit to the deterministic refusal (answer text = the canned
refusal, **0 citations**, `hedged_cited=0.000`), but the refusal wording
("không **chứa** thông tin") didn't match the classifier's uncertainty marker
("không **có** thông tin"), so `classify_refusal` scored each refusal as
"hallucinated".

**Fix:** the deterministic refusal text now carries a canonical uncertainty marker
("không có thông tin" / "no information"), so `_has_uncertainty` + `classify_refusal`
+ the UI all recognise it as a clean abstain. Proven deterministically by
`tests/agent/test_answerability_gate.py::test_deterministic_refusal_classifies_as_abstention`
(no re-run of the slow eval needed). The short-circuit itself works: with empty
post-gate context the answer LLM is never called → no parametric hallucination, 0
citations.

**Net #1:** short-circuit refusal is correct and now correctly scored. A re-run would
report these as `refusal_rate` (clean abstain), not hallucination.

## Side note
- The JSON-lenient fix (`17716b3`) cut decide/HyDE parse failures from storms to ~5
  in this run — the agent loop churns far less.
- Latency remains high (p50 183s) on this host — dominated by multi-hop gemini calls;
  unrelated to these changes (the report already flags benchmark latency as not
  production-representative).
