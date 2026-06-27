# Eval-Fidelity Redesign — Design

- **Date:** 2026-06-26
- **Branch context:** `feat/ragas-langfuse-reranker`
- **Status:** approved design → ready for implementation plan

## 0. Problem

`answer_correctness` sits at a ~0.74 plateau across every system lever tried:

- Retrieval architecture (T6: CR / RAPTOR / CRAG / multihop / cache, and CR+RAPTOR at n=80) → no move.
- Answer-model quality (flash → pro A/B) → no move (+0.006, noise).

Meanwhile **faithfulness** (reference-*free*: answer vs retrieved context) **did** move +0.04 with the
pro model. A reference-free metric responds to a better model; a reference-based one does not. That is
the smoking gun: **the cap is in the ruler (gold answers + judge), not the system.**

### Mechanism (why the ruler caps at 0.74)

`answer_correctness` is the **RAGAS** metric, default `0.75 · factual_F1 + 0.25 · semantic_similarity`.
`factual_F1`:

1. Decomposes both system answer and gold into atomic claims.
2. LLM judge labels each TP (both), **FP (answer-only)**, FN (gold-only).
3. F1 over those counts.

Scored against **terse synthetic gold** (`vn_bkai` / `vn_legal`, short public-dataset stubs):

- Every *extra true* claim a grounded answer makes = **False Positive** → precision drops.
- Every gold claim phrased differently = **False Negative** → recall drops.

→ A more complete, faithful answer scores *lower*. The cap is structural and independent of retrieval
or generator quality.

## 1. Goal & success criteria

Build a ruler that **resolves real correctness gains**, in two sequenced phases.

- **Phase 1 (prove):** oracle answer (strong model + gold context) scored through the *current*
  claim-F1 lands near system 0.74 → the cap is the metric/gold, **quantified**, not asserted.
- **Phase 2 (raise):** new ensemble judge vs **human** labels agreement ≥ 0.7 (Cohen's κ / Pearson);
  prod-corpus correctness spread **wider** than synthetic (ruler now separates systems); CR+RAPTOR A/B
  settled on real docs.

## 2. The ensemble correctness judge (shared core — built in P1, reused in P2)

New module `src/agentrag/eval/correctness_judge.py`. Two reference-based but phrasing-robust scorers:

- **nugget-recall** — decompose **gold** into atomic must-have facts (nuggets).
  `score = covered_fraction − contradiction_penalty`. Extra true info is free; only contradictions
  are punished.
- **reference-guided rubric** — one judge call: `Q + gold + gold_context` → anchored 0–1 level.
- **ensemble output** — `{nugget, rubric, mean, abs_delta}`. Flag `abs_delta > 0.2` as low-confidence
  (self-validating cross-check).

Judge LLM injected via `llm_gateway` (mirrors `answer_eval.evaluate_answer`). Pure
decomposition/aggregation functions split out for unit tests (mirrors the `ragas_eval` mapper pattern).

## 3. Phase 1 — Diagnostic (prove). No ingest, current `vn` datasets.

`scripts/eval/oracle_probe.py`:

1. **Oracle ceiling** — strong model (pro) answers Q given **gold context** (oracle retrieval). Score
   vs gold through (a) current claim-F1 and (b) the new ensemble. If oracle ≈ system on claim-F1 → cap
   is gold/metric. If the ensemble lifts both *and* oracle ≈ system → metric artifact confirmed.
2. **Judge-noise floor** — re-score the same answers with a 2nd judge model (deepseek ↔ gemini).
   Correlation = noise floor; if low, no metric layered on top is trustworthy.

**Output:** `docs/eval/eval_fidelity_probe_<date>.md`. **Gate:** proceed to P2 (expected outcome).

## 4. Phase 2 — Prod-corpus ruler (raise). Full 114 PDFs.

`scripts/eval/build_prod_evalset.py`:

1. Sample chunks across the 114 ingested docs (reuse `mine_finetune_pairs._mine_synthetic_positives`)
   → synth-Q + **gold source-chunk** (= gold context, for free).
2. For each `(Q, gold-chunk)`: generate a **rich grounded gold answer** with a strong model
   **constrained to the gold chunk only** (extend `eval/dataset.generate_golden_dataset`). Kills the
   terse-stub penalty.
3. Emit `data/eval/prod_corpus_evalset.jsonl` in **`EvalExample` shape**
   (`id, question, reference_answer, gold_contexts, lang, source="prod_corpus"`).

**Harness integration (minimal):**

- `benchmark_datasets.py` — add `kind="local_jsonl"` and register `prod_corpus`; `load_suite("prod_corpus")`
  reads the local file. Everything downstream of `load_suite` in `run_benchmark.py` is unchanged.
- Run: `run_benchmark --suite prod_corpus --skip-ingest` (docs already live-indexed, so no re-ingest).
  Gold source-chunks feed DeepEval recall/precision; the ensemble judge scores correctness.

**CR+RAPTOR A/B fold-in:** ingest 114 with CR+RAPTOR **off** → score prod set; re-ingest **on** →
re-score; compare on real docs → settle the deferred decision. The two ingests *are* the campaign
(~8h total).

## 5. Calibration — the trust anchor (non-optional)

Sample **25–30 Q** from the prod set → **human** labels correctness (binary correct + completeness
note). Measure ensemble-vs-human agreement. **If κ < 0.7, the ruler is rejected** and the judge is
fixed before any number is trusted. Stored at `data/eval/calibration_human.jsonl`; agreement reported
in the P2 doc.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Gold answer is LLM-written → inherits model bias | Strong model + constrained to gold chunk + **human calibration** (§5) |
| Synth-Q answerable from 1 chunk → too easy | Tag difficulty; multi-chunk subset deferred to v1.1 |
| 8h ingest cost | Chosen path; checkpoint each ingest; CR+RAPTOR A/B amortizes it |
| Judge variance | Ensemble Δ-flag + 2nd-judge noise floor (P1) |

## 7. File/module plan

- **NEW** `src/agentrag/eval/correctness_judge.py` — ensemble judge (P1 + P2)
- **NEW** `scripts/eval/oracle_probe.py` — P1 diagnostic
- **NEW** `scripts/eval/build_prod_evalset.py` — P2 eval-set + gold answers
- **EDIT** `src/agentrag/eval/benchmark_datasets.py` — `local_jsonl` kind + `prod_corpus` registration
- **EDIT** `scripts/eval/generate_dataset.py` — fix stale `src.pam.*` → `src.agentrag.*`
- **NEW (gitignored)** `data/eval/prod_corpus_evalset.jsonl`, `data/eval/calibration_human.jsonl`
- **NEW** `docs/eval/eval_fidelity_probe_<date>.md`, `docs/eval/benchmark_prod_corpus_<date>.md`

## 8. Sequencing

P1 (judge module + oracle probe, ~1 day, no ingest) → **gate** → P2 build
(eval-set gen → CR-off ingest+score → CR-on ingest+score → calibration → report).

## 9. Scope boundary (YAGNI)

Multi-chunk / multi-hop gold questions are **deferred to v1.1**. v1 ships single-chunk gold (clean,
fast). No unrelated harness refactoring beyond the `local_jsonl` loader and the stale-import fix.
