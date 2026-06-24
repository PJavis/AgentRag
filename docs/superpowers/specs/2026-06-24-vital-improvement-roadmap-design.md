# VITAL — Comprehensive Improvement Roadmap

**Date:** 2026-06-24 · **Branch:** `feat/ragas-langfuse-reranker` · **Objective:** balanced full roadmap (ROI-phased, all aspects)

## Context

VITAL (formerly AgentRag) is a self-hosted Vietnamese-medical RAG QA platform.
Two-plane architecture (Reasoning / Execution + DI `ServiceContainer`), hybrid
retrieval (BM25 + kNN + RRF) + cross-encoder rerank (`bge-reranker-v2-m3`) + 15×14
domain routing + StructMem knowledge memory, LangGraph 13-node agent, page-aware
citations, abstain-on-thin-context (relevance floor 0.6), Vision LLM, mindmap/summary,
DeepEval/RAGAS eval harness.

Latest benchmark (19/06/2026, hard corpus, n=80 in-corpus + 15 out-of-corpus):
all quality thresholds **pass** — faithfulness 0.951, recall 0.873, correctness
0.721, citation 0.788, precision 0.699 (borderline), failure 0.000; latency p50 17.7s.

Recent change (commit `e4eb895`): the structured-SQL reasoning path was **removed**
(`extractor`, `pipeline`, `query_classifier`, `schema_discovery`, `sql_engine`,
`synthesizer` deleted) → agent is now a single semantic flow.

### Present-status gap inventory (all aspects)

| # | Aspect | Problem |
|---|---|---|
| 1 | Safety | 8/15 out-of-corpus questions are "confident hallucinate" — ignore refusal directive. #1 named risk, medical-critical |
| 2 | Dormant value | WS1–5 (Contextual Retrieval, RAPTOR, CRAG+multihop, adaptive fast-path, semantic cache) built but never validated/enabled (default-OFF flags) |
| 3 | Quality | precision 0.699 at threshold; per-specialty rerank-floor tuning untried |
| 4 | Latency | p50 17.7s high; fast-path/cache target <10s unvalidated |
| 5 | Docs drift | `README.md` + `ARCHITECTURE.md` still describe removed SQL path/module; dead `STRUCTURED_REASONING_ENABLED` flag; config drift |
| 6 | Eval fidelity | cost/latency not production-representative (orchestration forced to cloud during benchmark) |
| 7 | Test health | 50 backend + 16 FE test files; known pre-existing FE reds; post-SQL-removal status unverified |
| 8 | Observability | Langfuse/Phoenix integrations exist but OFF; no online quality monitor / feedback loop |
| 9 | ML depth | generic `bge-m3` (no VN-medical embedding); no DPO/ORPO from thumbs feedback |
| 10 | Ops/repo | long-lived feat branch unmerged; 3 branches drifting; prod hardening/scaling unverified |
| 11 | Multimodal | vision only; no video/audio lecture ingestion |

## Sequencing principle

Truth + green baseline first (cheap, unblocks everything) → core quality/safety/latency
(the product) → robustness/ops/observability (parallelizable) → research (exploratory).
Each item carries an acceptance criterion so "done" is measurable.

---

## P0 — Truth & baseline (days, low effort, high ROI)

Goal: docs match reality, known-green test floor. Gates all later work.

1. **Fix stale docs.** `README.md` (overview, arch summary, module table) +
   `ARCHITECTURE.md` (overview, file map) still describe the removed structured-SQL
   path/module. Rewrite to single semantic path.
   *Accept:* no "structured SQL" / "two paths" references; module table has no dead
   `structured/` link.
2. **Kill dead config.** `.env.example` `STRUCTURED_REASONING_ENABLED=true` + any
   `config.py` remnant for the removed path.
   *Accept:* grep clean; app boots.
3. **Test baseline.** Run full `pytest` + frontend `vitest`; record pass/fail; fix
   SQL-removal fallout.
   *Accept:* documented baseline; backend green; FE reds catalogued as known-pre-existing.
4. **Branch hygiene.** `feat/ragas-langfuse-reranker` long-lived + 3 branches drifting.
   Decide merge-to-master or rebase plan.
   *Accept:* written integration decision.

## P1 — Safety · Quality · Latency (the core, 1–3 weeks)

Goal: close the #1 risk, unlock dormant value, cut wait.

5. **Kill confident-hallucinate** (8/15 out-of-corpus). Strengthen refusal prompt +
   drop temperature when context thin + add explicit "answerable from context?"
   pre-gate before answer. Re-eval the refusal set.
   *Accept:* confident-fabricate rate ≤ 0.15 (from 0.533); in-corpus quality flat.
6. **Validate WS1–5 via ablation.** Run `scripts/eval/run_ablation.py --suite both`
   on the live stack (PG + ES + DeepSeek key + Ollama); enable only rows that beat
   baseline. Sub-targets:
   - adaptive fast-path + semantic cache → latency p50 < 10s (from 17.7s);
   - Contextual Retrieval + RAPTOR (re-ingest) → precision 0.699 → 0.80+, recall;
   - CRAG + multihop → faithfulness / correctness on hard corpus.
   *Accept:* one ablation row per WS; flags flipped ON only on proven win; results doc'd.
7. **Per-specialty rerank floor.** Measure rerank-score distribution on real medical
   corpus; replace global 0.6 with per-domain floors.
   *Accept:* floor table + precision lift without recall loss.

## P2 — Observability · Ops · CI (1–2 weeks, parallel to P1)

8. **Wire Langfuse online** (branch is named for it; flag OFF). Online traces + quality
   monitor. *Accept:* live trace per `/chat`.
9. **Eval fidelity.** Env-gate benchmark so cost/latency uses internal models (report's
   known distortion) + health preflight (Ollama up, judge quota).
   *Accept:* prod-representative cost/latency numbers.
10. **CI pipeline.** GitHub Actions: `test-fast` + lint on PR; fix FE locale-parity +
    e2e mis-collect. *Accept:* green CI gate.
11. **Feedback capture.** Persist thumbs up/down → dataset (substrate for P3 DPO).
    *Accept:* feedback table populated.
12. **Ops hardening.** Verify auth/rate-limit; document prod deploy + worker autoscale.
    *Accept:* deploy runbook.

## P3 — Research depth (exploratory, longer)

13. **VN-medical embedding** — domain-adapt `bge-m3` on medical corpus.
14. **DPO/ORPO** from the P2 feedback dataset.
15. **Multimodal** — video/audio lecture ingestion.
16. **Learned per-specialty thresholds** (replaces #7 heuristic).

---

## Dependencies & critical path

- P0.3 (green tests) gates all subsequent work.
- P1.6 needs the live stack (PG + ES + DeepSeek key + Ollama).
- P3.14 needs P2.11 data.
- **Critical path to most value:** P0.1–3 → P1.5 (safety) → P1.6 (ablation = latency
  + quality in one pass).

## Out of scope (YAGNI for now)

- Net-new reasoning paths (the SQL path was just removed for being unproven — do not
  re-add speculative branches).
- Graph-DB (Neo4j/Graphiti) migration — StructMem-on-ES was a deliberate cost choice.

## Implementation note

P0 + P1 are the first executable plan (writing-plans). P2/P3 get their own plan cycles
after P1 lands and re-benchmarks.
