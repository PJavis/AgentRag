# VITAL — Changelog (improvement effort 2026-06)

Canonical record of the P0-P4 improvement effort on `feat/ragas-langfuse-reranker`
(merged to `master`). Dated detail lives in the `docs/eval/` + `docs/security/` reports.
Roadmap: `docs/superpowers/specs/2026-06-24-vital-improvement-roadmap-design.md` (P0-P8).

> **Status:** P0 + P1 + P2 complete; P3 tooled (train at home); P4 in progress; P5-P8 roadmapped.
> CI green on GitHub Actions; `master` is the agentrag/VITAL line (the old `pam` project
> retired to tag `archive/pam-master`).

## 🔴 Critical regression fixes (the live system was broken before these)
The `e4eb895` "checkpoint" commit shipped two breakages; unit tests masked both (mocked retrieval).
- **`bootstrap_search(intent=)`** — deleted the `intent` param but left 5 call sites → every
  live chat raised `TypeError`. Fixed (`e7dc53e`).
- **rerank `no_candidate_ids`** — `maybe_rerank` keyed candidates on `id`; the assemble
  pipeline carries `content_hash` → rerank silently skipped → no `rerank_score` → abstain/
  floor/answerability all inert. Backfill `id=content_hash` (`af29043`).

## 🟢 Safety & quality (P1)
- **Abstain re-validated.** Rerank fix revived thin-context abstain (refusal 0.000→0.267 on
  the out-of-corpus set). Prompt-layer abstain is weak alone; the hard relevance-floor gate
  drives distractor citations → 0. See `docs/eval/benchmark_answerability_ab_2026-06-24_vi.md`.
- **Deterministic empty-context refusal** (`b1ca39e`) — when the floor gate empties context,
  refuse WITHOUT calling the answer LLM (kills parametric hallucination). Wording carries a
  canonical uncertainty marker so it scores as a clean abstain.
- **Gray-band answerability gate** + citation-scrub (`f14ba6f`/`9a1c3d2`, default OFF).
- **Lenient LLM JSON parse** (`17716b3`) — tolerate trailing data/prose/fences from
  `gemini-2.5-flash-lite` (cut decide/HyDE parse-failure storms).
- **Rerank config:** `RETRIEVAL_RERANK_BACKEND=local_cross_encoder` is the only backend that
  emits `rerank_score` (powers abstain) — startup guards an API model-name under it.
- **T6 WS1-5 ablation** (separate session): CRAG/fast-path/semantic-cache/multi-hop OFF (no
  above-noise gain). CR+RAPTOR looked like contextual_precision +0.04 at n=10, **but the n=40
  confirmation (80 q/config) did NOT hold it** — precision +0.014 (noise), faithfulness −0.039,
  for a heavy ingest cost (`benchmark_ablation_2026-06-25-n40-gs8.md`). All on the synthetic
  `vn_bkai`/`vn_legal` sets, so **CR+RAPTOR is kept ON in `.env` pending a prod-corpus A/B**
  (synthetic Q-gen over `data/originals`, **deferred**).

## 🧭 Key finding — the correctness ceiling is the EVAL, not the system (2026-06-26)
- **answer-model A/B** (`benchmark_answer_model_ab_2026-06-26.md`): same index + same vn n=40,
  `answer=flash` vs `answer=pro`. pro moved **correctness +0.006 (noise)** — a 2× answer model
  does NOT lift correctness — while faithfulness rose +0.04 (0.93→0.97) at ~2× latency. **Keep
  `answer=deepseek-v4-flash`.**
- **Conclusion:** two independent strong levers — retrieval architecture (T6) AND answer-model
  quality (this A/B) — both fail to move correctness off **~0.74**. When neither better
  retrieval nor a better generator shifts a metric, the metric is the bottleneck. Faithfulness
  (reference-free) *does* move, so the system responds; correctness (reference-based, vs terse
  synthetic public-dataset gold + LLM judge) is **gold/judge-bound**. So the plateau is an
  **eval-fidelity ceiling, not a system limit or design flaw** (architecture is sound; faith
  0.93–0.97 proves the hard part works).
- **Next investment = the ruler, not the engine:** build a prod-corpus eval set over
  `data/originals` with realistic gold (the deferred A/B, now the priority) + consider a
  rubric/reference-free correctness judge. Only then is the P3 embedding/reranker fine-tune
  (the remaining structural lever) worth measuring.
- **✅ RESOLVED (2026-06-26) — the ruler was built; the 0.74 cap was the OLD metric.**
  Ensemble correctness judge (`src/agentrag/eval/correctness_judge.py`, nugget-recall +
  reference-guided rubric, `task`-routable) + prod-corpus eval set
  (`scripts/eval/build_prod_evalset.py`, synth-Q + grounded gold over `Segment.content`, no
  re-ingest) + oracle probe (`scripts/eval/oracle_probe.py`). Prod-corpus probe
  (`docs/eval/eval_fidelity_probe_prod_2026-06-26.md`): the new judge credits a correct answer
  **~0.98** (oracle) — NOT capped at 0.74 — and two judge models (flash vs pro) agree **0.965**
  (trustworthy). So the plateau was **RAGAS `answer_correctness` (claim-F1) penalising
  extra-true/rephrased claims vs terse public gold**, not a universal eval cap. **Adopt the
  ensemble ruler.**

## 🟢 Abstain robustness — flaky false-abstention fixed (2026-06-26)
The prod-corpus probe surfaced 3 questions the live system refused **despite the gold chunk at
rank 0**. Root-caused (systematic-debugging) to the thin-context abstain gate — NOT retrieval,
NOT generation:
- **Floor `0.6→0.55`** (`f5dfe76`) — bge scores paraphrased-relevant VN chunks ~0.61 (others a
  flat 0.5); the agent's query-rewrites made `max(rerank_score)` wobble around 0.6 → flip
  answer↔abstain run-to-run. 0.55 sits mid-band (OOC ~0.50 | floor | relevant ~0.61).
- **Raw-question retrieval injected into the rerank pool** (`c706d6c`,
  `RETRIEVAL_INCLUDE_RAW_QUERY=true`) — the agent's decide-step rewrites (hybrid_kg + variants)
  retrieved worse chunks than the raw question; injecting the raw hits guarantees rewrites only
  ADD, never drop the best chunk below the floor.
- **Deterministic query-rewrite** (temp 0, threaded through `json_response`).
- **Hang budget** (`dcbe196`, `AGENT_TOTAL_TIMEOUT_S=90` + per-call `LLM_REQUEST_TIMEOUT_S=60`) —
  bounds the whole `agent.chat` loop; graceful "busy, retry" on exceed (a 42-min hang was
  observed under a gemini 503 storm).
- **Validated** (`docs/eval/eval_fidelity_probe_prod_v2_2026-06-26.md`): system
  **0.842→0.950**, oracle−system **+0.134→+0.019** (gap → noise), **0 hard misses (was 3)**,
  OOC safety preserved (genuine out-of-corpus still abstains at 0.55). Directional (n=20, high
  gemini-503 skip) — a cleaner higher-n run is the home-run follow-up.
- **Cleaner n=50 follow-up** (`docs/eval/eval_fidelity_probe_prod_v3_2026-06-26.md`, 0 skips):
  oracle−system **+0.046 (< 0.05) → ceiling conclusion HOLDS** at the larger sample. But system
  avg is **0.888, not 0.950** — the v2 0.950 was inflated by the 10/30 gemini-503 skips
  (easy-Q selection bias) + a different judge; trust the clean 0.888. Caveat: free-tier gemini
  serves `gemini-2.5-pro` at limit:0 so this ran **all-DeepSeek** judges (flash vs pro) →
  judge-noise pearson 0.730 is the optimistic same-provider case + mild self-preference risk;
  a **paid-gemini cross-provider judge** is needed before quoting a correctness number with
  confidence. Corpus = indexed `vn_bkai`/`vn_legal` residue, not `data/originals` (real-corpus
  A/B still deferred).

## 🔬 Miss-bucketing + CRAG A/B tooling (2026-07-14, `feat/miss-buckets-crag-flywheel`)
Follow-up to the 2026-07-13 real-corpus read (oracle−system **+0.088** → system-bound, loss in
~5/40 misses): tooling to name those failures and act on them. Plan:
`docs/superpowers/plans/2026-07-14-miss-buckets-crag-flywheel.md`.
- **Per-row probe capture** — `oracle_probe.py --rows-out x.jsonl` dumps per question: system/
  oracle answers, judge1/judge2 scores, packed passages + rerank scores, inline `[n]` citations,
  `classify_refusal` class, tool queries (`src/agentrag/eval/probe_rows.py`).
- **Miss-bucket classifier** — `src/agentrag/eval/miss_buckets.py` +
  `scripts/eval/report_miss_buckets.py`: each miss (sys < 0.5) → `false_abstention` (floor/gate
  work) | `retrieval_miss` (gold never packed, Jaccard < 0.35 — HippoRAG-2 gate evidence) |
  `generation_miss` (gold packed, answer wrong); judge-gap rows (|sys−judge2| ≥ 0.4) flagged.
- **Citation-reward flywheel (RMM)** — `src/agentrag/eval/citation_mining.py` +
  `scripts/eval/mine_citation_pairs.py`: the answer LLM's own inline citations label the rerank
  pool (cited = positive, hardest uncited = hard negative, only rows sys ≥ 0.75) → triplets in
  the exact `finetune_reranker.py`/`finetune_embedding.py` input shape; `--append` accumulates
  across probe runs. Zero manual labels.
- 18 new tests (`test_probe_rows`, `test_miss_buckets`, `test_citation_mining`); eval suite
  84/84. Live runs (bucket report, CRAG on/off A/B + refusal-safety gate, flywheel seed) are a
  home-run — the CRAG loop itself was already built (WS3, default OFF); pre-registered enable
  rule: Δsystem ≥ +0.02 AND zero new hallucinated on the OOC refusal set.
- **Corpus-fingerprint guard** — kills the v3-landmine class (eval set built on residue corpus
  silently scoring sys=0.00 against the real one): `build_prod_evalset.py` stamps every row
  with `corpus_fp` (sha1 over sorted document-title:segment-count pairs);
  `oracle_probe.py --eval-set` recomputes the live fingerprint and REFUSES on mismatch
  (`--allow-corpus-mismatch` to override; unstamped legacy sets warn only).
  `src/agentrag/eval/corpus_fingerprint.py`, 9 tests.
- **Prod-traffic citation miner** — `scripts/eval/mine_citation_pairs_prod.py`: same RMM
  signal from rated production turns (thumbs-up ⨝ `chat_messages.citations`; rating stands in
  for the judge score) → flywheel accumulates from real usage, not just eval runs. Pure
  converter `feedback_to_row` + 4 tests; eval suite 97/97.
- **HippoRAG-2 StructMem spec DRAFT** (`docs/superpowers/specs/2026-07-14-hipporag2-structmem-design.md`)
  — phrase/passage bi-modal graph, synonym edges via embedding threshold (replaces the
  canonicalization prerequisite), query-to-triple seeding + recognition filter + PPR, union
  merge into the existing rerank pool (can only ADD candidates — floor/abstain untouched).
  **GATED: build only if the home-run bucket report shows `retrieval_miss` dominant.**

## 🧭 Miss-bucketing campaign — RESULTS (2026-07-14 → 07-16)
The home run executed both CRAG arms and, after a per-row audit, three eval-set cleaning
passes. Outcome docs under `docs/eval/` (`crag_ab_2026-07-14.md`,
`clean_remeasure_v2_2026-07-16.md`, `miss_buckets_clean_v2_2026-07-16.md`,
`generation_miss_diagnostic_2026-07-16.md`).
- **CRAG: keep OFF (decided).** On the real c2 n=40 set, CRAG-off 0.740 vs CRAG-on 0.755 →
  **Δ+0.015 < the +0.02 threshold**, inside judge-noise (pearson 0.94). Safe (0/15 hallucinated
  on the OOC set) but flips only 1 row — not worth the critique→corrective-retrieve latency.
  `CRAG_ENABLED` stays `False`.
- **The "headroom" was partly a broken-eval-set artifact.** The first bucket read
  (retrieval_miss 6/9) was contaminated: ~6/9 misses were unanswerable synthetic questions —
  dangling demonstratives ("bệnh nhân **này**") and meta-references to the source artifact
  ("câu 8 đến 12 trong **đề thi**", English OCR captions). On those, oracle scores ~1.0 only
  because it is handed the gold; the system has no anchor to retrieve on. New
  `src/agentrag/eval/question_quality.py` (`is_context_dependent`) filters them at build time;
  hardened over three passes (dangling nouns, bare "câu N", `(các|những) câu hỏi`, pure-ASCII
  non-Vietnamese guard).
- **Clean re-measure — headroom shrinks with each cleaning, as predicted:** dirty 0.740/+0.171 →
  clean-v1 0.787/+0.163 → **clean-v2 0.802/+0.118**. The residual **+0.118 is real** (> the ~0.05
  metric-ceiling band); false_abstention went to 0.
- **HippoRAG-2: SHELVED (gate resolved NO).** On the clean set, **both multi-hop misses failed at
  generation, not retrieval** (gold_overlap 1.00) — multi-hop retrieval already works, so an
  entity-graph/PPR traversal targets a non-occurring failure. Spec marked shelved. The real
  +0.118 splits into single-hop **retrieval coverage** (4/7 misses — high rerank ~0.72 but low
  gold_overlap ~0.1: the reranker confidently picks wrong chunks on broad "list" clinical Qs →
  embedding/reranker lever) and **answer generation** (3/7 — gold packed, answer wrong; diagnosed
  per-row in `generation_miss_diagnostic_2026-07-16.md`: 1 trust-low-rerank-context prompt lever,
  1 completeness tweak, 1 needs-eyeballing).
- **Flywheel seeded** — 107 citation triplets accumulated (`data/finetune/citation_pairs.jsonl`);
  a reranker retrain off this seed is the evidence-backed next lever (hits both residual buckets).
  Mining now dedups negatives against positives so a duplicate (hybrid+RRF-merged) passage cannot
  emit a degenerate (q, X, X) training pair. Eval suite 106/106.

## 📊 Observability + feedback loop (P2)
- **Langfuse online + per-turn traces** — `observe_chat_turn`/`update_turn_trace` group each
  `/chat` turn into one trace (`session_id=conversation_id`). Self-host on `:3002`.
- **Feedback → Langfuse score** — thumbs 👍/👎 land as a `user_feedback` score on the turn's trace.
- **Feedback capture** — `adapter_chat_feedback` table (migration `2026062501`) + the miners
  (`mine_finetune_pairs`, `mine_sft`, `mine_preference`).
- **Benchmark preflight** (`run_benchmark.py`) — fails fast on a half-up stack (ES/embedding/judge).
- **CI** — GitHub Actions: backend test gate (pg/ES services) + frontend vitest, both green.

## 🔐 Security & privacy (P4, in progress)
- **PHI trace gate** `OBSERVABILITY_CAPTURE_CONTENT` (default **False**) — question/answer/
  comment TEXT is NOT sent to the trace store unless opted in.
- **Prompt-injection defense** — `ANTI_INJECTION_RULE` in both answer prompts (treat retrieved
  passages as untrusted data, not instructions).
- **Account deletion** — `DELETE /chat/account` + `account_deletion.delete_user_data(user_id)`:
  full PG-authoritative wipe (documents+segments, conversations+messages, feedback, events,
  user row) + best-effort ES/image purge. Auth-gated (refuses anonymous/legacy).
- **AuthZ audit** (`docs/security/authz-audit-2026-06-25.md`) — **finding: IDOR/BOLA** on
  notebooks/sources/notes/insights/transformations (operate by id, no ownership check).
  **Resolved 2026-06-25: tenancy = single-tenant / on-prem → IDOR accepted by design**
  (`AUTH_ENABLED` is an access gate, not isolation). Re-opens as a launch blocker if the
  product goes multi-account / multi-clinic / multi-VM — then add the shared ownership
  dependency (audit Recommendation #2) first.
- **`.env.example` hardening** — empty-value lines no longer take their inline comment as the
  value (`ADAPTER_ADMIN_TOKEN` was silently set to the comment on `cp`-and-go → now disabled).

## 🤖 Fine-tune pipeline (P3, run at home on 16 GB)
- `scripts/mine_preference.py` — KTO/DPO preference data from thumbs.
- `scripts/finetune_dpo.py` — **reference-free KTO/ORPO LoRA, 3B default, VRAM-safe** (avoids the
  7B-DPO 16 GB ceiling).
- Retrieval FT run plan: `docs/superpowers/plans/2026-06-25-retrieval-finetune-run.md`.
- See `docs/FINETUNE_STRATEGY.md` + `docs/HOME-RUN.md`.

## 🧹 Truth & repo (P0)
- README / ARCHITECTURE / `README-full.md` / 18 module READMEs synced to code (structured-SQL
  path removed everywhere; gate flags, PHI gate, endpoints, finetune scripts, langfuse all current).
- Test baseline: `make test-fast` green; full suite minus the un-seeded ontology/ingestion env tests.
- `master` consolidated (was the stale `pam` project); `structmem` branch removed.
