# VITAL — Changelog (improvement effort 2026-06)

Canonical record of the P0-P4 improvement effort on `feat/ragas-langfuse-reranker`
(merged to `master`). Dated detail lives in `CHANGELOG-2026-06-24.md`,
`CHANGELOG-2026-06-25.md` (T6 ablation), and the `docs/eval/` + `docs/security/` reports.
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
  (synthetic Q-gen over `data/originals`, **deferred**). `docs/CHANGELOG-2026-06-25.md`.

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
