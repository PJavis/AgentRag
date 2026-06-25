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
- **T6 WS1-5 ablation** (separate session): **CR+RAPTOR = contextual_precision +0.04 (keep ON)**;
  CRAG/fast-path/semantic-cache/multi-hop OFF. `docs/CHANGELOG-2026-06-25.md`.

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
  notebooks/sources/notes/insights/transformations (operate by id, no ownership check). High
  if multi-tenant; needs a tenancy decision + a shared ownership dependency (open follow-up).
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
