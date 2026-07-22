# VITAL — Home-Run Guide

How to run VITAL on your home box (app + the GPU fine-tuning). For production deploy
hardening see `DEPLOY-RUNBOOK.md`; for the full manual see `README-full.md`.

Your home GPU: **RTX 5060 Ti 16 GB** (the embedding/reranker/KTO/ORPO trains fit here;
7B DPO does not — see §3).

---

## 0. ⚠️ Critical `.env` settings (the gotchas this effort uncovered)

```bash
# Reranker — local cross-encoder is the ONLY backend that emits rerank_score, which
# powers abstain/floor safety. Startup REJECTS an API model name under this backend.
RETRIEVAL_RERANK_BACKEND=local_cross_encoder
RETRIEVAL_RERANK_MODEL=BAAI/bge-reranker-v2-m3         # cached, free, GPU

# LLM — committed default is self-hosted: Ollama llama3.2:3b orchestration + DeepSeek
# for plan/answer.  Needs DEEPSEEK_API_KEY + `ollama pull llama3.2:3b`.
# Cloud alternative: GEMINI_API_KEY + swap LLM_TASK_MODEL_MAP to the gemini-2.5-* line.

# Embeddings — live config serves the fine-tuned e5 (retrieval gate: recall@10 +0.21)
# via TEI on :8080 (`make serve-embed`; 768-dim, --pooling=mean). Fresh-deploy fallback:
# EMBEDDING_PROVIDER=ollama + EMBEDDING_MODEL=nomic-embed-text, or bge-m3 via TEI.
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=agentrag-embed-v1
EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1/
EMBEDDING_OUTPUT_DIM=768

# Privacy — keep OFF (default): medical question/answer TEXT is NOT sent to the trace store.
OBSERVABILITY_CAPTURE_CONTENT=false
```

## 1. Bring up the app

```bash
cp .env.example .env && $EDITOR .env          # keys + the §0 gotchas
make docker-up                                 # postgres + elasticsearch + valkey + ollama
ollama pull llama3.2:3b                         # orchestration (skip if cloud LLM)
ollama pull nomic-embed-text                    # default embeddings
make migrate                                    # alembic upgrade head (incl. adapter_chat_feedback)
make seed-ontology                              # REQUIRED — ontology resolver + section tagger
make health
make up-bg                                      # api + worker + frontend (or `make dev`)
```

| URL | What |
|---|---|
| http://localhost:3000 | Chat UI (Trace per AI turn) |
| http://localhost:3000/cost | Cost/latency dashboard |
| http://localhost:8000/docs | API |

### Optional — Langfuse online (per-turn traces + thumbs scores)
```bash
docker compose up -d langfuse langfuse-db       # UI at http://localhost:3002 (auto-provisions dev keys)
# .env: LANGFUSE_ENABLED=true / LANGFUSE_HOST=http://localhost:3002 /
#       LANGFUSE_PUBLIC_KEY=pk-lf-agentrag-dev / LANGFUSE_SECRET_KEY=sk-lf-agentrag-dev
# restart the API → each /chat is one trace; 👍/👎 lands as a user_feedback score.
# With OBSERVABILITY_CAPTURE_CONTENT=false the trace shows structure only (no PHI text).
```

## 2. Verify it works

1. `make test-fast` → green backend gate.
2. **Chat works** (regression fixes): in-corpus question → answer with `[n]` citations, no `TypeError` in `make logs`.
3. **Abstain safety**: ask an out-of-corpus question (a made-up drug, e.g. *"Thuốc Zxylopraxin-9 dùng để làm gì?"*) → it refuses ("Tài liệu hiện có không có thông tin…") and cites nothing. If it answers confidently, your rerank backend is wrong (§0).
4. **Account deletion** (right-to-delete): as an authenticated user, `DELETE /chat/account` wipes all your data (documents/segments/conversations/messages/feedback/events). Refuses anonymous/legacy with 403.

## 3. Fine-tuning at home (the real quality lever)

All scripts exist (`scripts/`). Run on the 16 GB box. Detail: `docs/FINETUNE_STRATEGY.md` +
`docs/superpowers/plans/2026-06-25-retrieval-finetune-run.md`.

### 3a. Retrieval fine-tune — highest ROI, ~5-6 GB (do this first)
```bash
make mine-pairs                                  # synthetic Q-gen + thumbs + hard negs → embed_triplets.jsonl
make split-pairs                                 # 90/10 train/test
uv run python scripts/finetune_embedding.py --base BAAI/bge-m3 \
    --train data/finetune/embed_train.jsonl --out models/agentrag-embed-v1
uv run python scripts/eval_retrieval.py --baseline BAAI/bge-m3 \
    --candidate models/agentrag-embed-v1 --test data/finetune/embed_test.jsonl   # GATE: promote only on a gain
# (optional, step 2) finetune_reranker.py --base BAAI/bge-reranker-v2-m3 ...
# Promote: `make serve-embed` (TEI :8080) + point EMBEDDING_* at the new model; re-ingest.
```

### 3b. LLM preference tuning — KTO/ORPO, ~8-11 GB (after ≥500 thumbs)
```bash
uv pip install "unsloth[cu121-torch240]" trl
uv run python scripts/mine_preference.py --out data/finetune/preference.jsonl --format kto
uv run python scripts/finetune_dpo.py --method kto \
    --train data/finetune/preference.jsonl --out models/qwen-agentrag-pref-3b
# --method orpo for {prompt,chosen,rejected} pairs. 3B base default = VRAM-safe.
# Convert: scripts/convert_to_ollama.sh → ollama model → set in LLM_TASK_MODEL_MAP.
```
> Do NOT run 7B DPO on 16 GB (it holds chosen+rejected + a reference model ≈ ceiling → OOM).
> KTO/ORPO are reference-free; the 3B default leaves margin.

## 4. Known open item
**AuthZ IDOR** (`docs/security/authz-audit-2026-06-25.md`): notebooks/sources/notes endpoints
have no per-user ownership check. Fine for **single-user/on-prem**; a launch blocker if
**multi-tenant** (decide the tenancy model, then add the ownership dependency).

## 5. Eval-fidelity probe — the new ruler (2026-06-26)

The 0.74 `answer_correctness` plateau was the OLD RAGAS claim-F1 metric, not a system limit. A new
**ensemble correctness judge** (nugget-recall + reference-guided rubric) over a **prod-corpus eval
set** measures correctness honestly. Background: `docs/eval/eval_fidelity_probe_prod_2026-06-26.md`
(+ `_v2_` post-fix). This also surfaced + fixed a flaky false-abstention (2026-06-26 update).

**Relevant `.env` (independent-judge path — needs a PAID gemini key):**
```bash
# eval task slots — deepseek oracle/gold, CROSS-PROVIDER judge pair: primary judge gemini
# (independent of the deepseek answer model → no self-preference), judge2 deepseek so the
# judge-noise pearson measures real cross-provider agreement
LLM_TASK_MODEL_MAP={..., "oracle_gen":"deepseek-v4-pro","gold_gen":"deepseek-v4-pro",
                    "eval_judge":"gemini-2.5-pro","eval_judge2":"deepseek-v4-pro"}
RETRIEVAL_RELEVANCE_FLOOR=0.55       # calibrated (was 0.6 — flaky false-abstain)
RETRIEVAL_INCLUDE_RAW_QUERY=true     # inject raw-question hits into the rerank pool
AGENT_TOTAL_TIMEOUT_S=90             # bound the whole agent.chat loop (graceful "busy")
LLM_REQUEST_TIMEOUT_S=60             # per-call gemini timeout
```
> The `anthropic` provider IS wired (`agent/llm.py` auto-derive + `_resolve_backend_for`,
> tests in `tests/agent/test_anthropic_provider.py`) — a `claude-*` model in `eval_judge`
> works with just `ANTHROPIC_API_KEY` set. Alternative independent judge to paid gemini.

**Run it** (stack up; the corpus already ingested in ES; no re-ingest):
```bash
# 1. build a prod-corpus eval set (synth-Q + grounded gold over Segment.content)
uv run python scripts/eval/build_prod_evalset.py --n 50 --out data/eval/prod_corpus_evalset.jsonl

# 2. probe: live system vs oracle (perfect retrieval + strong model), ensemble-judged,
#    --retries rides out gemini 503 spikes
uv run python scripts/eval/oracle_probe.py \
  --eval-set data/eval/prod_corpus_evalset.jsonl --n 50 --retries 3 \
  --out docs/eval/eval_fidelity_probe_prod.md
```
> ⚠️ **`prod_corpus_evalset_v3.jsonl` is a landmine** — its questions were generated from the
> 2026-06-26 corpus, which was `vn_bkai`/`vn_legal` eval RESIDUE (driving-law fines etc.), not
> the real medical docs. Against the current `data/originals` corpus it scores `sys=0.00` on
> every question (verified 2026-07-13). Use `data/eval/c2_evalset_n40.jsonl` (built from the
> real corpus) or rebuild with `build_prod_evalset.py` after any re-ingest. Rule: an eval set
> is only valid against the corpus snapshot it was generated from.

**Read:** `oracle − system` small (< ~0.05) → system is at the eval ceiling (the metric, not
retrieval/generation, is the cap); `judge-noise pearson` high → the judge is trustworthy. The
**clean n=50 run is done** (`eval_fidelity_probe_prod_v3_2026-06-26.md`, 0 skips): system **0.888**,
oracle−system **+0.046 (<0.05) → ceiling CONFIRMED**. (The earlier v2 0.950 was inflated by 10/30
gemini-503 skips — easy-Q selection bias; trust the clean **0.888**.)

> ✅ **Judge independence CLOSED (2026-07-13,** `docs/eval/c2_probe_n40_gemini-judge.md`**).**
> With a paid gemini key, `eval_judge=gemini-2.5-pro` vs `eval_judge2=deepseek-v4-pro` on the
> real-corpus n=40 set: cross-provider pearson **0.921** (vs 0.730 same-family) and system
> 0.759 ≈ the deepseek-judged 0.764 → the deepseek-judged history was not self-preference-inflated.
> Same run shows **oracle−system +0.088** on the real corpus — unlike the residue corpus, real-doc
> correctness is **system-bound** (real headroom), not metric-bound.
