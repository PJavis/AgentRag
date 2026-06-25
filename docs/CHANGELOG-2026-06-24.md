# Changelog & Home-Run Guide — 2026-06-24

Improvement pass on `feat/ragas-langfuse-reranker` (now merged to `master`).
22 commits, `93da233..5bea6c7`. Two shipped regressions fixed, safety story
corrected and verified, repo consolidated.

---

## 1. Changelog

### 🔴 Critical fixes (the live system was broken before these)
- **`e7dc53e` Agent crash on every chat.** The `e4eb895` "checkpoint" deleted the
  `intent` param from `bootstrap_search` but left 5 callers passing `intent=None` →
  every real chat raised `TypeError`. Unit tests masked it (mocked retrieval).
- **`af29043` Rerank inert system-wide.** `LLMReranker.maybe_rerank` keyed candidates
  on `item["id"]`, but the assemble pipeline's candidates carry `content_hash` → it
  bailed `no_candidate_ids` → no `rerank_score` reached the answer node → the
  relevance-floor / abstain / answerability safety was **all structurally dead**, and
  retrieval was un-reranked (degraded precision). Backfill `id = content_hash`.

### 🟢 Safety features
- **`f14ba6f` Gray-band answerability gate** (`ANSWERABILITY_GATE_ENABLED`, default
  OFF). **`9a1c3d2`** also scrubs distractor citations on a gray-band refusal.
- **`b1ca39e` Deterministic empty-context refusal.** When context is empty (e.g. the
  floor gate dropped everything), refuse **without calling the answer LLM** — it can't
  then hallucinate from parametric memory. **`5bea6c7`** aligned the refusal wording to
  the canonical uncertainty marker so it scores as a clean abstain.

### 🛡️ Hardening
- **`b8a2f2d` Config guard.** Startup now rejects an API/chat model name under
  `RETRIEVAL_RERANK_BACKEND=local_cross_encoder` (the silent OSError trap that made
  rerank inert).
- **`17716b3` Lenient LLM JSON parse.** `gemini-2.5-flash-lite` emits valid JSON +
  trailing text (`Extra data`); `json_response` now uses `raw_decode` + balanced-brace
  fallback. Decide/HyDE parse-failure storms → ~5 per run.

### 📋 Docs / eval / decisions
- **`b48c921` / `2c5f22d`** README + ARCHITECTURE + `.env.example` aligned to the
  single semantic path (structured-SQL was removed in `e4eb895`); dead flags scrubbed.
- **`f639b88`** test baseline: `make test-fast` 142/142 green; the 10 full-suite +
  10 frontend failures are all env/known-pre-existing, **0 SQL-removal fallout**.
- Eval: rerank id-fix revived abstain (refusal 0.000→0.267); **hard floor-gate
  ON kills distractor citations but costs in-corpus recall 0.873→0.550 → kept OFF**
  (`docs/eval/validation_2026-06-24_hardgate_shortcircuit.md`).
- **`d2230d7`** GitHub Actions CI (test-fast + pg/ES services + ruff). Needs first-PR
  validation.
- Roadmap spec + plan: `docs/superpowers/specs/2026-06-24-vital-improvement-roadmap-design.md`,
  `docs/superpowers/plans/2026-06-24-vital-improvement-p0-p1.md`.

### 🗂️ Repo
- `master` was the stale **`pam`** project; it is now the agentrag/VITAL line. Old pam
  preserved at tag `archive/pam-master`. `structmem` branch deleted (was contained).

---

## 2. Home-run guide

### Prereqs
- Docker + Docker Compose, `uv`, Node (for frontend).
- LLM key: `DEEPSEEK_API_KEY` for the committed default stack (or `GEMINI_API_KEY` for
  the cloud alternative). Ollama for orchestration (`llama3.2:3b`) + embeddings.

### ⚠️ Critical `.env` settings (the gotchas this pass uncovered)
```
# Reranker — local cross-encoder is the ONLY backend that emits rerank_score,
# which powers abstain/floor safety. llm_chat (gemini) reorders but no score.
RETRIEVAL_RERANK_BACKEND=local_cross_encoder
RETRIEVAL_RERANK_MODEL=dengcao/bge-reranker-v2-m3   # NOT an API model name (startup now guards this)

# LLM — committed default is self-hosted: Ollama llama3.2:3b for orchestration
# (classify/decide/domain_router/followup) + DeepSeek for plan/answer/synthesize.
#   → needs DEEPSEEK_API_KEY  +  `ollama pull llama3.2:3b`
# Cloud alternative: set GEMINI_API_KEY and swap LLM_TASK_MODEL_MAP to the
# commented gemini-2.5-* line in .env.example.
DEEPSEEK_API_KEY=<your key>

# Embeddings — default Ollama nomic-embed-text (`ollama pull nomic-embed-text`).
# For best Vietnamese retrieval use bge-m3 via TEI instead (`make serve-embed` → :8080,
# then EMBEDDING_PROVIDER=openai / EMBEDDING_MODEL=BAAI/bge-m3 /
# EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1/). T6 ran on bge-m3.
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
```
If you set `local_cross_encoder` but leave an API model name, startup raises a clear
error (by design). The bge cross-encoder runs on GPU if present, else CPU.

### Bring it up
```bash
make docker-up        # postgres + elasticsearch + valkey + ollama
ollama pull llama3.2:3b              # orchestration (classify/decide/router/followup)
ollama pull nomic-embed-text         # default embeddings
# optional, better VN retrieval: bge-m3 embeddings via TEI (needs GPU) → make serve-embed
make migrate          # alembic upgrade head
make seed-ontology    # REQUIRED — ontology resolver + section tagger need it (else those tests/features fail)
make health           # verify pg/es/valkey/ollama + providers reachable
```

### Run
```bash
make dev              # api + worker + frontend, foreground (Ctrl+C stops all)
# or: make up-bg      # background; `make logs` / `make stop`
```
| URL | What |
|---|---|
| http://localhost:3000 | Chat UI (Trace per AI turn) |
| http://localhost:3000/cost | Cost/latency dashboard |
| http://localhost:8000/docs | API |

### Verify the fixes actually work
1. `make test-fast` → expect **green** (the merge gate; full `pytest` shows 10
   ontology/ingestion failures only if you skipped `seed-ontology` / lack Tesseract).
2. **Chat works at all** (regression `e7dc53e`): ask any in-corpus question → you get
   an answer with `[n]` citations, no `TypeError` in `make logs`.
3. **Abstain safety works** (regressions `af29043` + rerank config): ask an
   **out-of-corpus** question (e.g. *"Thuốc Zxylopraxin-9 dùng để làm gì?"* — a made-up
   drug) → the system should **refuse** ("Tài liệu hiện có không có thông tin…") and
   **cite nothing**, not invent an answer. If it confidently answers with citations,
   your `.env` rerank backend is wrong (see the critical settings above).

### Optional — re-run the eval
```bash
RETRIEVAL_RERANK_BACKEND=local_cross_encoder RETRIEVAL_RERANK_MODEL=dengcao/bge-reranker-v2-m3 \
PYTHONPATH=$PWD uv run python scripts/eval/run_refusal_ab.py            # gate A/B on refusal set
PYTHONPATH=$PWD uv run python scripts/eval/run_benchmark.py --suite vn --n 10 \
  --refusal-set data/eval/refusal_set.json --judge-provider gemini     # in-corpus + refusal
```
Note: the agent is slow per question on cloud LLMs (p50 ~minutes); use small `--n`.
The ablation (`run_ablation.py`) needs a sturdier `decide` model than flash-lite to
finish in reasonable time.

### Known follow-ups (not done)
- Langfuse online (`LANGFUSE_ENABLED`) + feedback capture → DPO dataset.
- Lower-drop-floor (~0.52) gate so it cuts distractors without hurting in-corpus recall.
- VN-medical embedding (replace generic `nomic-embed-text` / `bge-m3`).
- Faster eval harness / internal-model benchmark path.
