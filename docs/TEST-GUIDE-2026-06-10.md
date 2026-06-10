# Test Guide — RAG Enhancements + UI Signals (2026-06-10)

Everything below is on branch **`feat/ragas-langfuse-reranker`** (pushed to origin, HEAD `f53e0ad`).
This session added 5 RAG capabilities + a UI layer that surfaces them. **All new behavior is behind default-OFF flags** — with flags off the app behaves exactly as before. To test the new stuff you turn the flags on, re-ingest, and look for the signals.

> TL;DR fastest path: §1 (stack up) → §2 (run unit tests, no keys needed) → §3 (enable flags + re-ingest one doc) → §4 (chat + watch the chips/trace) → §6 (ablation benchmark for the real numbers).

---

## 0. What you're testing

| Feature | Flag (in `.env`) | Needs re-ingest? | How you see it |
|---|---|---|---|
| Contextual Retrieval | `CONTEXTUAL_RETRIEVAL_ENABLED=true` | **Yes** | citation hover shows a "context" line above the excerpt |
| RAPTOR summary layer | `RAPTOR_ENABLED=true` | **Yes** | citation hover shows `Σ Summary · L1` badge on summary nodes |
| CRAG critique + correction | `CRAG_ENABLED=true` | No | `🧠 Verified` / `↻ Self-corrected` chip + Critique stage in Trace |
| Multi-hop chaining | `AGENT_MULTIHOP_ENABLED=true` | No | `hop` tags in Trace tool list |
| Adaptive fast-path | `ADAPTIVE_ROUTING_ENABLED=true` | No | `⚡ Fast path` chip, faster simple answers |
| Semantic cache | `SEMANTIC_CACHE_ENABLED=true` | No | `⚡ Instant · cached` chip on a repeated/similar query |

---

## 1. Bring up the stack

Prereq: Docker + Docker Compose, and a populated `.env` (copy from `.env.example`, fill at least `DEEPSEEK_API_KEY` or `OPENAI_API_KEY`, and a `GEMINI_API_KEY` if you want the Gemini judge).

```bash
git fetch && git checkout feat/ragas-langfuse-reranker && git pull

# First time only — brings up infra (postgres, elasticsearch, valkey, tei, ollama),
# syncs python deps, installs frontend deps, runs migrations:
make install

# Day-to-day: infra + app (api + worker + frontend) in background:
make docker-up          # infra containers
make up-bg              # api + worker + frontend (logs in .run/)
make logs               # tail them
# stop everything:  make stop   (app)   /   make docker-down (infra)
```

Infra-only + run app from source instead (handy for backend iteration):
```bash
make docker-up
make api        # FastAPI on :8000   (or: uv run uvicorn main:app --reload)
make worker     # ARQ worker (REQUIRED so async StructMem/RAPTOR extraction runs)
make frontend   # Next.js on :3000
```

The **worker must be running** — Contextual Retrieval, RAPTOR, and StructMem extraction run as background ARQ jobs (status `searchable → enriching → done`).

---

## 2. Run the test suites (no API keys / no live stack needed)

These prove the new code is wired correctly before you touch live data.

```bash
# Backend — all new + touched units (pure, no Postgres/ES needed):
uv run pytest tests/agent tests/retrieval tests/services tests/orchestration \
  tests/ingestion/test_contextualizer.py tests/ingestion/test_raptor.py \
  tests/ingestion/test_es_mapping_fields.py tests/eval/test_ablation_runner.py -q
# expect: all pass

# Frontend — Vitest:
cd frontend && npm test
# expect: the new files pass (MessageSignals / TraceDialog / CitationHoverCard /
#   CostCharts / skeleton / cost-charts / signal-types).
# KNOWN pre-existing reds (NOT from this work — ignore):
#   - src/lib/locales/index.test.ts  (locale-parity for 9 old keys + unused-key)
#   - e2e/*.spec.ts                  (Playwright files vitest mis-collects)
cd ..
```

---

## 3. Enable the flags + re-ingest

Edit `.env` — turn the flags on and route the new ingest tasks to DeepSeek (cheap, prompt-cached):

```bash
# Feature flags
CONTEXTUAL_RETRIEVAL_ENABLED=true
RAPTOR_ENABLED=true
CRAG_ENABLED=true
AGENT_MULTIHOP_ENABLED=true
ADAPTIVE_ROUTING_ENABLED=true
SEMANTIC_CACHE_ENABLED=true

# Route the two new ingest-time tasks to DeepSeek (needs routing ON):
LLM_ROUTING_ENABLED=true
# add "contextualize" and "raptor_summary" to LLM_TASK_MODEL_MAP, e.g.:
LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","decide":"llama3.2:3b","domain_router":"llama3.2:3b","followup":"llama3.2:3b","plan":"deepseek-v4-pro","schema_discovery":"deepseek-v4-pro","sql_compile":"deepseek-v4-pro","synthesize":"deepseek-v4-flash","answer":"deepseek-v4-flash","mindmap":"deepseek-v4-flash","summary":"deepseek-v4-flash","contextualize":"deepseek-v4-flash","raptor_summary":"deepseek-v4-flash"}
```

> If you skip the routing keys, `contextualize`/`raptor_summary` fall back to the default agent model (likely local Ollama) — works, just slower.

Restart api + worker so they pick up the `.env`. Then **re-ingest** (CR + RAPTOR only apply to docs ingested AFTER the flags are on):

- **Easiest:** upload a fresh PDF/doc through the UI (a notebook → Add source). Watch the source card status go `searchable → enriching → done`. RAPTOR needs a doc with **≥8 chunks** (`RAPTOR_MIN_LEAVES`) to build summaries — use a real multi-page document.
- Existing docs keep their old flat chunks until re-ingested (delete + re-add, or re-run ingest).

---

## 4. Manual feature checks (in the chat UI, port 3000)

Open a notebook with the re-ingested doc and try these. Look **under each AI answer** for the chip row, and click **Trace** for the node graph.

| Try this | Expect |
|---|---|
| Short factual single-domain Q, e.g. *"Triệu chứng của nhồi máu cơ tim là gì?"* | `⚡ Fast path` chip; quick answer; Trace shows the fast-path explainer (no decide loop) |
| Ask the **same/similar** question again right after | `⚡ Instant · cached` chip on the 2nd answer |
| A broad/summary Q, e.g. *"Tóm tắt chương về tim mạch"* | hover a `[n]` citation → a `Σ Summary · L1` badge on a RAPTOR node |
| Hover any citation | a muted **context line** above the excerpt (Contextual Retrieval) |
| A question your corpus barely covers | `🧠 Verified` (passed) or `↻ Self-corrected` (CRAG re-retrieved); Trace → Critique stage + a `corrective` tool row |
| A dependent multi-step Q | Trace tool list shows `hop` tags (multi-hop chaining) |
| Open Trace on any answer | per-tool `cached` / mode badges, multi-hop/corrective tags |
| Go to **/cost** page → **Charts** tab | recharts: cost-over-time area + spend-by-model bar; cards show skeletons on first load |

Nothing should look *broken* with flags off — the chips simply don't appear.

---

## 5. Quick API smoke (optional, without the UI)

```bash
# health of providers:
curl -s localhost:8000/health/providers | jq

# a chat turn (adjust to your auth — disable auth in .env for local, or pass a token):
curl -s -X POST localhost:8000/api/chat/execute \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"<id>","message":"Triệu chứng nhồi máu cơ tim?","context":{"sources":{},"notes":{}}}' | jq '.messages[-1] | {reasoning_path, semantic_cache_hit, domain_route, timings_ms}'
# expect reasoning_path "fast"/"semantic", and (when CRAG ran) timings_ms.critique > 0
```

---

## 6. The real quality gate — ablation benchmark

This measures whether the flags actually beat baseline (correctness / precision / latency). **Long run** (hours — it re-ingests per config), needs the full live stack (Postgres + ES + DeepSeek key + Ollama):

```bash
# full matrix (baseline / cr / cr_raptor / cr_raptor_crag / full):
uv run python scripts/eval/run_ablation.py --suite both --n 20

# quick smoke first (2 configs, fewer questions):
uv run python scripts/eval/run_ablation.py --only baseline,full --n 10
```

Writes `docs/eval/benchmark_ablation_<date>.md`. **Acceptance gates** for the `full` row:
- answer_correctness > 0.792 (target ≥0.85)
- contextual_precision > 0.819 (target ≥0.88)
- faithfulness ≥ 0.80 (no regression)
- p50 latency < 10s (with local 3B orchestration; keep Ollama alive as a systemd service for clean latency numbers)

**Only enable a flag in production once its ablation row beats baseline.** If a config doesn't help, leave that flag off.

---

## 7. Rollback / safety

- All flags default **OFF** → set them back to `false` in `.env` and restart = original behavior, instantly. No data migration to undo.
- RAPTOR/CR only *added* summary nodes + fields to Elasticsearch; they don't alter existing leaf chunks. To purge: re-create the ES index (`make reset-data` wipes data, or delete the source).
- The branch is not merged to `master` — nothing in prod changes until you merge.

---

## 8. Troubleshooting

- **No chips appear** → flags off, or the worker hasn't finished extraction (status not `done`), or you're querying a doc ingested *before* enabling flags. Re-ingest.
- **`⚡ Instant·cached` never shows** → semantic cache is in-process per worker + only for unfiltered/default-scope queries; a domain filter or document scope bypasses it. Ask the same broad question twice in one session.
- **No `Σ Summary` citations** → the doc had <8 chunks (RAPTOR skipped), or RAPTOR_ENABLED was off at ingest time.
- **Ingest stuck at `enriching`** → the ARQ worker isn't running (`make worker`) or Ollama died; `make logs`.
- **Ablation rows all equal baseline** → likely a flag-name typo in the child env, or the corpus wasn't re-ingested for that config (the runner forces `STRUCTMEM_INGEST_MODE=sync` per config — give it time).
- **Frontend test reds** → the only expected reds are `src/lib/locales/index.test.ts` and the `e2e/*.spec.ts` collection errors; both predate this work.

---

## 9. Reference

- RAG design + plan: `docs/superpowers/specs/2026-06-10-rag-enhancement-design.md`, `docs/superpowers/plans/2026-06-10-rag-enhancement.md`
- UI design + plan: `docs/superpowers/specs/2026-06-10-ui-enhancement-design.md`, `docs/superpowers/plans/2026-06-10-ui-enhancement.md`
- Per-module details: `src/agentrag/<module>/README.md` (all 19 documented)
- Prior benchmark baseline: `docs/eval/benchmark_kg_vs_nokg_2026-06-06_vi.md`
