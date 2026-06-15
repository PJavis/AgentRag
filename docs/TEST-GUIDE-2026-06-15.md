# Test Guide — Streaming fix, harder benchmark, ontology, config (2026-06-15)

Branch **`feat/ragas-langfuse-reranker`** (pushed). Commits this round: `74ca7a2..ad1ca12`.
Five changes from the "do it all" pass. Most need no new content to verify — except the real-corpus / real-gold-set items, which are yours.

> TL;DR: §1 (streaming fix — the big one) → §2 (run the harder benchmark, no new content) → §3 (re-seed ontology) → §4 (config + your .env latency knobs).

---

## 0. What changed

| # | Change | Commit | Why it matters |
|---|---|---|---|
| 1 | **Streaming chat runs the LangGraph** | `74ca7a2` | CRAG / fast-path / critique / abstain now run for **streaming users** (they ran on nothing before — real bug) |
| 2 | **Benchmark that can fail** (`--group-size`, `--refusal-set`) | `4cff43f` | RAPTOR + distractors finally activate; out-of-corpus refusal/hallucination now measurable |
| 3 | **Ontology** 59→76 terms, 0 nulls | `a1bb6d3` | all 15 systems covered; better domain routing + query expansion |
| 4 | **Config** rerank ON, AGENT_MAX_STEPS 4→3, VN-embed warning | `ad1ca12` | free precision win + faster defaults + steer off weak nomic embeddings |

---

## 1. Streaming → graph fix (the important one)

**Before:** the streaming endpoint (`/chat/execute-stream`, what the UI actually uses) ran a separate hand-rolled loop that **bypassed** the whole graph — no CRAG, no adaptive fast-path, no critique, no abstain. Everything built this session only ran on the non-streaming path.
**After:** streaming runs the full graph, then streams the grounded answer.

### Unit test (no stack)
```bash
uv run pytest tests/agent/test_chat_stream_graph.py -v   # 2 pass
uv run pytest tests/agent/ -q                            # 29 pass
```

### Live verify (UI, port 3000) — with the RAG flags ON in .env
Open a notebook chat (which streams). You should now see, **on streamed answers**, the same signals the non-streaming path had:
- simple single-domain Q → `⚡ Fast path` chip (adaptive routing now fires on stream)
- out-of-corpus / unanswerable Q (e.g. *"Thuốc Zxylopraxin-9 dùng để làm gì?"*) → the assistant **abstains** ("Tôi không tìm thấy thông tin...") instead of hallucinating — the graph's abstain/critique now runs on stream
- Trace dialog shows the critique stage / corrective tags on streamed answers

If you DON'T see chips/abstain on streamed answers, confirm you're on `ad1ca12+` and the RAG flags are enabled (`CRAG_ENABLED`, `ADAPTIVE_ROUTING_ENABLED`).

> Tradeoff (intended): streaming now waits for the full graph before the first answer char (answer is replayed in ~40-char chunks). A `status: retrieve` frame keeps the UI responsive. Live LLM token-streaming was traded for graph-feature parity + correctness.

---

## 2. Run the harder benchmark (no new content needed)

Your old benchmark couldn't fail — index = only gold contexts (no distractors), 1 passage/doc (RAPTOR skipped on EN), all answerable (refusal never tested). Two new flags fix that.

```bash
# grouped corpus (multi-chunk docs → RAPTOR + distractors activate)
#   + out-of-corpus refusal eval (does it abstain or hallucinate?)
uv run python scripts/eval/run_benchmark.py --suite both --n 20 \
  --group-size 4 --refusal-set data/eval/refusal_set.json
```

What to read in the output:
- **`--group-size 4`** merges 4 gold contexts per ingested doc → docs become multi-chunk → RAPTOR builds summary nodes + reranking/CRAG have distractors to filter. Compare against a plain run (`--group-size 0`) to see if CR/RAPTOR/CRAG now move precision/citation (they had nothing to do before).
- **`report["refusal"]`** + a printed table → `refusal_rate` = fraction of out-of-corpus questions correctly abstained. **This is a new, high-stakes medical safety metric** (hallucinating a confident answer to an unknown drug is the dangerous failure). Low refusal_rate = it's making things up.

Pure-logic units (no stack): `uv run pytest tests/eval/test_refusal.py -v` (5 pass).
Seed set: `data/eval/refusal_set.json` (15 out-of-corpus Qs — trivia, unrelated tech, fabricated entities). Add your own.

> Needs the live stack (Postgres + ES + DeepSeek/judge key + Ollama). Long run.

---

## 3. Re-seed the ontology

The 12 previously-null `system_tag`s are now `da_he`, plus 17 new conditions (59→76 terms).

```bash
make seed-ontology      # seeds custom_terms.yaml + backfills ES tags
```

**⚠ Important — the seeder is insert-only (idempotent).** On a DB that's **already seeded**, `make seed-ontology` ADDS the 17 new terms but does **NOT update** the 12 existing rows whose tag changed null→da_he. To apply the tag fixes too, wipe + re-seed:
```bash
# wipe the ontology table, then re-seed fresh:
uv run python -c "import asyncio; from src.agentrag.database import AsyncSessionLocal; from sqlalchemy import text; \
asyncio.run((lambda: (lambda s: s)(None))()) " 2>/dev/null  # (or just psql)
psql "$DATABASE_URL" -c "TRUNCATE ontology_terms;" && make seed-ontology
```
On a **fresh** DB (`make reset-data` then ingest) the fixes apply automatically. After re-seeding, `make backfill-tags` propagates the tags onto existing ES chunks.

Verify:
```bash
curl -s localhost:8000/on/api/ontology/systems | jq      # 15 systems
uv run python -c "import yaml; d=yaml.safe_load(open('data/ontology/custom_terms.yaml')); \
print('terms', len(d), 'nulls', sum(1 for t in d if t.get('system_tag') is None))"   # 76, 0
```

---

## 4. Config + your .env latency knobs

`config.py` defaults changed (`RETRIEVAL_RERANK_ENABLED` recommended on in `.env.example`, `AGENT_MAX_STEPS` 4→3). **But your live `.env` overrides these** — it currently has:
- `AGENT_MAX_STEPS=6` → lower to `3` for a faster p50 (fewer serial decide-loops)
- `AGENT_MODEL=gemini-2.5-pro` → the slow answer model. Route `answer` to flash instead (in `LLM_TASK_MODEL_MAP`, with `LLM_ROUTING_ENABLED=true`) — big tail-latency cut, quality flat given the near-ceiling baseline. Validate correctness doesn't regress.
- `RETRIEVAL_RERANK_ENABLED=true` → already on (good).
- `EMBEDDING_MODEL=nomic-embed-text` → **weak on Vietnamese.** For quality, switch to **bge-m3 via TEI** (`EMBEDDING_PROVIDER=openai`, `EMBEDDING_MODEL=BAAI/bge-m3`, `EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1/`, `make serve-embed`). **Changes the vector dim → wipe + re-ingest** (`make reset-data`, re-upload).

These are your runtime choices — I didn't touch your `.env`.

---

## 5. What's still yours (can't be done in code)

- **Real VN-medical corpus** — upload actual textbooks. The benchmark grouped mode (§2) is ready to measure it.
- **Real VN-medical gold set** — Q/context/answer triples from your textbooks. Add a `DATASETS`/`SUITES` entry in `src/agentrag/eval/benchmark_datasets.py` + a local loader. Drop me the content and I'll wire it.

---

## 6. Reference
- Strategy memo + why this work: this session's analysis (the RAG is near-ceiling on an easy off-domain benchmark; real value = content + a benchmark that can fail + latency + the streaming fix).
- Prior guide: `docs/TEST-GUIDE-2026-06-10.md` (the 5 RAG-enhancement flags).
- Design specs: `docs/superpowers/specs/2026-06-10-*.md`.
