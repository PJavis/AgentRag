# Next Steps — AgentRag (2026-06-16)

Branch **`feat/ragas-langfuse-reranker`** (pushed, HEAD `3b0b1f7`). All code is shipped.
What's left is **your** decisions: configure → measure → decide → invest in the moat.

> The single most important action is **§2 — run the discriminating benchmark 3 ways.** Everything else waits on those numbers. Don't enable flags in prod by vibes; enable what the grouped benchmark proves.

---

## TL;DR priority

1. Set `.env` (§1) — gate ON for the experiment + latency knobs.
2. Run the 3-way benchmark (§2) — baseline vs gate vs full.
3. Decide prod flags from the numbers (§3).
4. Build the data moat (§4) — real VN-medical content + gold set. *This is the real value.*
5. Housekeeping (§5) — re-seed ontology, merge when validated.

---

## 1. Configure `.env`

### Experiment (to test what was just shipped)
```bash
RETRIEVAL_RERANK_ENABLED=true            # already on in your .env
RETRIEVAL_RELEVANCE_GATE_ENABLED=true    # NEW — prune distractors below the rerank floor
RETRIEVAL_RELEVANCE_FLOOR=0.3            # untuned default; sweep in §2 (try 0.2 / 0.3 / 0.4)
```
The gate only bites with `RETRIEVAL_RERANK_BACKEND=local_cross_encoder` (your default).

### Latency (real UX win — your live values are slow: p50 ~25s)
```bash
AGENT_MAX_STEPS=3                         # was 6 — fewer serial decide→tool round-trips
LLM_ROUTING_ENABLED=true
# route the answer to flash, not gemini-2.5-pro (the slow serial generation call):
LLM_TASK_MODEL_MAP={"classify":"...","answer":"deepseek-v4-flash", ...}
```

### Quality (optional — needs re-index, dims change)
```bash
# nomic-embed-text is weak on Vietnamese. For real use:
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1/   # make serve-embed
# then: make reset-data  + re-ingest (vector dim changed 768→1024)
```

---

## 2. Run the discriminating benchmark (3 ways)

Your old benchmark couldn't fail (index = only gold, no distractors). The grouped + refusal harness can. Run **the same command** under 3 configs and diff the table:

```bash
uv run python scripts/eval/run_benchmark.py --suite both --n 20 \
  --group-size 4 --refusal-set data/eval/refusal_set.json
```

| Config | `.env` flags |
|---|---|
| **baseline** | CR/RAPTOR/CRAG OFF, gate OFF |
| **gate** | rerank ON + `RETRIEVAL_RELEVANCE_GATE_ENABLED=true` (else OFF) |
| **full** | CR + RAPTOR + CRAG + gate all ON |

Each config re-ingests (sync) — long run (~hours), needs live stack (Postgres + ES + DeepSeek key + Ollama). Save each report to a dated file.

### What to read (the new metrics)
- **`contextual_precision`** — should go **UP** with the gate (distractors pruned). Baseline grouped was ~0.689.
- **`hedged_cited_rate`** (refusal eval) — should go **DOWN** with the gate (hedging answers stop citing distractors).
- **`hallucination_rate`** — must stay **~0** (the dangerous failure: confident fabrication). Your last run = ~0, i.e. it does NOT make things up.
- **`refusal_rate`** (clean abstain) — should go **UP** with the gate.
- **`answer_correctness` / `contextual_recall`** — must **NOT drop**. If they fall, the floor is too high (over-abstaining on real questions) → lower `RETRIEVAL_RELEVANCE_FLOOR`.

### Tune the floor
Sweep `RETRIEVAL_RELEVANCE_FLOOR` (0.2 / 0.3 / 0.4). Pick the highest floor that drops `hedged_cited_rate` + lifts `precision` **without** dropping `correctness`/`recall`.

---

## 3. Decide prod config from the numbers

Rule: **enable a flag only if its grouped-benchmark row beats baseline.** Don't trust the old easy benchmark or n=10 deltas (judge noise ±0.05).

Expected, pending your run:
- **Keep:** reranker, relevance gate (if §2 confirms precision↑ / clean abstain), adaptive routing + semantic cache (latency wins, quality-neutral).
- **Drop / leave OFF:** CR + RAPTOR — neutral-to-slightly-negative so far; confirm on the grouped set. RAPTOR summary nodes can dilute verbatim citation.
- **Corpus-aware SQL gate** (`STRUCTURED_REQUIRE_TABULAR=true`) — keep on; prose corpus never wastes the SQL path.

---

## 4. Build the data moat (the real value)

Per the strategy analysis: the RAG pipeline is near-ceiling on an *off-domain easy* benchmark. The metrics won't move much more from algorithm work. The actual product value is:

1. **Real VN-medical corpus** — upload actual textbooks (your DB is effectively ~2 PDFs today). The benchmark grouped mode is ready to measure it.
2. **Real VN-medical gold set** — Q / gold-context / reference-answer triples from your textbooks. Add a `DATASETS` + `SUITES` entry in `src/agentrag/eval/benchmark_datasets.py` + a local loader. **Drop the content and I wire the loader.**
3. **Study UX** — page-jump citations, mindmaps/summaries, MCQ — features med students actually use (not benchmark %).

This is where effort compounds. Everything in §1–3 is tuning a commodity pipeline.

---

## 5. Housekeeping

- **Re-seed ontology** (seeder is insert-only, so the 12 null→da_he tag fixes need a fresh seed):
  ```bash
  psql "$DATABASE_URL" -c "TRUNCATE ontology_terms;"
  make seed-ontology && make backfill-tags
  ```
- **Merge** `feat/ragas-langfuse-reranker` → `master` once §2/§3 validate (it's a big branch; nothing in it is enabled by default).
- **Streaming fix is live** — `chat_stream` now runs the full graph (CRAG/fast-path/abstain apply to streaming users). Verify in the UI: streamed answers show chips + abstain on out-of-corpus.

---

## What's already done (this branch)

- 5 RAG enhancements (CR / RAPTOR / CRAG / adaptive / semantic cache) — flag-gated, default OFF.
- UI signal surfacing (chips, trace, citation hovers, cost charts).
- **Streaming → graph fix** (real bug: streaming bypassed the graph).
- **Corpus-aware SQL gate** (prose corpora skip the SQL path).
- **Benchmark-that-fails** (grouped-corpus + refusal/hallucination metrics).
- **Relevance-floor gate** (prune distractors before answer).
- Ontology 59→76 terms; config defaults (rerank on, lower max-steps); 19 module READMEs.

Reference: `docs/TEST-GUIDE-2026-06-15.md`, `docs/TEST-GUIDE-2026-06-10.md`, `docs/superpowers/specs/2026-06-10-*.md`.
