# RAG Enhancement — Contextual Retrieval, RAPTOR, CRAG, Adaptive Routing, Semantic Cache

**Date**: 2026-06-10
**Status**: Approved (verbal, brainstorming session)
**Author**: dungnq + Claude
**Branch**: `feat/ragas-langfuse-reranker`
**Scope**: One comprehensive spec, 5 integrated workstreams, single coordinated build.

## Context

The system passes all benchmark gates (n=80, VN+EN, with StructMem/KG):

| Metric | Threshold | Current (with KG) | Status |
|---|---|---|---|
| Contextual recall | ≥0.70 | 0.900 | strong |
| **Contextual precision** | ≥0.70 | **0.819** | weakest retrieval metric |
| Faithfulness | ≥0.80 | 0.950 | strong |
| **Answer correctness** | ≥0.70 | **0.792** | weakest quality metric |
| Citation accuracy | ≥0.70 | 0.884 | good |
| Failure rate | <0.05 | 0.000 | — |
| **Latency p50 / p95 / p99** | ref | **24.9s / 164.6s / 198s** | broken UX |

The retrieval pipeline already has: soft-HyDE + decompose query rewrite
(`retrieval/query_rewriter.py:50-104`), parallel sub-query retrieval via
`asyncio.gather` (`agent/graph_service.py:181-189`), 3-layer domain-partitioned
KB (ontology terms → domain-tagged chunks → federated routing), StructMem KG
(factual/relational/synthesis entries in `agentrag_memory_doc`), and
`bge-reranker-v2-m3` cross-encoder rerank (RRF k=60).

Gaps that cap quality and latency:

1. **Flat chunks** — no hierarchical/summary layer (no RAPTOR, no parent-child).
   Broad/synthesis questions ("tóm tắt", multi-section) have no node to match at
   the right scale.
2. **No `critique` node** — README promises `plan→decide→tool→assemble→answer→critique`,
   but no critique exists. No grounding re-check, no corrective re-retrieval.
3. **No multi-hop chaining** — decomposed sub-query results are flat-RRF merged,
   never fed forward as context to dependent sub-queries.
4. **No Contextual Retrieval** — chunks embedded raw, no LLM-prepended situating
   context (Anthropic technique → large precision/recall gain).
5. **No adaptive depth** — every query runs the full decide-loop regardless of
   complexity → the p50/p95 latency killer.
6. **Cache-key bug** — `_cache_key()` (`elasticsearch_retriever.py:25-28`) omits
   `dense_query`/HyDE variant/filters → cache collisions; module-level dict cache
   is non-distributed.

## Goals & Non-goals

**Goals** (benchmark-measured):
- Answer correctness 0.792 → **≥0.85**
- Contextual precision 0.819 → **≥0.88**
- Latency p50 24.9s → **<10s** (production config, local 3B orchestration)
- Hold faithfulness ≥0.80, recall ≥0.70, citation ≥0.70, failure <0.05.

**Non-goals**:
- No Neo4j/Cypher graph engine (StructMem stays vector-indexed).
- No reranker model swap (`bge-reranker-v2-m3` kept; only wiring/tuning).
- No Postgres schema change beyond what re-index needs.
- No frontend redesign (citations/trace UI unchanged; new fields are additive).

## Decisions (brainstorming output)

| ID | Decision | Rationale |
|---|---|---|
| D1 | Primary goal = quality + latency together | User pick. Both weak spots; highest ROI |
| D2 | Re-ingest existing KB allowed | User pick. CR + RAPTOR need re-embed |
| D3 | Ingest-time LLM = cloud (DeepSeek + prompt-cache) | User pick. Faster/better than local 3B, cheap with doc-prefix cache |
| D4 | One comprehensive spec, build all 5 together | User pick. Coherent end-state; ablation flags mitigate bisect risk |
| D5 | Every workstream feature-flagged | Safe rollout + benchmark ablation (CR/RAPTOR/CRAG on×off) |
| D6 | CRAG fires conditionally only | Avoid latency tax on the ~80% already-good answers |
| D7 | Adaptive fast-path gated high-confidence + single-domain | No quality regression on borderline questions |

## Architecture

### Plane assignment (respects S4 split, `ARCHITECTURE.md`)

```
Execution plane (new IO services, no branching):
  - ingestion/contextualizer.py     (WS1)
  - ingestion/raptor.py             (WS2)
  - services/semantic_cache.py      (WS5)

Reasoning plane (new decision logic, no IO):
  - agent/graph_service.py: critique node + adaptive edge   (WS3, WS4)
  - structured/query_classifier.py: complexity signal       (WS4)
```

### New end-to-end data flow

```
INGEST:
  parse → chunk
        → [WS1 CONTEXTUALIZE: LLM situate, doc as cached prefix]
        → embed (TEI bge-m3)
        → index leaf nodes (node_level=0)
        → [WS2 RAPTOR: cluster embeddings → summarize cluster → embed → index node_level≥1, until 1 root]
        → SectionTagger (propagate tags to summary nodes from children)
        → StructMem extract (unchanged)

QUERY:
  classify (+ WS4 complexity score)
        → [WS4 FAST PATH | FULL AGENT LOOP]
        → [WS5 semantic-cache check]
        → federated hybrid_kg retrieve (over leaf + summary nodes)
        → rerank (bge-reranker-v2-m3)
        → answer
        → [WS3 CRITIQUE: conditional relevance + grounding → corrective re-retrieve]
        → ground (citations)
```

---

## WS1 — Contextual Retrieval

**Technique** (Anthropic): before embedding each chunk, an LLM writes a 50–100
token context situating the chunk within its parent document. That context is
prepended to the text used for **both** dense embedding and BM25 indexing. The
embedded/indexed text is contextualized; the **displayed and cited text remains
the original** `content`.

**Why**: −35% retrieval failures (−49% combined with rerank, which we already
have). Directly targets the two weakest retrieval metrics (precision, and the
correctness that depends on it).

**Cost control**: per document, send the whole doc (or section window for very
large docs) once as a **DeepSeek cached prefix**; each per-chunk prompt is just
"situate this chunk: <chunk>" → only the chunk tokens are uncached. Keeps cost
small per the D3 decision.

**Components**:
- New `ingestion/contextualizer.py`: `Contextualizer.contextualize(doc_text, chunks) -> list[str]`.
  Uses `LLMGateway` task `"contextualize"` (new task → routable model, cloud default).
  Batches chunks under one cached doc prefix. SHA256 cache key per (model, doc_hash,
  chunk_hash) reusing the StructMem extract-cache pattern (`STRUCTMEM_CACHE_DIR`
  sibling dir) to make backfill idempotent.
- `ingestion/pipeline.py`: insert contextualize step **before** embed, behind
  `CONTEXTUAL_RETRIEVAL_ENABLED`.
- ES mapping (`ingestion/stores/elasticsearch_store.py`): add `context_text` (text,
  analyzed) field. Embedder input becomes `context_text + "\n\n" + content` when
  present; BM25 query matches across `content` + `context_text`.
- `content` field unchanged → citations, page-aware highlights, dedup hashes all
  keep working on the original text.

**Config**: `CONTEXTUAL_RETRIEVAL_ENABLED=true`, `CONTEXTUAL_RETRIEVAL_TASK=contextualize`,
`CONTEXTUAL_MAX_CONTEXT_TOKENS` (doc window cap).

**Failure handling**: contextualize returns `None`/empty on LLM error → chunk
indexed with `content` only (graceful degrade, never blocks ingest).

---

## WS2 — RAPTOR Summary Layer

**Technique**: after leaf chunks are embedded, recursively (a) cluster leaf
embeddings (UMAP dim-reduction → Gaussian Mixture soft clustering, RAPTOR paper
default), (b) summarize each cluster with DeepSeek, (c) embed each summary, (d)
index summaries as nodes with `node_level≥1`, repeat on summary embeddings until
one root remains. Retrieval is **collapsed-tree**: all nodes (leaf + every
summary level) live in the one `agentrag_segments` index and are retrieved
together by similarity. The paper shows collapsed-tree ≈ best tree-traversal
variant and is far simpler to serve.

**Why**: multi-scale matching. Broad/synthesis questions ("tóm tắt chương",
multi-section "so sánh") match a summary node instead of one arbitrary leaf →
targets answer correctness. Cheap at query time (just more nodes in the same
index).

**Components**:
- New `ingestion/raptor.py`: `RaptorBuilder.build(leaf_nodes) -> list[SummaryNode]`.
  - Cluster: `umap-learn` + `sklearn.mixture.GaussianMixture`. Small-doc guard:
    if `len(leaves) < RAPTOR_MIN_LEAVES` (e.g. 8) → skip (no value).
  - Summarize via `LLMGateway` task `"raptor_summary"` (cloud default).
  - Each `SummaryNode` carries `node_level`, `child_ids`, `system_tag`/`specialty_tag`
    propagated (union) from children, `segment_type="raptor_summary"`.
- `ingestion/pipeline.py`: insert RAPTOR step **after** leaf embed/index, behind
  `RAPTOR_ENABLED`. Summary nodes go through the same embed + index path.
- ES mapping: add `node_level` (integer, default 0), `child_ids` (keyword[]).
- Retriever (`elasticsearch_retriever.py`): optional `_balance_levels` so a single
  query doesn't return only summaries — cap summary share (reuse the existing
  `_balance_segment_types` machinery, add a level cap analogous to the 0.3 image cap).

**Config**: `RAPTOR_ENABLED=true`, `RAPTOR_MAX_LEVELS=3`, `RAPTOR_MIN_LEAVES=8`,
`RAPTOR_CLUSTER_SIZE`, `RAPTOR_SUMMARY_MAX_LEVEL_RATIO` (summary cap in results).

**Deps**: add `umap-learn`, `scikit-learn` to `pyproject.toml`.

**Failure handling**: clustering or summary failure on a doc → log + skip RAPTOR
for that doc; leaves still fully searchable.

---

## WS3 — CRAG Critique Node (conditional, multi-hop)

**Technique** (CRAG / Self-RAG): a new LangGraph node after `answer` evaluates
(a) **retrieval relevance** and (b) **answer grounding**, and triggers a bounded
corrective action only when needed.

- **Relevance evaluator**: reuse the cross-encoder rerank scores already computed
  for the top candidates → bucket `{correct | ambiguous | incorrect}` by
  `CRAG_SCORE_THRESHOLD`. No extra model call for the relevance signal.
- **Grounding check**: verify the answer's claims map to cited support using the
  existing citation map (`_build_packed_citations`, `agent/service.py:682-719`).
  Weak/uncited → flag.
- **Corrective action** (only on ambiguous/incorrect/weak-grounding):
  - Query rewrite: step-back abstraction + keyword refinement (extend
    `retrieval/query_rewriter.py`).
  - **One** bounded re-retrieve (`AGENT_CRITIQUE_MAX_RETRIES=1`), then StructMem
    (`hybrid_kg` entries/synthesis) fallback, then re-answer.
- **Conditional firing**: when top rerank score ≥ threshold AND grounding strong
  → pass straight through. ~80% of currently-passing answers pay zero extra
  latency. Honors D6.

**Multi-hop chaining** (folded here): the planner (`agent/service.py:494-542`)
gains a `dependent: bool` flag per sub-query. Dependent sub-queries run
**sequentially**, feeding sub-query-1's answer text into sub-query-2's retrieval
query, instead of the current flat `asyncio.gather` + RRF. Independent
sub-queries keep the existing parallel path.

**Components**:
- `agent/graph_service.py`: add `critique` node + conditional edge
  `answer → critique → {ground | corrective_retrieve → answer}`.
- `agent/service.py`: `_critique(state) -> CritiqueDecision`; extend `_plan_subqueries`
  with dependency detection; sequential chain executor for dependent sub-queries.

**Config**: `CRAG_ENABLED=true`, `CRAG_SCORE_THRESHOLD`, `CRAG_GROUNDING_ENABLED`,
`AGENT_CRITIQUE_MAX_RETRIES=1`, `AGENT_MULTIHOP_ENABLED=true`.

---

## WS4 — Adaptive Routing (latency)

**Technique**: the `classify` node emits a complexity score alongside the existing
intent. A query that is **high-confidence (≥0.85) + single-domain + short factual**
takes a **FAST PATH** — one `hybrid_kg` retrieve + single-shot answer — skipping
the `plan → decide ⟷ tool` loop entirely. Anything complex, multi-domain, or
low-confidence takes the current full agent loop.

**Why**: the decide-loop's repeated LLM round-trips are the dominant p50/p95 cost.
Simple lookups (the common case) shouldn't pay for iterative reasoning.

**Guard** (honors D7): fast-path only when the classifier is confidently simple.
Borderline → full path. Worst case for a misroute is a slightly thinner answer
on a single-domain factual question; faithfulness/grounding still enforced by WS3
when enabled.

**Components**:
- `structured/query_classifier.py`: add `complexity ∈ {simple, complex}` +
  `single_domain: bool` to the L1 rule output and L2 LLM schema.
- `agent/graph_service.py`: `classify` conditional edge → new `fast_answer` path
  (reuse `bootstrap` retrieve + `answer_node`, bypass `semantic_plan`/`decide`/`tool_exec`).

**Config**: `ADAPTIVE_ROUTING_ENABLED=true`, `ADAPTIVE_FASTPATH_MIN_CONFIDENCE=0.85`.

---

## WS5 — Semantic Cache + Cache-Key Fix

**Components**:
1. **Bug fix**: `_cache_key()` (`elasticsearch_retriever.py:25-28`) must include
   `dense_query` (HyDE-augmented variant), `filters`, and `rerank` flag. Current
   omission lets HyDE and non-HyDE queries collide.
2. **Semantic cache** — new `services/semantic_cache.py`:
   - Embed incoming query (reuse the already-TTL-cached embedding service).
   - Cosine ≥ `SEMANTIC_CACHE_THRESHOLD` (0.97) against recent query embeddings →
     return the cached retrieval result.
   - **Two-tier**: exact-key lookup first, then semantic. **Redis-backed** (the
     ARQ Redis already in `docker-compose.yml`) → distributed, replaces the
     module-level 512-item dict.
3. **Answer cache** (optional, `ANSWER_CACHE_ENABLED`): domain-scoped query-embedding
   → answer, short TTL, **invalidated on ingest** to that domain (hook in
   `ingestion/pipeline.py` completion).

**Config**: `SEMANTIC_CACHE_ENABLED=true`, `SEMANTIC_CACHE_THRESHOLD=0.97`,
`SEMANTIC_CACHE_TTL`, `SEMANTIC_CACHE_BACKEND=redis`, `ANSWER_CACHE_ENABLED=false`.

---

## Migration & re-ingest

- **ES mapping**: add `context_text`, `node_level`, `child_ids`. New index version
  + reindex (existing `Makefile`/admin reindex path).
- **Backfill**: one-time background job over the 3031 existing `agentrag_segments`:
  re-contextualize → re-embed → re-index → build RAPTOR per document. Reuses the
  existing async worker + `enriching → done` status (progressive-ingestion design,
  `2026-06-05-progressive-ingestion-design.md`). CR + RAPTOR are LLM-per-chunk →
  async only; upload never blocks.
- **Postgres**: unchanged.

## Testing & acceptance

- **Unit**: `tests/ingestion/test_contextualizer.py`, `test_raptor.py`;
  `tests/agent/test_critique.py`, `test_adaptive_routing.py`;
  `tests/services/test_semantic_cache.py` (incl. the cache-key regression).
- **Integration**: extend `tests/integration/` — full ingest with CR+RAPTOR,
  fast-path vs full-path routing, critique corrective loop.
- **Benchmark ablation** (extend existing DeepEval harness, `scripts/eval`):
  table over CR × RAPTOR × CRAG on/off + adaptive on/off. Acceptance gates:
  - correctness > 0.792 (target ≥0.85)
  - precision > 0.819 (target ≥0.88)
  - faithfulness ≥ 0.80 (no regression)
  - recall ≥ 0.70, citation ≥ 0.70, failure < 0.05
  - **p50 < 10s** (production config: local 3B orchestration, Ollama on systemd)
- **Rollout**: all flags default safe; enable per-workstream, re-benchmark, keep
  what wins.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| One-time re-ingest cost (3031 chunks × DeepSeek) | Doc-prefix prompt cache; async bg worker; accepted in D2/D3 |
| RAPTOR adds `umap-learn`/`scikit` deps + ingest latency | Async-only; `RAPTOR_MIN_LEAVES` skip for small docs |
| CR+RAPTOR slow ingest | Background worker + `enriching` status; upload non-blocking |
| Semantic-cache staleness | Ingest-time invalidation per domain; conservative 0.97 threshold |
| Adaptive misroute (complex → fast path) | High confidence gate + single-domain requirement + WS3 grounding net |
| CRAG adds latency | Conditional firing; bounded to 1 retry; reuses existing rerank scores |
| Integration bisect risk (build-together) | Per-workstream flags → ablation isolates regressions |

## Open questions (resolve during planning)

- RAPTOR summary node domain-tag propagation: union of child tags vs re-resolve
  via `SectionTagger` on the summary text? (Lean: union, cheaper.)
- Semantic-cache scope: per-(domain,user) vs global? (Lean: domain-scoped,
  user-agnostic for retrieval cache; answer cache user-agnostic too since answers
  are grounded, not personalized.)
- Whether to expose RAPTOR summary nodes as citable sources or retrieval-only
  signal. (Lean: retrieval-only first; cite the leaf children.)
