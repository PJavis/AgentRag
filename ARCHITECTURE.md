# AgentRag — Architecture

> S4 — Reasoning Plane / Execution Plane split. "Few agents, many workers/services."

## Overview

AgentRag is a Vietnamese-medical RAG platform with two distinct planes:

```
┌────────────────────────── REASONING PLANE ──────────────────────────┐
│ Decides WHAT to do. Owns state machines, prompts, LLM decision      │
│ loops. Stateless across turns.                                       │
│                                                                      │
│  agent/graph_service.py      GraphAgentService (LangGraph — active) │
│  agent/service.py            AgentService (internal helper only)    │
│  agent/tools.py              AgentTools tool registry               │
│  orchestration/domain_router.py     S5 domain classifier            │
│  services/reasoning_knowledge.py    pure helpers (expand_query…)    │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  calls services by Protocol
┌────────────────────────── EXECUTION PLANE ──────────────────────────┐
│ Does the IO. Stateless workers + service facades. No prompts,       │
│ no decision branching.                                               │
│                                                                      │
│  services/container.py             ServiceContainer (singleton)     │
│  services/protocols.py             Protocol contracts               │
│  services/llm_gateway.py           LLMGateway                       │
│  services/embedding_service.py     EmbeddingService                 │
│  services/vision_service.py        VisionService                    │
│  services/storage_service.py       StorageService                   │
│  services/retrieval_service.py     RetrievalService                 │
│  services/context_assembly_service.py                               │
│  retrieval/elasticsearch_retriever.py                               │
│  retrieval/federated.py            (filter-only, no router)         │
│  ingestion/parsers/* chunkers/* stores/*                            │
│  worker/functions.py               ARQ jobs                         │
└──────────────────────────────────────────────────────────────────────┘
```

## The rules

1. **Reasoning never instantiates concrete IO classes.** Use the
   `ServiceContainer`:

   ```python
   from src.agentrag.services.container import get_container
   container = get_container()
   hits = await container.retrieval.search(query=q, mode="hybrid")
   ```

2. **Execution never decides.** A service may translate inputs and call IO,
   but must not branch on prompts, run LLM decisions, or own routing logic.
   - `FederatedRetriever` takes filters, never classifies.
   - `RetrievalService` has no prompts.
   - `EmbeddingService` does not pick which texts to embed.

3. **Domain routing is reasoning.** Reasoning code calls
   `DomainRouter.classify(query)` and forwards picks to
   `RetrievalService.search(system_override=…, specialty_override=…)`.

4. **Protocols define the contract.** Type Reasoning dependencies against
   `services/protocols.py`, not concrete classes. Tests inject mocks via
   `container.override(retrieval=mock)`.

5. **Workers are pure functions.** ARQ jobs in `worker/functions.py`
   accept primitive kwargs, return JSON, never depend on request state.

## Boundary leaks (known, not fixed in S4)

- `ElasticsearchRetriever` owns `LLMReranker` — reranker should be a
  Reasoning post-processor. Documented; not refactored to avoid
  destabilising existing rerank tests + call sites.
- `KnowledgeService` still mixes reasoning (`expand_query`,
  `normalize_tool_call`) and execution (`bootstrap_search`). New code
  should use `services/reasoning_knowledge.py` + `services/retrieval_service.py`
  instead.
- `AgentTools` instantiates `ElasticsearchRetriever()` directly. Future
  work: have it pull from the container.

## Worker contract

An ARQ job is a function `f(ctx, *kwargs) → dict`:

- **Stateless**: no module-level mutable state.
- **Idempotent**: re-running with same kwargs is safe.
- **Primitive kwargs only**: strings, ints, lists/dicts of JSON
  primitives. No SQLAlchemy sessions, no event loops, no callables.
- **Returns JSON**: must be `json.dumps`-able.

See `src/agentrag/worker/functions.py` for the live set:
`graph_ingest`, `vision_extract`, `consolidate`, `chat_memory`.

## File map

| Concern              | Path                                          | Plane |
|----------------------|-----------------------------------------------|-------|
| Agent loop (active)  | `src/agentrag/agent/graph_service.py`         | R     |
| Agent helpers        | `src/agentrag/agent/service.py`               | R     |
| Tool registry        | `src/agentrag/agent/tools.py`                 | R     |
| Domain router        | `src/agentrag/orchestration/domain_router.py` | R     |
| Reasoning helpers    | `src/agentrag/services/reasoning_knowledge.py`| R     |
| Service container    | `src/agentrag/services/container.py`          | E     |
| LLM gateway          | `src/agentrag/services/llm_gateway.py`        | E     |
| Embedding service    | `src/agentrag/services/embedding_service.py`  | E     |
| Vision service       | `src/agentrag/services/vision_service.py`     | E     |
| Storage service      | `src/agentrag/services/storage_service.py`    | E     |
| Retrieval service    | `src/agentrag/services/retrieval_service.py`  | E     |
| Federated retriever  | `src/agentrag/retrieval/federated.py`         | E     |
| ES retriever         | `src/agentrag/retrieval/elasticsearch_retriever.py` | E |
| Parsers / chunkers   | `src/agentrag/ingestion/parsers/*`            | E     |
| Stores               | `src/agentrag/ingestion/stores/*`             | E     |
| ARQ workers          | `src/agentrag/worker/functions.py`            | E     |

R = Reasoning Plane, E = Execution Plane.

## Storage split: PG segments vs ES agentrag_segments

Both stores receive every chunk on ingest. **Not redundant** — each serves a distinct read path.

| Store | Role | Readers |
|---|---|---|
| Postgres `segments` | Durable, ordered, joinable. Source of truth for full-text reconstruction and per-document operations. | `adapter/routers/insights.py::_source_full_text` (concat by `position`), `adapter/routers/sources.py::_segment_count` (health), `agent/tools.py::get_document_segments` (agent tool, joins `Segment ⨝ Document` by title), `graph/vision_jobs.py::upsert_image_segments` (max-position numbering, deletion cascade in `delete_source`) |
| Elasticsearch `agentrag_segments` | Retrieval-only: BM25 + dense kNN + RRF hybrid. | `ingestion/stores/elasticsearch_store.py::{sparse,dense,hybrid}_search`, `retrieval/elasticsearch_retriever.py`, `generation/summary_service.py` (4 sites), `generation/mindmap_service.py` |

**Rule:** PG is the durable spine + tool-API substrate. ES is the retrieval projection. Anything that needs ordered enumeration, deletion cascade, or content_hash dedupe goes through PG. Anything that needs ranked relevance goes through ES. Do NOT add a retrieval read against PG, and do NOT add a row-level join against ES.

ES is rebuildable from PG (see `scripts/` — add `reindex_es.py` if a re-projection is ever needed). PG is NOT rebuildable from ES (loses original FK + position monotonicity).
