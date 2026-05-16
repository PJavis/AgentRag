# S4 — Reasoning Plane / Execution Plane Split

> "Few agents, many workers/services." Make the seam explicit, move
> boundary leaks, add missing service facades. Surgical not big-bang.

## Goals

1. **Name the planes**: a single `ARCHITECTURE.md` + per-package `README.md`
   stating which side a module belongs to and its contract.
2. **Move boundary leaks** (3 quick wins):
   - `FederatedRetriever` no longer owns `DomainRouter`. Router belongs to
     the Reasoning Plane — its decision is passed *into* the retriever as
     `system_override`/`specialty_override`.
   - `ElasticsearchRetriever` no longer owns `LLMReranker`. Reranker is a
     Reasoning-plane post-processor; retriever returns raw hits.
   - `KnowledgeService` splits into:
     - `ReasoningKnowledge` (query expansion, tool selection, normalization)
     - `RetrievalToolset` (thin wrapper over `AgentTools` for execution).
3. **Add missing service facades** so the Execution Plane has one entry per
   concern:
   - `EmbeddingService` — wraps `build_embedding_provider()` factory; only
     way reasoning code embeds text.
   - `VisionService` — wraps `ImageParser`; only way reasoning code calls
     vision.
   - `StorageService` — facade over `PostgresStore` + `ElasticsearchStore`
     for CRUD that's currently scattered.
4. **DI container** — `ServiceContainer` singleton constructed once at app
   startup, lazily wires concrete services from settings. Reasoning code
   fetches by interface (Protocol), never instantiates concrete classes.
5. **Worker contract** — ARQ functions stay where they are; document the
   contract: "stateless, idempotent, accept primitive kwargs, return JSON."

## Non-goals

- No directory rename. Modules keep their paths.
- No new orchestration framework. LangGraph backend stays.
- No new transport. Workers still ARQ; services still in-process.
- No new tests for code we don't change.

## Plane definition

```
┌──────────────────── REASONING PLANE ────────────────────┐
│ Decides WHAT to do. Owns state machines, prompts,        │
│ LLM decision loops. Stateless across calls.              │
│                                                          │
│  agent/service.py            — AgentService (loop)       │
│  agent/graph_service.py      — LangGraph StateGraph      │
│  agent/planner.py            — sub-query planner         │
│  agent/critic.py             — self-critique             │
│  orchestration/domain_router.py                          │
│  structured/query_classifier.py                          │
│  structured/pipeline.py       — SQL reasoning            │
│  services/reasoning_knowledge.py (NEW)                   │
└──────────────────────────────────────────────────────────┘
            ▼ calls services by Protocol
┌──────────────────── EXECUTION PLANE ────────────────────┐
│ Does the IO. Stateless workers + service facades.        │
│ No prompts, no decision branching.                       │
│                                                          │
│  services/llm_gateway.py     — LLMGateway                │
│  services/embedding_service.py (NEW)                     │
│  services/vision_service.py  (NEW)                       │
│  services/storage_service.py (NEW)                       │
│  services/retrieval_service.py (NEW, thin facade)        │
│  services/context_assembly_service.py                    │
│  retrieval/elasticsearch_retriever.py (no reranker/router)│
│  retrieval/federated.py (filters-only, no router)        │
│  ingestion/*  (parsers, chunkers, stores)                │
│  worker/functions.py (ARQ jobs)                          │
└──────────────────────────────────────────────────────────┘
```

## Files changed

| File                                                   | Change |
|--------------------------------------------------------|--------|
| `src/agentrag/services/embedding_service.py`           | NEW    |
| `src/agentrag/services/vision_service.py`              | NEW    |
| `src/agentrag/services/storage_service.py`             | NEW    |
| `src/agentrag/services/retrieval_service.py`           | NEW    |
| `src/agentrag/services/reasoning_knowledge.py`         | NEW    |
| `src/agentrag/services/container.py`                   | NEW    |
| `src/agentrag/services/protocols.py`                   | NEW    |
| `src/agentrag/retrieval/federated.py`                  | router moved out |
| `src/agentrag/retrieval/elasticsearch_retriever.py`    | reranker moved out (or kept as opt-in) |
| `src/agentrag/agent/service.py`                        | use ServiceContainer |
| `src/agentrag/agent/graph_service.py`                  | use ServiceContainer |
| `src/agentrag/adapter/app.py`                          | bootstrap container in `startup` |
| `ARCHITECTURE.md`                                      | NEW (root) |
| `README.md`                                            | new §1.x linking to ARCHITECTURE |
| `tests/integration/test_s4_planes.py`                  | NEW |

## Tasks

T1. Protocols module (`services/protocols.py`) — Embedding, Vision, Storage,
    Retrieval, Reranker protocols (Python `Protocol` from typing).
T2. EmbeddingService facade.
T3. VisionService facade.
T4. StorageService facade.
T5. RetrievalService facade — wraps FederatedRetriever; this is the only
    way reasoning code retrieves.
T6. Move DomainRouter out of FederatedRetriever; router is Reasoning-plane.
    FederatedRetriever now takes filters only (no router dependency).
T7. ReasoningKnowledge — split from KnowledgeService.
T8. ServiceContainer — singleton wiring all services from settings.
T9. Wire ServiceContainer into AgentService + GraphAgentService bootstrap.
T10. ARCHITECTURE.md + README §1.x.
T11. Tests (`tests/integration/test_s4_planes.py`) — protocol satisfaction,
     no reasoning-from-execution imports (lint), container wiring.
T12. Push + tag `s4-complete`.

## Acceptance

1. `grep -r "ElasticsearchRetriever()" src/agentrag/agent/` returns nothing
   (reasoning never instantiates concrete retriever).
2. `FederatedRetriever.__init__` takes no `router` arg.
3. `ServiceContainer.get(EmbeddingService)` returns the same instance across
   calls in a request.
4. All existing tests still pass.
5. New plane tests pass.
