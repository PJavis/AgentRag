# PAM Context — System Design Document

> **Business Knowledge Layer for LLMs**
> Ingest documents, build a knowledge graph, and answer questions with citations via a Claude-powered retrieval agent.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Infrastructure](#3-infrastructure)
4. [Data Models](#4-data-models)
5. [Ingestion Pipeline](#5-ingestion-pipeline)
6. [Retrieval & Search](#6-retrieval--search)
7. [Agent System](#7-agent-system)
8. [API Layer](#8-api-layer)
9. [Frontend](#9-frontend)
10. [Configuration](#10-configuration)
11. [Evaluation Framework](#11-evaluation-framework)
12. [CI/CD & Quality](#12-cicd--quality)
13. [Key Design Decisions](#13-key-design-decisions)

---

## 1. System Overview

PAM Context is a RAG (Retrieval-Augmented Generation) platform that ingests business documents, indexes them across multiple storage backends, builds a temporal knowledge graph, and provides an AI agent that answers questions with source citations.

### Core Capabilities

- **Document ingestion** from Markdown files, Google Docs, and Google Sheets
- **Layout-aware parsing** via Docling with hybrid chunking
- **Multi-backend search** combining Elasticsearch vector/BM25 fusion, entity VDB, and Neo4j knowledge graph
- **LLM-powered agent** with tool-use loop (Claude) that selects retrieval strategies per query
- **Temporal knowledge graph** with bitemporal timestamps and chunk-level diffing on re-ingestion
- **Token-budgeted context assembly** (LightRAG-inspired) with deduplication and truncation
- **Query classification** with two-tier routing (rules + LLM fallback) across 5 retrieval modes

### Tech Stack Summary

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12+, FastAPI, SQLAlchemy 2.0 (async) |
| Frontend | React 18, Vite, TypeScript, Tailwind CSS |
| Primary DB | PostgreSQL 16 |
| Search / Vector DB | Elasticsearch 8.15 (RRF, kNN) |
| Knowledge Graph | Neo4j 5.26 + Graphiti |
| Cache | Redis 7 |
| LLM (Agent) | Claude Sonnet (Anthropic SDK) |
| LLM (Graph extraction) | Claude Sonnet via Graphiti |
| Embeddings | OpenAI text-embedding-3-large (1536d) |
| Parsing | Docling 2.0 |
| Reranking | sentence-transformers cross-encoder |

---

## 2. Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│   React 18 + Vite + Tailwind                                │
│   ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌─────────────┐ │
│   │ Chat Page│ │ Documents  │ │  Admin   │ │Graph Explorer│ │
│   └─────┬────┘ └─────┬──────┘ └────┬─────┘ └──────┬──────┘ │
└─────────┼─────────────┼────────────┼───────────────┼────────┘
          │  HTTP / SSE │            │               │
┌─────────┼─────────────┼────────────┼───────────────┼────────┐
│         ▼             ▼            ▼               ▼        │
│                    FastAPI (port 8000)                       │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│   │  /chat   │ │ /ingest  │ │  /stats  │ │  /graph  │      │
│   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘      │
│        │             │            │             │            │
│   ┌────▼─────┐ ┌─────▼──────┐ ┌──▼───┐ ┌──────▼──────┐     │
│   │  Agent   │ │  Pipeline  │ │  DB  │ │ Graph Query │     │
│   └────┬─────┘ └─────┬──────┘ └──┬───┘ └──────┬──────┘     │
│        │             │            │             │            │
│ ┌──────▼──────────────▼────────────▼─────────────▼────────┐ │
│ │                  Service Layer                           │ │
│ │  HybridSearch │ Embedder │ GraphitiService │ VDB Store  │ │
│ └──────┬────────────┬───────────┬──────────────┬──────────┘ │
└────────┼────────────┼───────────┼──────────────┼────────────┘
         │            │           │              │
    ┌────▼────┐  ┌────▼────┐ ┌───▼────┐  ┌──────▼──────┐
    │   ES    │  │ OpenAI  │ │ Neo4j  │  │ PostgreSQL  │
    │ 8.15   │  │  API    │ │  5.26  │  │    16       │
    └─────────┘  └─────────┘ └────────┘  └─────────────┘
                                          ┌─────────────┐
                                          │   Redis 7   │
                                          └─────────────┘
```

### Module Structure

```
src/pam/
├── common/          # Config, models, database, cache, logging, utils
├── ingestion/       # Pipeline orchestrator + sub-components
│   ├── connectors/  #   Data source adapters (Markdown, Google Docs/Sheets)
│   ├── parsers/     #   Docling layout-aware parser
│   ├── chunkers/    #   Hybrid token-aware chunker
│   ├── embedders/   #   OpenAI embedder with LRU cache
│   ├── extractors/  #   Entity extraction
│   └── stores/      #   PostgreSQL, Elasticsearch, Entity/Relationship VDB
├── retrieval/       # Search backends (ES hybrid, Haystack), rerankers
├── agent/           # Claude tool-use loop, tools, query classifier, context assembly
├── graph/           # Neo4j/Graphiti service, extraction, entity types
└── api/             # FastAPI app, routes, middleware, deps, auth
    └── routes/      #   chat, search, ingest, documents, graph, admin, auth

web/                 # React frontend
├── src/
│   ├── pages/       #   ChatPage, DocumentsPage, AdminDashboard, GraphExplorerPage
│   ├── components/  #   ChatInterface, MessageBubble, graph/, chat/, ui/
│   ├── hooks/       #   useChat, useGraphExplorer, useDocuments, useAuth
│   └── api/         #   Centralized fetch client + types
└── e2e/             # Playwright tests

eval/                # LLM-as-judge evaluation framework
```

---

## 3. Infrastructure

### Docker Compose Services

| Service | Image | Port | Purpose | Health Check |
|---------|-------|------|---------|-------------|
| PostgreSQL | postgres:16-alpine | 5433 | Document & segment storage, auth, sync log | pg_isready |
| Elasticsearch | elasticsearch:8.15.0 | 9200 | Vector indices, BM25 search, entity/relationship VDB | curl /_cluster/health |
| Redis | redis:7-alpine | 6379 | Search result cache, session cache | redis-cli ping |
| Neo4j | neo4j:5.26-community | 7474, 7687 | Knowledge graph (Graphiti) | cypher-shell |

**Elasticsearch** runs in single-node mode with security disabled (dev), 2GB Java heap.

**Neo4j** runs with APOC plugin, 512MB heap, 1GB max memory, 512MB page cache.

All services use named Docker volumes for persistence.

### Database Schema (7 Migrations)

| Migration | Tables/Changes |
|-----------|---------------|
| 001 - Initial | projects, documents, segments, sync_log + indexes |
| 002 - Tasks | ingestion_tasks (async job tracking) |
| 003 - Auth | users, user_project_roles (RBAC) |
| 004 - Entities | extracted_entities (structured entity storage) |
| 005 - Hash index | Concurrent index on content_hash, role uniqueness constraint |
| 006 - Graph sync | graph_synced boolean on documents |
| 007 - Modified at | modified_at timestamp on documents |

---

## 4. Data Models

### Core ORM Models (PostgreSQL)

```
┌─────────────┐       ┌──────────────┐       ┌──────────────────┐
│   Project   │       │   Document   │       │     Segment      │
├─────────────┤       ├──────────────┤       ├──────────────────┤
│ id (UUID)   │       │ id (UUID)    │──1:N─▶│ id (UUID)        │
│ name        │       │ source_type  │       │ document_id (FK) │
│ description │       │ source_id    │       │ content          │
│ created_at  │       │ title        │       │ content_hash     │
│ updated_at  │       │ content_hash │       │ segment_type     │
└─────────────┘       │ graph_synced │       │ section_path     │
                      │ modified_at  │       │ position         │
                      │ last_synced  │       │ metadata (JSONB) │
                      │ created_at   │       │ version          │
                      │ updated_at   │       │ created_at       │
                      └──────┬───────┘       └──────────────────┘
                             │
                      ┌──────▼───────┐       ┌──────────────────┐
                      │   SyncLog    │       │ ExtractedEntity  │
                      ├──────────────┤       ├──────────────────┤
                      │ id (UUID)    │       │ id (UUID)        │
                      │ document_id  │       │ entity_type      │
                      │ action       │       │ entity_data(JSON)│
                      │ segments_aff │       │ confidence       │
                      │ details(JSON)│       │ source_segment   │
                      │ created_at   │       │ source_text      │
                      └──────────────┘       └──────────────────┘

┌─────────────┐       ┌──────────────────┐
│    User     │       │ UserProjectRole  │
├─────────────┤       ├──────────────────┤
│ id (UUID)   │──1:N─▶│ id (UUID)        │
│ email       │       │ user_id (FK)     │
│ name        │       │ project_id (FK)  │
│ google_id   │       │ role (enum)      │
│ is_active   │       └──────────────────┘
└─────────────┘

┌──────────────────┐
│  IngestionTask   │
├──────────────────┤
│ id (UUID)        │
│ status (enum)    │
│ folder_path      │
│ total/processed  │
│ succeeded/failed │
│ skipped          │
│ results (JSONB)  │
│ error            │
│ timestamps       │
└──────────────────┘
```

### Elasticsearch Indices

| Index | Purpose | Key Fields |
|-------|---------|-----------|
| `pam_segments` | Primary search | content, embedding (1536d kNN), meta.* (nested) |
| `pam_entities` | Entity VDB | name, type, description, embedding |
| `pam_relationships` | Relationship VDB | src_entity, tgt_entity, rel_type, keywords, embedding |

### Central DTO: KnowledgeSegment

The `KnowledgeSegment` Pydantic model flows through the entire pipeline:

```
Connector → Parser → Chunker → Embedder → KnowledgeSegment → Stores
                                           ├── id, content, embedding
                                           ├── source_type, source_id
                                           ├── section_path, segment_type
                                           ├── position, metadata
                                           └── document_title, document_id
```

---

## 5. Ingestion Pipeline

### Pipeline Flow

```
                    ┌─────────────────┐
                    │  API: POST      │
                    │  /ingest/folder │
                    └────────┬────────┘
                             │ creates IngestionTask (202 Accepted)
                             ▼
                    ┌─────────────────┐
                    │ Task Manager    │
                    │ (asyncio spawn) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Connector     │
                    │ list_documents()│
                    └────────┬────────┘
                             │ for each document:
                    ┌────────▼────────┐
                    │ Content Hash    │◄── Skip if unchanged
                    │ Check (SHA-256) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Docling Parser  │
                    │ (layout-aware)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Hybrid Chunker  │
                    │ (512 tokens)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ OpenAI Embedder │
                    │ (batch + cache) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌───────────┐
     │ PostgreSQL │  │   Elastic  │  │ Graphiti  │
     │  (atomic)  │  │(fault-tol.)│  │(non-block)│
     └────────────┘  └────────────┘  └─────┬─────┘
                                           │
                                    ┌──────▼──────┐
                                    │ Entity/Rel  │
                                    │  VDB Store  │
                                    └─────────────┘
```

### Component Details

| Component | Class | Responsibility |
|-----------|-------|---------------|
| **Connectors** | `MarkdownConnector`, `GoogleDocsConnector`, `GoogleSheetsConnector` | Fetch raw documents from sources, provide content hashes |
| **Parser** | `DoclingParser` | Layout-aware parsing (PDF, DOCX, Markdown) via Docling |
| **Chunker** | `HybridChunker` (Docling) | Token-aware chunking (configurable, default 512 tokens), preserves section paths |
| **Embedder** | `OpenAIEmbedder` | Batch embedding (100/request), LRU cache (10K entries), exponential backoff retry |
| **PostgresStore** | `PostgresStore` | Atomic upsert: document + segments in single transaction |
| **ElasticsearchStore** | `ElasticsearchStore` | Bulk indexing with fault tolerance (failures logged, not raised) |
| **Graph Extraction** | `GraphExtraction` | Per-chunk Graphiti episode ingestion with chunk-level diffing |
| **Entity/Rel VDB** | `EntityRelationshipVDBStore` | LightRAG 3-VDB pattern: entities + relationships in ES |
| **Diff Engine** | `DiffEngine` | Chunk-level comparison for re-ingestion (add/remove/unchanged) |
| **Task Manager** | `TaskManager` | Background asyncio task spawning, status tracking, progress updates |

### Fault Isolation Strategy

Storage writes are ordered by criticality with isolated failure domains:

1. **PostgreSQL** (atomic, must succeed) — document + segments committed together
2. **Elasticsearch** (fault-tolerant) — failures logged, pipeline continues
3. **Graph** (non-blocking) — failures logged, `graph_synced` stays false for retry

### Re-ingestion & Diff Engine

On re-ingestion, the pipeline performs chunk-level diffing:

1. Compare old vs new segments by content_hash
2. Identify: added chunks, removed chunks, unchanged chunks
3. **PostgreSQL**: Delete old segments, insert new ones
4. **Elasticsearch**: Remove stale, index new
5. **Graph**: Remove episodes for deleted/changed chunks, add episodes for new/changed chunks
6. **SyncLog**: Persist diff summary for audit trail

---

## 6. Retrieval & Search

### Search Architecture

```
                         User Query
                             │
                   ┌─────────▼──────────┐
                   │  Query Classifier  │
                   │  (rules + LLM)    │
                   └─────────┬──────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
   ┌────────────────┐ ┌───────────┐ ┌──────────────────┐
   │ ES Hybrid      │ │ Entity    │ │ Knowledge Graph  │
   │ (BM25 + kNN)   │ │ VDB Search│ │ (Graphiti)       │
   └───────┬────────┘ └─────┬─────┘ └────────┬─────────┘
           │                │                 │
           │         ┌──────▼──────┐          │
           │         │ Relationship│          │
           │         │ VDB Search  │          │
           │         └──────┬──────┘          │
           │                │                 │
           └────────────────┼─────────────────┘
                            │
                   ┌────────▼────────┐
                   │ Context Assembly│
                   │ (4-stage)       │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │  Agent Response │
                   └─────────────────┘
```

### Search Backends

**1. Hybrid Search (ES RRF)**
- Combines BM25 text matching + kNN vector similarity
- Reciprocal Rank Fusion for score merging
- Optional cross-encoder reranking (sentence-transformers)
- Redis caching (15 min TTL)
- Filters: source_type, project, date range

**2. Haystack Pipeline (optional)**
- Alternative backend via `USE_HAYSTACK_RETRIEVAL=true`
- Same `SearchService` protocol interface (polymorphic)
- Pluggable Haystack 2.x components

**3. Entity VDB Search**
- Searches `pam_entities` ES index by embedding similarity
- Returns entity name, type, description, confidence

**4. Relationship VDB Search**
- Searches `pam_relationships` ES index
- Returns src_entity, tgt_entity, relationship type, keywords

**5. Knowledge Graph (Graphiti)**
- Natural language → Graphiti semantic search
- Returns entity relationships with temporal validity
- Bitemporal: valid_at / invalid_at timestamps

**6. Smart Search (combined)**
- Concurrent 4-way search: ES + Entity VDB + Relationship VDB + Graph
- Results merged via context assembly pipeline

### Query Classification (Two-Tier)

| Tier | Method | Purpose |
|------|--------|---------|
| Rules | Pattern matching | Fast classification via regex (PascalCase → entity, "when/changed" → temporal, etc.) |
| LLM | Claude Haiku | Fallback for ambiguous queries when rules confidence < threshold |

**Retrieval Modes**: ENTITY, CONCEPTUAL, TEMPORAL, FACTUAL, HYBRID

### Keyword Extraction (Dual-Level)

Per query, Claude Haiku extracts:
- **High-level keywords**: Themes and concepts (for ES search)
- **Low-level keywords**: Specific entities and terms (for VDB/graph search)

### Context Assembly (4-Stage Pipeline)

```
Raw Results → Collect → Truncate → Dedup → Build → Structured Prompt
```

| Stage | Action | Budget |
|-------|--------|--------|
| **Collect** | Normalize results into uniform dicts | — |
| **Truncate** | Per-category token limits, sorted by score | entities: 4K, relationships: 6K |
| **Dedup** | Remove duplicate segment_ids | — |
| **Build** | Structured Markdown prompt assembly | max: 12K tokens |

Per-item truncation cap: 500 tokens. Inspired by LightRAG's context assembly approach.

---

## 7. Agent System

### Tool-Use Loop

```python
# Simplified agent loop (~50 lines)
for i in range(MAX_TOOL_ITERATIONS):  # max 5
    response = client.messages.create(
        model=agent_model,
        system=system_prompt,
        messages=conversation,
        tools=tool_definitions,
    )
    if response.stop_reason == "end_turn":
        break
    if response.stop_reason == "tool_use":
        results = execute_tools(response.tool_calls)
        conversation.append(results)
```

The agent is **not** built on LangGraph or any framework — it is a direct tool-use loop with the Anthropic SDK.

### Available Tools (8)

| Tool | Purpose |
|------|---------|
| `search_knowledge` | Full-text + vector hybrid search via ES |
| `get_document_context` | Fetch complete document by ID |
| `get_change_history` | Query sync_log for recent ingestion changes |
| `query_database` | DuckDB SQL over CSV/Parquet/JSON (SELECT only, guarded) |
| `search_entities` | Entity VDB semantic search |
| `search_knowledge_graph` | Graphiti natural language graph search |
| `get_entity_history` | Bitemporal entity snapshots |
| `smart_search` | Concurrent 4-way search (ES + Entity VDB + Relationship VDB + Graph) |

### Agent Response

Each response includes:
- **Answer** with inline citations: `[Source: title > section](url)`
- **Citations** extracted from tool results
- **Token usage** (input + output)
- **Latency** (end-to-end)
- **Retrieval mode** used and confidence score

### DuckDB Analytics

The `query_database` tool provides SQL analytics over structured data:
- Scans a configurable data directory for CSV, Parquet, and JSON files
- Guards: SELECT only, no multi-statement queries, max 1000 rows
- Enables analytical queries the agent can compose on the fly

---

## 8. API Layer

### Endpoints

| Route | Method | Endpoint | Purpose |
|-------|--------|----------|---------|
| **Chat** | POST | `/chat` | Non-streaming agent response |
| | POST | `/chat/stream` | Server-Sent Events streaming |
| **Search** | POST | `/search` | Direct hybrid search (bypasses agent) |
| **Ingest** | POST | `/ingest/folder` | Start async folder ingestion (202) |
| | GET | `/ingest/tasks/{id}` | Poll task status |
| | GET | `/ingest/tasks` | List recent tasks (paginated) |
| | POST | `/ingest/sync-graph` | Manual graph sync retry |
| **Documents** | GET | `/documents` | List documents (cursor pagination) |
| | GET | `/documents/{id}` | Document details + segments |
| | GET | `/segments/{id}` | Single segment with parent context |
| | GET | `/stats` | System statistics |
| **Graph** | GET | `/graph/status` | Neo4j status + entity counts |
| | GET | `/graph/entities` | List all entities (paginated) |
| | GET | `/graph/entity/{name}` | Entity + 1-hop neighborhood |
| | GET | `/graph/neighborhood/{name}` | Graph neighborhood data |
| | GET | `/graph/history/{entity}` | Temporal change history |
| | GET | `/graph/sync-logs` | Ingestion diff events |
| **Admin** | POST | `/admin/roles` | Assign user-project role |
| **Auth** | GET | `/auth/status` | Check auth status |
| | POST | `/auth/dev-login` | Dev token (no password) |
| | GET | `/auth/me` | Current user info |

### Streaming Protocol (SSE)

```
event: status   → { type: "status", content: "Searching..." }
event: token    → { type: "token", content: "The answer is..." }
event: citation → { type: "citation", data: { title, document_id, segment_id } }
event: done     → { type: "done", conversation_id, metadata: { token_usage, latency_ms } }
event: error    → { type: "error", message: "..." }
```

### Pagination

Keyset-based (cursor) pagination to avoid offset performance cliff:
- Cursor = base64(sort_value, id)
- Default ordering: `updated_at DESC, id DESC`

### Middleware

- **CorrelationIdMiddleware**: Injects `X-Correlation-ID` into every request via contextvars
- **RequestLoggingMiddleware**: Structured request/response logging via structlog

---

## 9. Frontend

### Pages

| Page | Route | Features |
|------|-------|---------|
| **Chat** | `/` | Streaming responses, citations with tooltips, source viewer panel, search filters, token usage metrics |
| **Documents** | `/documents` | Folder ingestion form, real-time progress polling, document list, completion summary |
| **Admin** | `/admin` | System stats cards, entity breakdowns, graph sync button, recent tasks table |
| **Graph Explorer** | `/graph/explore` | Neo4j NVL visualization, D3 force layout, entity sidebar, temporal timeline, diff overlay |
| **Login** | `/login` | Dev login form (when `AUTH_REQUIRED=true`) |

### Key Frontend Patterns

**State Management**: React hooks only (no Redux/Zustand). Custom hooks per domain:
- `useChat` — streaming, conversation state, cancellation (AbortController)
- `useGraphExplorer` — entity load, NVL conversion, expand/focus/deselect
- `useDocuments` — fetch, refresh, error handling
- `useIngestionTask` — polling with exponential backoff (1.5s → 30s)
- `useAuth` — JWT restore, optimistic decode, server validation

**Graph Visualization**: Neo4j NVL library with D3 force simulation
- Node color by entity type (7 color assignments)
- Node size by connection count (degree)
- Single-click → select, double-click → expand neighborhood
- Deep-linking: `?entity=name` auto-loads entity

**Markdown Rendering**: react-markdown with plugins
- GFM tables, LaTeX math (remark-math + rehype-katex), syntax highlighting
- Custom CodeBlock with copy button
- Citation link detection and SPA navigation for graph links

**Streaming**: Custom async generator parsing SSE manually (not EventSource API)
- Graceful fallback: streaming → non-streaming on failure
- AbortController for cancellation

---

## 10. Configuration

All configuration is env-var driven via Pydantic Settings with `.env` file support.

### Key Configuration Groups

| Group | Variables | Defaults |
|-------|----------|---------|
| **PostgreSQL** | `DATABASE_URL` | `psycopg://pam:pam@localhost:5432/pam_context` |
| **Elasticsearch** | `ELASTICSEARCH_URL`, `ELASTICSEARCH_INDEX` | `localhost:9200`, `pam_segments` |
| **OpenAI** | `OPENAI_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_DIMS` | —, `text-embedding-3-large`, `1536` |
| **Anthropic** | `ANTHROPIC_API_KEY`, `AGENT_MODEL` | —, `claude-sonnet-4-6` |
| **Neo4j** | `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | `bolt://localhost:7687`, `neo4j`, — |
| **Graphiti** | `GRAPHITI_MODEL`, `GRAPHITI_EMBEDDING_MODEL` | `claude-sonnet-4-6`, `text-embedding-3-small` |
| **Redis** | `REDIS_URL`, `REDIS_SEARCH_TTL`, `REDIS_SESSION_TTL` | `localhost:6379`, `900s`, `86400s` |
| **Auth** | `AUTH_REQUIRED`, `JWT_SECRET`, `JWT_ALGORITHM` | `false`, —, `HS256` |
| **Search** | `USE_HAYSTACK_RETRIEVAL`, `RERANK_ENABLED`, `RERANK_MODEL` | `false`, `false`, `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Entity VDB** | `ENTITY_INDEX`, `RELATIONSHIP_INDEX` | `pam_entities`, `pam_relationships` |
| **Smart Search** | `SMART_SEARCH_ES_LIMIT`, `SMART_SEARCH_GRAPH_LIMIT` | `5`, `5` |
| **Context Assembly** | `CONTEXT_ENTITY_BUDGET`, `CONTEXT_RELATIONSHIP_BUDGET`, `CONTEXT_MAX_TOKENS` | `4000`, `6000`, `12000` |
| **Mode Router** | `MODE_CONFIDENCE_THRESHOLD`, `LLM_FALLBACK_ENABLED` | `0.7`, `true` |
| **Ingestion** | `CHUNK_SIZE_TOKENS`, `INGEST_ROOT` | `512`, — |
| **App** | `LOG_LEVEL`, `CORS_ORIGINS` | `INFO`, `["http://localhost:5173"]` |
| **Frontend** | `VITE_GRAPH_ENABLED` | `false` |

---

## 11. Evaluation Framework

Located in `eval/`, the evaluation framework measures retrieval and answer quality.

### Components

| File | Purpose |
|------|---------|
| `questions.json` | 10 evaluation questions (3 simple, 4 medium, 3 complex) |
| `run_eval.py` | Orchestrator: runs search + chat, computes scores |
| `judges.py` | LLM-as-judge scoring via Claude |

### Scoring Dimensions (0–1 scale)

| Dimension | What It Measures |
|-----------|-----------------|
| **Factual Accuracy** | Correctness vs. expected answer |
| **Citation Presence** | Source attribution per major claim |
| **Completeness** | Coverage of key points |

### Usage

```bash
python eval/run_eval.py [--api-url URL] [--output results.json]
```

Outputs per-question scores, per-difficulty aggregates, and overall retrieval recall.

---

## 12. CI/CD & Quality

### CI Pipeline (GitHub Actions)

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Lint    │   │  Type    │   │  Migrate │   │  Test    │   │ Frontend │
│  (ruff)  │──▶│  (mypy)  │──▶│ (alembic)│──▶│ (pytest) │──▶│ (pnpm)   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
                                               80% min
                                               coverage
```

- **Python 3.12** with pip caching
- **PostgreSQL 16** service container for integration tests
- Ruff check + format validation
- MyPy strict mode with Pydantic plugin
- Alembic migration validation
- pytest with 80% coverage minimum (excludes integration tests)
- Frontend: pnpm install (frozen lockfile) + TypeScript build

### CD Pipeline

- Triggers on push to main or semver tags (v*.*.*)
- Docker build & push to `ghcr.io`
- Image tagging: branch refs, semver patterns, Git SHAs

### Code Quality

- **Ruff**: Python 3.12 target, 120 char line length
- **MyPy**: Strict mode, Pydantic plugin
- **Pre-commit**: Trailing whitespace, EOF fixer, YAML validation, large file detection, debug statement check
- **Coverage**: 80% threshold, excludes `__init__.py` and abstract methods

---

## 13. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Simple tool-use loop over LangGraph** | ~50 lines of code, full control, no framework coupling |
| **Async-first throughout** | SQLAlchemy asyncio, FastAPI async, asyncio task spawning for ingestion |
| **Content hashing (SHA-256)** | Change detection, embedding cache dedup, re-ingestion optimization |
| **Fault-isolated storage writes** | PG must succeed; ES failures logged but non-fatal; Graph failures retry later |
| **Polymorphic search backends** | SearchService protocol enables swapping ES ↔ Haystack without API changes |
| **Keyset pagination** | Avoids offset performance degradation at scale |
| **LightRAG-inspired context assembly** | Token-budgeted, per-category truncation prevents context window overflow |
| **Two-tier query classification** | Rules are fast and free; LLM fallback handles edge cases |
| **Chunk-level graph diffing** | Re-ingestion only touches changed chunks, preserving stable graph structure |
| **Bitemporal timestamps** | Tracks both when facts were true (valid_at) and when they were recorded |
| **Per-request agent isolation** | New agent instance per API call ensures thread safety for concurrent requests |
| **No auth in dev mode** | `AUTH_REQUIRED=false` by default; JWT + Google OAuth ready for production |
| **3-VDB pattern (LightRAG)** | Separate indices for segments, entities, and relationships enable targeted retrieval |
| **Dual-level keyword extraction** | High-level themes for broad search + low-level entities for precise graph queries |

---
