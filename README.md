## Setup

Copy [.env.example](/home/nguyenquocdung/PAM/.env.example) to `.env` and choose providers separately for:

- `EMBEDDING_PROVIDER`: `openai`, `gemini`, `hf_inference`, `ollama`
- `EXTRACTION_PROVIDER`: `openai`, `gemini`, `hf_inference`, `ollama`
- `GRAPH_EMBEDDING_PROVIDER`: `openai`, `gemini`, `hf_inference`, `ollama`
- `AGENT_PROVIDER` (optional): `openai`, `gemini`, `hf_inference`, `ollama`

Common combinations:

- Remote + local:
  `EMBEDDING_PROVIDER=hf_inference`, `EXTRACTION_PROVIDER=ollama`
- Fully OpenAI:
  `EMBEDDING_PROVIDER=openai`, `EXTRACTION_PROVIDER=openai`, `GRAPH_EMBEDDING_PROVIDER=openai`
- Mostly local:
  `EMBEDDING_PROVIDER=ollama`, `EXTRACTION_PROVIDER=ollama`, `GRAPH_EMBEDDING_PROVIDER=ollama`

Agent runtime model can be configured independently with:

- `AGENT_PROVIDER`
- `AGENT_MODEL`
- `AGENT_BASE_URL`
- `AGENT_TEMPERATURE`

If `AGENT_*` is not set, `/chat` falls back to `EXTRACTION_*`.

## Quick Start

```bash
cp .env.example .env
docker compose up -d
uv sync
uv run alembic upgrade head
uv run uvicorn main:app --reload
```

Ingest Markdown docs from a local folder:

```bash
curl -X POST http://127.0.0.1:8000/ingest/folder \
  -H "Content-Type: application/json" \
  -d '{"folder_path":"data/test_docs", "graph_ingest_mode":"async"}'
```

Check provider config quickly:

```bash
curl http://127.0.0.1:8000/config/validate
```

## Benchmark

Use the diagnostic script before a full ingest:

```bash
python3 scripts/benchmark_ingest.py data/test_docs/SYSTEM_DESIGN.md
python3 scripts/benchmark_ingest.py data/test_docs/SYSTEM_DESIGN.md --embed
python3 scripts/benchmark_graph.py data/test_docs/SYSTEM_DESIGN.md --max-chunks 5
python3 scripts/benchmark_retrieval.py data/benchmarks/retrieval_baseline.json --top-k 5
python3 scripts/benchmark_agent.py data/benchmarks/agent_baseline.json --repeat 1
```

This reports chunk counts, optional embedding latency, and graph extraction timing with the current `.env`.

The app and benchmark scripts now validate provider settings at startup. If a selected provider is missing its token, base URL, or model, startup fails immediately with a direct error.

If `EMBEDDING_PROVIDER=ollama` and you see `unsupported value: NaN`, the selected Ollama embedding model is unstable in your runtime. Use a dedicated embedding model such as `nomic-embed-text`, `nomic-embed-text-v2-moe`, or `mxbai-embed-large`.

## Retrieval

Search now supports:

- `sparse`: BM25-style lexical retrieval in Elasticsearch
- `dense`: vector kNN retrieval in Elasticsearch
- `hybrid`: BM25 + dense fusion using RRF
- `hybrid_kg`: chunk retrieval + graph retrieval fusion using RRF
- `graph_lookup`: temporal graph fact lookup (used by agent tool loop)
- optional LLM reranking on top of retrieved candidates

During ingest, Elasticsearch now stores:

- `pam_segments`: chunk-level retrieval index
- `pam_entities`: entity vector index (`name`, `type`, `description`, `embedding`)
- `pam_relationships`: relationship vector index (`src_entity`, `tgt_entity`, `rel_type`, `keywords`, `embedding`)

Example:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"temporal knowledge graph", "mode":"hybrid_kg", "top_k":5, "rerank":true}'
```

Filter by one document:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"MoneyPrinter", "mode":"hybrid_kg", "top_k":5, "document_title":"test"}'
```

Reranking config in `.env`:

- `RETRIEVAL_RERANK_ENABLED=true|false`
- `RETRIEVAL_RERANK_TOP_N=20`
- `RETRIEVAL_RERANK_BACKEND=llm_chat|local_cross_encoder`
- `RETRIEVAL_RERANK_PROVIDER` / `RETRIEVAL_RERANK_MODEL` (optional)

If reranker-specific settings are empty, the system falls back to `AGENT_*`, then `EXTRACTION_*`.
If response has `"reranked": false`, inspect `"rerank_reason"` to know whether rerank was disabled, model output schema was invalid, or provider call failed.

For local cross-encoder reranking (recommended for Ollama users without `/api/rerank`):

```env
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_TOP_N=20
RETRIEVAL_RERANK_BACKEND=local_cross_encoder
RETRIEVAL_RERANK_MODEL=dengcao/bge-reranker-v2-m3
```

## Agent (Local LLM)

`/chat` already runs a tool-use loop + 4-stage context assembly:

1. retrieve
2. dedupe
3. rank/trim
4. citation-pack

Minimal local check:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"MoneyPrinter có tính năng gì?", "document_title":"MoneyPrinter"}'
```

`document_title` is optional. `/chat` does not require `document_id`.
If `document_title` is omitted, retrieval searches across all indexed documents.

Conversation mode (multi-turn) is supported with Postgres + Redis cache:

```bash
# create conversation
curl -X POST http://127.0.0.1:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{"title":"MoneyPrinter discussion"}'

# chat with conversation_id
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"<uuid>", "question":"tiếp tục nhé", "document_title":"MoneyPrinter"}'

# list messages
curl http://127.0.0.1:8000/conversations/<uuid>/messages
```

You should verify:

- `tool_trace` has at least one retrieval step
- `context` is non-empty
- `citations` map to `document_title/section_path/position/content_hash` in `context`

Benchmark flow (after infra is up):

```bash
python3 scripts/benchmark_retrieval.py data/benchmarks/retrieval_baseline.json --top-k 5
```

## Architecture Rollout

Current rollout follows an incremental production path (see [ADR 0001](/home/nguyenquocdung/PAM/docs/adr/0001-target-architecture-rollout.md)):

- `AgentService` acts as Supervisor orchestration.
- `KnowledgeService` centralizes retrieval/tool execution.
- `ContextAssemblyService` is a dedicated context assembly facade.
- `SecurityService` is the query-time policy gate (v1).
- `LLMGateway` is the central LLM invocation facade (v1, ready for routing/cost tracking).

Target rollout phases:

1. Phase A: service boundaries + backward compatibility.
2. Phase B: policy enforcement, intent-aware retrieval, stage-level observability.
3. Phase C: LLM routing/cost control and specialized worker agents.

## Reset

```bash
docker compose down -v --remove-orphans
rm -rf data/neo4j_data data/es_data data/postgres_data
rm -rf .cache/pam/graph
docker compose up -d

uv sync
uv run alembic upgrade head
uv run uvicorn main:app --reload --port 8000

```
