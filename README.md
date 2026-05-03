# AgentRag

Nền tảng RAG (Retrieval-Augmented Generation) với hai luồng suy luận song song, hệ thống bộ nhớ phân cấp, và CLI tương tác.

- **Semantic path** — Hybrid retrieval (BM25 + vector + StructMem entries) + LLM synthesis
- **Structured path** — SQL reasoning trên dữ liệu trích xuất từ văn bản
- **Chat StructMem** — Bộ nhớ hội thoại semantic thay thế sliding-window history

---

## Mục lục

1. [Kiến trúc tổng quan](#1-kiến-trúc-tổng-quan)
2. [Storage Layer](#2-storage-layer)
3. [Yêu cầu hệ thống](#3-yêu-cầu-hệ-thống)
4. [Cài đặt & Khởi động](#4-cài-đặt--khởi-động)
5. [Cấu hình `.env`](#5-cấu-hình-env)
6. [API Reference](#6-api-reference)
7. [CLI](#7-cli)
8. [StructMem — Knowledge Extraction](#8-structmem--knowledge-extraction)
9. [Chat StructMem — Bộ nhớ hội thoại](#9-chat-structmem--bộ-nhớ-hội-thoại)
10. [Background Workers & Auto-scaler](#10-background-workers--auto-scaler)
11. [Structured SQL Reasoning](#11-structured-sql-reasoning)
12. [LLM Routing](#12-llm-routing)
13. [MCP Server](#13-mcp-server)
14. [Security Policy](#14-security-policy)
15. [Benchmark & Kiểm thử](#15-benchmark--kiểm-thử)
16. [Reset môi trường](#16-reset-môi-trường)
17. [Cấu trúc thư mục](#17-cấu-trúc-thư-mục)
18. [Module READMEs](#18-module-readmes)

---

## 1. Kiến trúc tổng quan

```
POST /chat
    │
    ├── SecurityService.validate_chat_request()
    ├── ChatMemoryService.retrieve()          ← Chat StructMem (nếu bật)
    ├── QueryIntentClassifier.classify()
    │       ├── L1: regex (Vietnamese + English)
    │       └── L2: LLM fallback (rule+llm mode)
    │
    ├── [intent = structured] StructuredReasoningPipeline
    │       ├── KnowledgeService.bootstrap_search()  → chunks
    │       ├── SchemaDiscoveryModule.discover()     → RelationalSchema
    │       ├── StructuredExtractor.extract()        → rows (CLEAR A+B)
    │       ├── SQLReasoningEngine.execute()         → SQL + results
    │       └── AnswerSynthesizer.synthesize()       → answer + citations
    │
    └── [intent = semantic] AgentService semantic loop
            ├── KnowledgeService.bootstrap_search()  (hybrid_kg)
            │       ├── ES hybrid (BM25 + kNN) → chunk hits
            │       └── ES entries + synthesis → structmem hits (RRF fused)
            ├── Agent loop: decide → tool → context assembly
            └── LLMGateway.json_response(task="answer")
            │
            └── [async] ARQ: enqueue chat_memory job → ChatMemoryService.process_turn()

POST /ingest/folder
    │
    ├── Parse (Markdown / PDF via MarkItDown / Excel)
    ├── Chunk (search 512 tok + graph 1536 tok)
    ├── Embed → PostgresStore + ElasticsearchStore (agentrag_segments)
    └── ARQ: enqueue graph_ingest job
            ├── StructMemService.sync_chunks()
            │       └── Per chunk: asyncio.gather(factual_call, relational_call)
            ├── index_structmem_views() → agentrag_entries
            └── [if chunks ≥ threshold] ARQ: enqueue consolidate job → agentrag_synthesis
```

---

## 2. Storage Layer

| Store | Vai trò | Indices / Tables |
|---|---|---|
| **PostgreSQL** | Source of truth: documents, segments, conversations | `documents`, `segments`, `conversations`, `chat_messages` |
| **Elasticsearch** | BM25 + kNN hybrid search + StructMem knowledge | `agentrag_segments`, `agentrag_entries`, `agentrag_synthesis`, `agentrag_chat_entries`, `agentrag_chat_synthesis` |
| **Redis** | Chat history cache (TTL) + ARQ job queue | key-value + sorted sets |

**ES Indices:**

| Index | Nội dung | Dùng cho |
|---|---|---|
| `agentrag_segments` | Chunks gốc từ tài liệu | Hybrid search (BM25 + kNN) |
| `agentrag_entries` | Factual + relational entries (doc StructMem) | Knowledge retrieval |
| `agentrag_synthesis` | Cross-chunk synthesis hypotheses | Multi-hop reasoning |
| `agentrag_chat_entries` | Factual + relational entries từ chat turns | Chat memory retrieval |
| `agentrag_chat_synthesis` | Cross-turn synthesis hypotheses | Long-context conversation |

---

## 3. Yêu cầu hệ thống

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Docker + Docker Compose
- Ít nhất một LLM provider: Ollama (local/container), OpenAI, Gemini, HuggingFace Inference

---

## 4. Cài đặt & Khởi động

```bash
# 1. Copy config
cp .env.example .env
# Chỉnh .env theo provider (xem Section 5)

# 2. Khởi động infra
docker compose up -d
# Tuỳ chọn Ollama với GPU:
docker compose --profile local-llm up -d
docker exec agentrag-ollama ollama pull qwen2.5:14b-instruct
docker exec agentrag-ollama ollama pull nomic-embed-text
docker exec agentrag-ollama ollama pull dengcao/bge-reranker-v2-m3

# 3. Cài dependencies
uv sync

# 4. Migration database
uv run alembic upgrade head

# 5. API server
uv run uvicorn main:app --reload --port 8000

# 6. Background workers (terminal riêng)
arq src.agentrag.worker.settings.WorkerSettings

# 7. Auto-scaler (terminal riêng, thay thế cho bước 6 nếu muốn tự scale)
python scaler.py
```

Kiểm tra:
```bash
curl http://127.0.0.1:8000/config/validate
curl http://127.0.0.1:8000/health/providers
```

### Ingest tài liệu

```bash
curl -X POST http://127.0.0.1:8000/ingest/folder \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "data/docs", "graph_ingest_mode": "async"}'

# Upload single file
curl -X POST http://127.0.0.1:8000/ingest/upload \
  -F "file=@report.pdf"

# Theo dõi tiến độ
curl http://127.0.0.1:8000/documents/<document_id>/graph-status
curl http://127.0.0.1:8000/ingest/queue
```

---

## 5. Cấu hình `.env`

### Tier 1 — CPU Only (RAM ≥ 16 GB)

```env
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EXTRACTION_PROVIDER=ollama
EXTRACTION_MODEL=llama3.2:3b
RETRIEVAL_RERANK_ENABLED=false
STRUCTURED_REASONING_ENABLED=false
STRUCTMEM_MAX_CONCURRENCY=1
```

### Tier 2 — GPU 6–8 GB VRAM

```env
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EXTRACTION_PROVIDER=ollama
EXTRACTION_MODEL=qwen2.5:7b-instruct
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_BACKEND=local_cross_encoder
RETRIEVAL_RERANK_MODEL=dengcao/bge-reranker-v2-m3
STRUCTURED_REASONING_ENABLED=true
LLM_ROUTING_ENABLED=true
LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","decide":"llama3.2:3b","answer":"qwen2.5:7b-instruct"}
STRUCTMEM_MAX_CONCURRENCY=2
```

### Tier 3 — GPU 16–24 GB VRAM

```env
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=mxbai-embed-large
EXTRACTION_PROVIDER=ollama
EXTRACTION_MODEL=qwen2.5:14b-instruct
AGENT_PROVIDER=ollama
AGENT_MODEL=qwen2.5:32b-instruct
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_BACKEND=local_cross_encoder
STRUCTURED_REASONING_ENABLED=true
LLM_ROUTING_ENABLED=true
LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","decide":"llama3.2:3b","schema_discovery":"qwen2.5:14b-instruct","sql_compile":"qwen2.5:14b-instruct","answer":"qwen2.5:32b-instruct"}
STRUCTMEM_MAX_CONCURRENCY=4
```

### Tier 4 — Cloud API (OpenAI / Gemini)

```env
OPENAI_API_KEY=sk-...
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EXTRACTION_PROVIDER=openai
EXTRACTION_MODEL=gpt-4o-mini
AGENT_PROVIDER=openai
AGENT_MODEL=gpt-4o
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_BACKEND=llm_chat
LLM_ROUTING_ENABLED=true
LLM_TASK_MODEL_MAP={"classify":"gpt-4o-mini","decide":"gpt-4o-mini","answer":"gpt-4o"}
```

### 5.1 API Keys

```env
OPENAI_API_KEY=
GEMINI_API_KEY=
HF_TOKEN=
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1/
```

### 5.2 Database & Cache

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=rag
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5433

ELASTICSEARCH_URL=http://localhost:9200
REDIS_URL=redis://127.0.0.1:6379/0
```

### 5.3 StructMem (document)

```env
STRUCTMEM_ENABLED=true
STRUCTMEM_INGEST_MODE=async          # sync | async
STRUCTMEM_CONSOLIDATION_THRESHOLD=20
STRUCTMEM_CONSOLIDATION_HISTORY_TOP_K=15
STRUCTMEM_MAX_CONCURRENCY=3
STRUCTMEM_CHUNK_TIMEOUT_SECONDS=300
STRUCTMEM_CHUNK_RETRIES=3
STRUCTMEM_ENABLE_CACHE=true
STRUCTMEM_CACHE_DIR=.cache/agentrag/extract
```

### 5.4 Chat StructMem

```env
CHAT_STRUCTMEM_ENABLED=false         # true để bật bộ nhớ hội thoại semantic
CHAT_MEMORY_CONSOLIDATION_THRESHOLD=10   # số turns trước khi consolidate
CHAT_MEMORY_TOP_K=8
```

### 5.5 Retrieval & Reranking

```env
RETRIEVAL_TOP_K=10
RETRIEVAL_NUM_CANDIDATES=50
RETRIEVAL_RRF_K=60
RETRIEVAL_RERANK_ENABLED=false
RETRIEVAL_RERANK_BACKEND=llm_chat    # llm_chat | local_cross_encoder
```

### 5.6 LLM Routing

```env
LLM_ROUTING_ENABLED=false
LLM_TASK_MODEL_MAP={}
LLM_COST_TRACKING_ENABLED=false
```

Task keys: `classify`, `decide`, `schema_discovery`, `sql_compile`, `synthesize`, `answer`

### 5.7 Auto-scaler

```env
SCALER_MIN_WORKERS=1
SCALER_MAX_WORKERS=4
SCALER_SCALE_UP_AT=5        # +1 worker mỗi 5 jobs trong queue
SCALER_POLL_SECONDS=5
SCALER_COOLDOWN_SECONDS=30
```

---

## 6. API Reference

### `GET /config/validate`
```bash
curl http://127.0.0.1:8000/config/validate
# {"ok": true, "providers": {"embedding": "...", "extraction": "...", "agent": "..."}}
```

### `GET /health/providers`
```bash
curl http://127.0.0.1:8000/health/providers
```

### `POST /ingest/folder`
```bash
curl -X POST http://127.0.0.1:8000/ingest/folder \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "data/docs", "graph_ingest_mode": "async"}'
```

### `POST /ingest/upload`
```bash
curl -X POST http://127.0.0.1:8000/ingest/upload -F "file=@report.pdf"
```

### `GET /ingest/queue`
```bash
curl http://127.0.0.1:8000/ingest/queue
# {"queue": {"pending_jobs": 3}, "documents": {"done": 12, "processing": 1, ...}}
```

### `POST /search`

| Tham số | Mô tả | Mặc định |
|---|---|---|
| `query` | Câu truy vấn | bắt buộc |
| `mode` | `sparse` / `dense` / `hybrid` / `hybrid_kg` | `hybrid_kg` |
| `top_k` | Số kết quả | `RETRIEVAL_TOP_K` |
| `document_title` | Lọc theo tài liệu | tất cả |
| `rerank` | Bật reranking | `RETRIEVAL_RERANK_ENABLED` |

### `POST /chat`

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "So sánh hiệu suất A và B", "document_title": "report", "conversation_id": "<uuid>"}'
```

**Response:**
```json
{
  "answer": "...",
  "citations": [{"document_title": "...", "section_path": "...", "content_hash": "..."}],
  "reasoning_path": "structured | semantic",
  "sql_query": null,
  "tool_trace": [...],
  "timings_ms": {"total": 820, "decide": 45, "tool": 210, "answer": 560},
  "conversation_id": "<uuid>"
}
```

### `POST /chat/stream`

SSE endpoint. Events: `status`, `token`, `done`, `error`.

```bash
curl -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "conversation_id": "<uuid>"}' \
  --no-buffer
```

### Conversation APIs

```bash
POST   /conversations                       # tạo conversation
GET    /conversations                       # liệt kê (limit=20)
GET    /conversations/{id}/messages         # lịch sử tin nhắn
DELETE /conversations/{id}                  # xóa conversation
```

### Document APIs

```bash
GET    /documents                           # liệt kê tài liệu
GET    /documents/{id}/graph-status         # tiến độ StructMem
DELETE /documents/{id}                      # xóa tài liệu
```

### `GET /metrics`

```bash
curl http://127.0.0.1:8000/metrics
# Token usage + estimated cost (khi LLM_COST_TRACKING_ENABLED=true)
```

---

## 7. CLI

CLI tương tác theo phong cách Claude CLI. Persistent state lưu tại `~/.agentrag/state.json`.

```bash
# Chat với conversation hiện tại
python cli.py chat

# Chat với document cụ thể
python cli.py chat --document "report_2024"

# Tạo conversation mới và chat
python cli.py chat --new --title "Phân tích Q4"

# Quản lý conversations
python cli.py conversations list
python cli.py conversations new --title "Project X"
python cli.py conversations switch <id_prefix>
python cli.py conversations delete <id_prefix>
python cli.py conversations show <id_prefix>
```

**Inline commands trong chat:**

| Command | Mô tả |
|---|---|
| `/new` | Tạo conversation mới |
| `/switch <id>` | Chuyển conversation |
| `/list` | Liệt kê conversations |
| `/clear` | Xóa màn hình |
| `exit` / `quit` | Thoát |

---

## 8. StructMem — Knowledge Extraction

StructMem thay thế Graphiti + Neo4j. Extract knowledge entries trực tiếp vào Elasticsearch.

### So sánh chi phí

| Approach | LLM calls/chunk | Infrastructure | Cost/100 chunks |
|---|---|---|---|
| Graphiti | 4 sequential | Neo4j + ES | ~$1.28 |
| **StructMem** | **2 parallel** | **ES only** | **~$0.97** |

### Dual-Perspective Extraction (per chunk)

```
chunk content
  ├──▶ factual_call()   → {content, subject, fact_type, confidence}
  └──▶ relational_call() → {content, source_entity, target_entity, relation_type}
```

### Cross-Chunk Consolidation

Trigger tự động khi `total_chunks >= STRUCTMEM_CONSOLIDATION_THRESHOLD`:

```
unconsolidated entries
  ├──▶ embed → cosine search → top-K historical seeds
  ├──▶ LLM synthesis → cross-chunk hypotheses
  ├──▶ index vào agentrag_synthesis
  └──▶ mark entries consolidated=true
```

---

## 9. Chat StructMem — Bộ nhớ hội thoại

Khi `CHAT_STRUCTMEM_ENABLED=true`, mỗi chat turn được xử lý qua pipeline tương tự doc StructMem nhưng áp dụng cho lịch sử hội thoại.

### Luồng xử lý

```
User turn → assistant response
  │
  └──▶ [ARQ async] ChatMemoryService.process_turn()
          ├── factual_call()    → facts từ lượt hội thoại
          ├── relational_call() → topic connections, user intent
          ├── embed + index → agentrag_chat_entries
          └── [if count ≥ threshold] consolidate() → agentrag_chat_synthesis

Next question
  └──▶ ChatMemoryService.retrieve(conversation_id, question)
          ├── KNN search trên agentrag_chat_entries
          ├── KNN search trên agentrag_chat_synthesis
          └── inject conversation_memory vào _decide() + _answer() prompts
```

### Khi nào nên bật

- Conversation dài (> 10 turns)
- Cần nhớ thông tin cụ thể từ nhiều turns trước
- Sliding-window history không đủ context

---

## 10. Background Workers & Auto-scaler

Jobs chạy nền qua **ARQ** (Redis-backed task queue) — survive process restart, scalable.

### Chạy worker

```bash
# Single worker
arq src.agentrag.worker.settings.WorkerSettings

# Multiple workers (scale manual)
arq src.agentrag.worker.settings.WorkerSettings &
arq src.agentrag.worker.settings.WorkerSettings &

# Auto-scaler (quản lý workers tự động theo queue depth)
python scaler.py
```

### Job types

| Job | Trigger | Mô tả |
|---|---|---|
| `graph_ingest` | POST /ingest/folder (async mode) | Parse → chunk → extract StructMem → index |
| `consolidate` | Sau graph_ingest khi chunks ≥ threshold | Cross-chunk synthesis |
| `chat_memory` | Sau mỗi chat turn (CHAT_STRUCTMEM_ENABLED) | Extract + index chat memory entries |

### Auto-scaler logic

```
queue depth 0–4  → 1 worker  (SCALER_MIN_WORKERS)
queue depth 5–9  → 2 workers
queue depth 10–14 → 3 workers
queue depth ≥ 15 → 4 workers (SCALER_MAX_WORKERS)
```

Cooldown 30s giữa các lần rescale để tránh thrashing.

---

## 11. Structured SQL Reasoning

Tự động kích hoạt cho câu hỏi so sánh, thống kê, xếp hạng.

### Pipeline 5 bước

```
Classify → Schema discovery → Extract (CLEAR A+B) → SQL compile → Synthesize
```

| `query_type` | Ví dụ |
|---|---|
| `comparison` | "So sánh A và B" |
| `aggregation` | "Tổng doanh thu là bao nhiêu?" |
| `ranking` | "Top 5 sản phẩm bán chạy nhất" |

Fallback về semantic path nếu bất kỳ bước nào thất bại.

---

## 12. LLM Routing

```env
LLM_ROUTING_ENABLED=true
LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","answer":"qwen2.5:32b-instruct"}
```

| Task | Gợi ý model |
|---|---|
| `classify`, `decide` | model nhỏ (3–7B) |
| `schema_discovery`, `sql_compile` | model trung (7–14B) |
| `synthesize`, `answer` | model lớn (32–72B) |

---

## 13. MCP Server

AgentRag expose tools qua **FastMCP** (Model Context Protocol) tại `/mcp`.

```
GET /mcp    → MCP server info
POST /mcp   → MCP tool calls (streamable HTTP transport)
```

**Tools có sẵn:**

| Tool | Mô tả |
|---|---|
| `search` | Hybrid knowledge base search (BM25 + dense + StructMem) |
| `structured_query` | SQL reasoning cho câu hỏi so sánh/tổng hợp |

Dùng với bất kỳ MCP-compatible client (Claude Desktop, Claude Code, custom client).

---

## 14. Security Policy

```python
registry.load_from_list([{
    "document_title": "internal_report",
    "denied_section_prefixes": ["Confidential/", "HR/"],
    "denied_section_patterns": [".*salary.*"],
    "max_results": 5,
}])
```

Áp dụng tại `SecurityService.filter_tool_results()` sau mỗi retrieval step.

---

## 15. Benchmark & Kiểm thử

```bash
python3 scripts/benchmark_ingest.py data/test_docs/SYSTEM_DESIGN.md
python3 scripts/benchmark_ingest.py data/test_docs/SYSTEM_DESIGN.md --embed
python3 scripts/benchmark_retrieval.py data/benchmarks/retrieval_baseline.json --top-k 5
python3 scripts/benchmark_agent.py data/benchmarks/agent_baseline.json --repeat 1
```

**Kiểm tra nhanh:**

```bash
# Semantic chat
curl -X POST http://127.0.0.1:8000/chat \
  -d '{"question": "Tính năng chính?", "document_title": "my_doc"}' \
  -H "Content-Type: application/json"

# Structured chat
curl -X POST http://127.0.0.1:8000/chat \
  -d '{"question": "So sánh module A và B về hiệu suất"}' \
  -H "Content-Type: application/json"

# Hybrid+KG search
curl -X POST http://127.0.0.1:8000/search \
  -d '{"query": "quan hệ phụ thuộc", "mode": "hybrid_kg", "top_k": 5}' \
  -H "Content-Type: application/json"
```

**Kiểm tra response `/chat`:**
- `reasoning_path` → `"structured"` hoặc `"semantic"`
- `tool_trace` → ít nhất 1 retrieval step
- `citations` → có `document_title` + `content_hash`

---

## 16. Reset môi trường

```bash
docker compose down -v --remove-orphans
rm -rf .cache/agentrag

docker compose up -d
uv sync
uv run alembic upgrade head
uv run uvicorn main:app --reload --port 8000
```

**Lỗi thường gặp:**

| Lỗi | Nguyên nhân | Xử lý |
|---|---|---|
| `Connection refused :9200` | Elasticsearch chưa sẵn sàng | Chờ 30s sau `docker compose up` |
| `ARQ pool not initialized` | Chạy app trước khi Redis sẵn sàng | Đảm bảo Redis đang chạy |
| `unsupported value: NaN` | Embedding không ổn định | Đổi sang `nomic-embed-text` |
| Structured path luôn fallback | Model quá nhỏ | Dùng model ≥7B |
| `agentrag_entries` rỗng | `STRUCTMEM_ENABLED=false` hoặc worker chưa chạy | Chạy `arq worker` hoặc `python scaler.py` |

---

## 17. Cấu trúc thư mục

```
AgentRag/
├── main.py                              # FastAPI app + lifespan (ARQ pool)
├── cli.py                               # CLI entry point
├── scaler.py                            # ARQ worker auto-scaler
├── docker-compose.yml                   # PostgreSQL, Elasticsearch, Redis, Ollama
├── pyproject.toml
├── migrations/                          # Alembic
├── data/
│   ├── docs/                            # Tài liệu để ingest
│   └── benchmarks/
└── src/agentrag/
    ├── config.py                        # Pydantic Settings
    ├── config_validation.py
    │
    ├── agent/                           # Semantic agent loop
    │   ├── service.py                   # AgentService — orchestrator + chat memory
    │   ├── context.py                   # ContextAssembler
    │   ├── llm.py                       # AgentLLM (multi-provider)
    │   └── tools.py                     # Tool registry + executor
    │
    ├── agents/                          # Multi-agent workers
    │   ├── data_agent.py
    │   ├── insight_agent.py
    │   └── report_agent.py
    │
    ├── chat/                            # Conversation + Chat StructMem
    │   ├── history.py                   # ConversationStore (Redis + PG)
    │   ├── structmem.py                 # ChatMemoryService (dual-perspective)
    │   └── memory_jobs.py               # ChatMemoryJob dataclass
    │
    ├── cli/                             # CLI (Typer + Rich)
    │   ├── app.py                       # CLI main entry
    │   ├── chat.py                      # Interactive chat loop + SSE parser
    │   ├── conversations.py             # Conversation management commands
    │   └── state.py                     # Persistent active-conversation state
    │
    ├── graph/                           # Doc StructMem extraction
    │   ├── structmem_service.py         # Dual-perspective extraction
    │   ├── structmem_sync.py            # Build + index entry docs
    │   ├── graph_jobs.py                # process_graph_job()
    │   └── consolidation_jobs.py        # process_consolidation_job()
    │
    ├── worker/                          # ARQ background worker
    │   ├── functions.py                 # graph_ingest, consolidate, chat_memory
    │   ├── pool.py                      # ARQ pool singleton (init/get/close)
    │   └── settings.py                  # WorkerSettings cho arq CLI
    │
    ├── mcp/                             # Model Context Protocol server
    │   ├── app.py                       # FastMCP tools (search, structured_query)
    │   └── server.py                    # MCPServer wrapper
    │
    ├── structured/                      # SQL Reasoning Pipeline
    │   ├── pipeline.py
    │   ├── query_classifier.py
    │   ├── schema_discovery.py
    │   ├── extractor.py
    │   ├── sql_engine.py
    │   └── synthesizer.py
    │
    ├── retrieval/
    │   ├── elasticsearch_retriever.py   # Hybrid search (BM25+kNN+StructMem)
    │   └── reranker.py
    │
    ├── ingestion/
    │   ├── pipeline.py                  # ingest_folder() entry point
    │   ├── chunkers/
    │   ├── connectors/
    │   ├── parsers/
    │   ├── embedders/
    │   └── stores/
    │
    ├── services/
    │   ├── llm_gateway.py               # LLM routing + cost tracking
    │   ├── knowledge_service.py
    │   ├── security_service.py
    │   └── context_assembly_service.py
    │
    ├── database/                        # ORM models + AsyncSessionLocal
    ├── common/                          # StageTracer, SecurityPolicy
    └── health/                          # Provider health checks
```

---

## 18. Module READMEs

| Module | README |
|---|---|
| Ingestion Pipeline | [src/agentrag/ingestion/README.md](src/agentrag/ingestion/README.md) |
| StructMem (doc) | [src/agentrag/graph/README.md](src/agentrag/graph/README.md) |
| Chat & StructMem | [src/agentrag/chat/README.md](src/agentrag/chat/README.md) |
| Retrieval | [src/agentrag/retrieval/README.md](src/agentrag/retrieval/README.md) |
| Agent (Semantic Loop) | [src/agentrag/agent/README.md](src/agentrag/agent/README.md) |
| Structured SQL | [src/agentrag/structured/README.md](src/agentrag/structured/README.md) |
| Services | [src/agentrag/services/README.md](src/agentrag/services/README.md) |
| Background Worker | [src/agentrag/worker/README.md](src/agentrag/worker/README.md) |
| CLI | [src/agentrag/cli/README.md](src/agentrag/cli/README.md) |
| MCP Server | [src/agentrag/mcp/README.md](src/agentrag/mcp/README.md) |
| Multi-Agent Workers | [src/agentrag/agents/README.md](src/agentrag/agents/README.md) |
| Common Utilities | [src/agentrag/common/README.md](src/agentrag/common/README.md) |
