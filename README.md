# PAM — Personal AI Memory

PAM là nền tảng RAG (Retrieval-Augmented Generation) với hai luồng suy luận song song và hệ thống bộ nhớ phân cấp:

- **Semantic path** — Hybrid retrieval (BM25 + vector + StructMem knowledge entries) + LLM synthesis
- **Structured path** — SQL reasoning trên bảng dữ liệu trích xuất từ văn bản

**StructMem** (thay thế Graphiti/Neo4j) là lớp trí nhớ phân cấp gồm 2 tầng:
1. **Dual-Perspective Extraction** — 2 LLM call song song/chunk (factual + relational entries), lưu vào Elasticsearch
2. **Cross-Chunk Consolidation** — background worker tổng hợp cross-passage hypotheses định kỳ

---

## Mục lục

1. [Kiến trúc tổng quan](#1-kiến-trúc-tổng-quan)
2. [Storage Layer](#2-storage-layer)
3. [Yêu cầu hệ thống](#3-yêu-cầu-hệ-thống)
4. [Cài đặt & Khởi động](#4-cài-đặt--khởi-động)
5. [Cấu hình `.env`](#5-cấu-hình-env)
6. [API Reference](#6-api-reference)
7. [StructMem — Knowledge Extraction](#7-structmem--knowledge-extraction)
8. [Structured SQL Reasoning](#8-structured-sql-reasoning)
9. [LLM Routing](#9-llm-routing)
10. [Multi-Agent Workers](#10-multi-agent-workers)
11. [Security Policy](#11-security-policy)
12. [Observability & Tracing](#12-observability--tracing)
13. [MCP Server](#13-mcp-server)
14. [Benchmark & Kiểm thử](#14-benchmark--kiểm-thử)
15. [Reset môi trường](#15-reset-môi-trường)
16. [Cấu trúc thư mục](#16-cấu-trúc-thư-mục)
17. [Module READMEs](#17-module-readmes)

---

## 1. Kiến trúc tổng quan

```
POST /chat
    │
    ├── SecurityService.validate_chat_request()
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

POST /ingest/folder
    │
    ├── Parse (Markdown / PDF via Docling / Excel)
    ├── Chunk (search chunks 512 tok + graph chunks 1536 tok)
    ├── Embed → PostgresStore + ElasticsearchStore (pam_segments)
    └── StructMem extraction (sync hoặc background async)
            ├── StructMemService.sync_chunks()
            │       └── Per chunk: asyncio.gather(factual_call, relational_call)
            ├── index_structmem_views() → pam_entries
            └── [if chunks ≥ threshold] ConsolidationJob → pam_synthesis
```

---

## 2. Storage Layer

| Store | Vai trò | Indices / Tables |
|---|---|---|
| **PostgreSQL** | Source of truth: documents, segments, conversations | `documents`, `segments`, `conversations`, `messages` |
| **Elasticsearch** | BM25 + kNN hybrid search | `pam_segments`, `pam_entries`, `pam_synthesis` |
| **Redis** | Chat history cache (TTL-based) | key-value |

> Neo4j đã được loại bỏ. Knowledge graph được thay bằng `pam_entries` và `pam_synthesis` trong Elasticsearch, không cần infrastructure bổ sung.

**ES Indices:**

| Index | Nội dung | Dùng cho |
|---|---|---|
| `pam_segments` | Chunks gốc từ tài liệu | Hybrid search (BM25 + kNN) |
| `pam_entries` | Factual + relational entries trích xuất từ StructMem | Knowledge retrieval (structmem source) |
| `pam_synthesis` | Cross-chunk synthesis hypotheses | Multi-hop reasoning (synthesis source) |

---

## 3. Yêu cầu hệ thống

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (package manager)
- Docker + Docker Compose
- Ít nhất một LLM provider: Ollama (local hoặc container), OpenAI, Gemini, hoặc HuggingFace Inference

### Ollama container với GPU (tuỳ chọn)

Yêu cầu NVIDIA GPU và NVIDIA Container Toolkit:

```bash
# Ubuntu / Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Kiểm tra GPU passthrough hoạt động
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
```

Khi dùng Ollama container, đặt `.env`:
```env
OLLAMA_BASE_URL=http://ollama:11434/v1/
```

> Nếu đã có Ollama chạy native local, **không cần** bật profile `local-llm` — giữ `OLLAMA_BASE_URL=http://127.0.0.1:11434/v1/`.

---

## 4. Cài đặt & Khởi động

```bash
# 1. Sao chép config
cp .env.example .env
# Chỉnh sửa .env theo provider bạn dùng (xem Section 5)

# 2. Khởi động infra (PostgreSQL, Elasticsearch, Redis)
docker compose up -d

# 2b. (Tuỳ chọn) Khởi động Ollama container với GPU passthrough
#     Yêu cầu: NVIDIA GPU + NVIDIA Container Toolkit
docker compose --profile local-llm up -d

# Pull models sau khi Ollama container chạy
docker exec agentrag-ollama ollama pull qwen2.5:14b-instruct
docker exec agentrag-ollama ollama pull nomic-embed-text
docker exec agentrag-ollama ollama pull llama3.2:3b
docker exec agentrag-ollama ollama pull dengcao/bge-reranker-v2-m3

# 3. Cài dependencies
uv sync

# 4. Chạy migration database
uv run alembic upgrade head

# 5. Khởi động server
uv run uvicorn main:app --reload --port 8000

# 6. Kết thúc, gỡ bỏ
docker compose --profile local-llm down -v --remove-orphans
rm -rf .cache
```

Kiểm tra config hợp lệ:
```bash
curl http://127.0.0.1:8000/config/validate
```

Kiểm tra kết nối provider:
```bash
curl http://127.0.0.1:8000/health/providers
```

### Ingest tài liệu

```bash
curl -X POST http://127.0.0.1:8000/ingest/folder \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "data/docs", "graph_ingest_mode": "async"}'
```

`graph_ingest_mode`:
- `"async"` (mặc định) — non-blocking, StructMem extraction chạy background
- `"sync"` — blocking, chờ extraction xong mới trả về

Theo dõi tiến độ:
```bash
curl http://127.0.0.1:8000/documents/<document_id>/graph-status
```

---

## 5. Cấu hình `.env`

### 5.0 Chọn nhanh theo phần cứng

#### Tier 1 — CPU Only (RAM ≥ 16 GB)

```env
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EXTRACTION_PROVIDER=ollama
EXTRACTION_MODEL=llama3.2:3b
RETRIEVAL_TOP_K=8
RETRIEVAL_RERANK_ENABLED=false
STRUCTURED_REASONING_ENABLED=false
GRAPH_MAX_CONCURRENCY=1
GRAPH_CHUNK_TIMEOUT_SECONDS=600
```

#### Tier 2 — GPU 6–8 GB VRAM

```env
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EXTRACTION_PROVIDER=ollama
EXTRACTION_MODEL=qwen2.5:7b-instruct
RETRIEVAL_TOP_K=10
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_BACKEND=local_cross_encoder
RETRIEVAL_RERANK_MODEL=dengcao/bge-reranker-v2-m3
STRUCTURED_REASONING_ENABLED=true
STRUCTURED_CLASSIFIER_METHOD=rule+llm
LLM_ROUTING_ENABLED=true
LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","decide":"llama3.2:3b","answer":"qwen2.5:7b-instruct"}
GRAPH_MAX_CONCURRENCY=2
```

#### Tier 3 — GPU 16–24 GB VRAM

```env
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=mxbai-embed-large
EXTRACTION_PROVIDER=ollama
EXTRACTION_MODEL=qwen2.5:14b-instruct
AGENT_PROVIDER=ollama
AGENT_MODEL=qwen2.5:32b-instruct
RETRIEVAL_TOP_K=15
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_BACKEND=local_cross_encoder
RETRIEVAL_RERANK_MODEL=dengcao/bge-reranker-v2-m3
STRUCTURED_REASONING_ENABLED=true
LLM_ROUTING_ENABLED=true
LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","decide":"llama3.2:3b","schema_discovery":"qwen2.5:14b-instruct","sql_compile":"qwen2.5:14b-instruct","synthesize":"qwen2.5:32b-instruct","answer":"qwen2.5:32b-instruct"}
GRAPH_MAX_CONCURRENCY=4
```

#### Tier 4 — Server / Multi-GPU (≥ 48 GB VRAM)

```env
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct
EXTRACTION_PROVIDER=ollama
EXTRACTION_MODEL=qwen2.5:72b-instruct
AGENT_PROVIDER=ollama
AGENT_MODEL=qwen2.5:72b-instruct
RETRIEVAL_TOP_K=20
RETRIEVAL_NUM_CANDIDATES=100
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_BACKEND=local_cross_encoder
RETRIEVAL_RERANK_MODEL=dengcao/bge-reranker-v2-m3
STRUCTURED_REASONING_ENABLED=true
LLM_ROUTING_ENABLED=true
GRAPH_MAX_CONCURRENCY=8
```

#### Tier 5 — Cloud API (OpenAI / Gemini)

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
RETRIEVAL_RERANK_PROVIDER=openai
RETRIEVAL_RERANK_MODEL=gpt-4o-mini
STRUCTURED_REASONING_ENABLED=true
LLM_ROUTING_ENABLED=true
LLM_TASK_MODEL_MAP={"classify":"gpt-4o-mini","decide":"gpt-4o-mini","answer":"gpt-4o"}
```

#### Tier 6 — Hybrid (HuggingFace + Ollama)

```env
HF_TOKEN=hf_...
EMBEDDING_PROVIDER=hf_inference
EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct
EXTRACTION_PROVIDER=ollama
EXTRACTION_MODEL=qwen2.5:7b-instruct
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_BACKEND=local_cross_encoder
RETRIEVAL_RERANK_MODEL=dengcao/bge-reranker-v2-m3
```

---

### 5.1 API Keys

```env
OPENAI_API_KEY=
GEMINI_API_KEY=
HF_TOKEN=
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1/
OLLAMA_API_KEY=ollama
```

### 5.2 Database & Cache

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=rag
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5433

ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX_NAME=pam_segments

REDIS_URL=redis://127.0.0.1:6379/0
```

### 5.3 StructMem

```env
STRUCTMEM_ENABLED=true
STRUCTMEM_ENTRIES_INDEX_NAME=pam_entries
STRUCTMEM_SYNTHESIS_INDEX_NAME=pam_synthesis
STRUCTMEM_CONSOLIDATION_THRESHOLD=20   # chunks tích luỹ trước khi trigger consolidation
STRUCTMEM_CONSOLIDATION_HISTORY_TOP_K=15

# StructMem dùng chung các settings sau với graph
GRAPH_INGEST_MODE=async                # sync | async
GRAPH_CHUNK_MAX_TOKENS=1536
GRAPH_CHUNK_OVERLAP_TOKENS=128
GRAPH_MAX_CONCURRENCY=3
GRAPH_CHUNK_TIMEOUT_SECONDS=300
GRAPH_CHUNK_RETRIES=3
GRAPH_ENABLE_CACHE=true
GRAPH_CACHE_DIR=.cache/pam/graph
```

### 5.4 Retrieval & Reranking

```env
RETRIEVAL_TOP_K=10
RETRIEVAL_NUM_CANDIDATES=50
RETRIEVAL_RRF_K=60
RETRIEVAL_RERANK_ENABLED=false
RETRIEVAL_RERANK_TOP_N=20
RETRIEVAL_RERANK_BACKEND=llm_chat    # llm_chat | local_cross_encoder
```

### 5.5 LLM Routing

```env
LLM_ROUTING_ENABLED=false
LLM_TASK_MODEL_MAP={}
LLM_COST_TRACKING_ENABLED=false
```

Task keys: `classify`, `decide`, `schema_discovery`, `sql_compile`, `synthesize`, `answer`, `insight`, `report`

---

## 6. API Reference

### `GET /config/validate`
```bash
curl http://127.0.0.1:8000/config/validate
# {"ok": true, "providers": {...}}
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

### `POST /search`

| Tham số | Mô tả | Mặc định |
|---|---|---|
| `query` | Câu truy vấn | bắt buộc |
| `mode` | `sparse` / `dense` / `hybrid` / `hybrid_kg` | `hybrid_kg` |
| `top_k` | Số kết quả | `RETRIEVAL_TOP_K` |
| `document_title` | Lọc theo tài liệu | tất cả |
| `rerank` | Bật reranking | `RETRIEVAL_RERANK_ENABLED` |

`hybrid_kg` mode: fuse chunk hits + structmem/synthesis hits qua RRF.

### `POST /chat`

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "So sánh hiệu suất module A và B", "document_title": "report"}'
```

**Response:**
```json
{
  "answer": "...",
  "citations": [{"document_title": "...", "section_path": "...", "content_hash": "..."}],
  "reasoning_path": "structured | semantic",
  "tool_trace": [...],
  "timings_ms": {"classify": 12, "retrieve": 45, "synthesize": 210},
  "conversation_id": "<uuid>"
}
```

### `GET /documents/{id}/graph-status`
```bash
curl http://127.0.0.1:8000/documents/<id>/graph-status
# {"graph_status": "done", "graph_progress": 1.0, "graph_processed_chunks": 24}
```

### Conversation APIs

```bash
POST /conversations                          # tạo conversation
GET  /conversations                          # liệt kê
GET  /conversations/{id}/messages            # lịch sử tin nhắn
```

---

## 7. StructMem — Knowledge Extraction

StructMem thay thế Graphiti + Neo4j bằng cách extract knowledge entries trực tiếp vào Elasticsearch.

### So sánh chi phí

| Approach | LLM calls/chunk | Infrastructure | Cost/100 chunks |
|---|---|---|---|
| Graphiti | 4 sequential | Neo4j + ES | ~$1.28 |
| **StructMem** | **2 parallel** | **ES only** | **~$0.97** |

Tiết kiệm ~37% base, tăng lên ~60%+ khi corpus lớn (Graphiti's dedup context tăng tuyến tính).

### Dual-Perspective Extraction

Mỗi chunk chạy 2 LLM call song song:

**Factual entries** — sự kiện cụ thể, tự chứa:
```json
{"content": "PAM sử dụng RRF với k=60 làm hệ số fusion mặc định",
 "subject": "PAM", "fact_type": "property", "confidence": "high"}
```

**Relational entries** — quan hệ nhân quả giữa các concept:
```json
{"content": "StructMem enables cost reduction by replacing 4 sequential Graphiti calls",
 "source_entity": "StructMem", "target_entity": "Graphiti",
 "relation_type": "contrasts", "confidence": "high"}
```

### Cross-Chunk Consolidation

Tự động trigger khi `total_chunks >= STRUCTMEM_CONSOLIDATION_THRESHOLD`:

1. Lấy unconsolidated entries trong group
2. Embed → cosine search → top-K historical seeds
3. Reconstruct context từ các chunk_position của seeds
4. LLM synthesis → cross-chunk hypotheses
5. Index vào `pam_synthesis`
6. Mark entries là `consolidated=true`

---

## 8. Structured SQL Reasoning

Tự động kích hoạt cho câu hỏi so sánh, thống kê, xếp hạng.

### Các dạng câu hỏi

| `query_type` | Ví dụ |
|---|---|
| `comparison` | "So sánh A và B" |
| `aggregation` | "Tổng doanh thu là bao nhiêu?" |
| `ranking` | "Top 5 sản phẩm bán chạy nhất" |
| `multi_filter` | "Tất cả sản phẩm loại A và giá > 100" |

### Pipeline 5 bước

```
Retrieve → Schema discovery → Extract (CLEAR A+B) → SQL compile → Synthesize
```

**CLEAR Validation:**
- Level A: drop null PK, coerce numeric strings
- Level B: dedup cùng PK, resolve conflict theo document position

Fallback về semantic path nếu bất kỳ bước nào thất bại.

---

## 9. LLM Routing

```env
LLM_ROUTING_ENABLED=true
LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","answer":"qwen2.5:32b-instruct"}
```

| Task | Gợi ý model |
|---|---|
| `classify`, `decide` | model nhỏ (3-7B) |
| `schema_discovery`, `sql_compile` | model trung (7-14B) |
| `synthesize`, `answer`, `report` | model lớn (32-72B) |

---

## 10. Multi-Agent Workers

```python
data_agent    = DataAgent(knowledge_service, llm_gateway)
insight_agent = InsightAgent(llm_gateway)
report_agent  = ReportAgent(llm_gateway)

data_results    = await asyncio.gather(*[data_agent.run(task) for task in sub_questions])
insight_results = await asyncio.gather(*[insight_agent.run(dr) for dr in data_results])
report          = await report_agent.run(insight_results)
```

---

## 11. Security Policy

```python
registry.load_from_list([{
    "document_title": "internal_report",
    "denied_section_prefixes": ["Confidential/", "HR/"],
    "denied_section_patterns": [".*salary.*"],
    "max_results": 5,
}])
```

Áp dụng tại `SecurityService.filter_tool_results()` sau mỗi retrieval.

---

## 12. Observability & Tracing

```json
{
  "timings_ms": {
    "classify": 12.3, "retrieve": 45.7,
    "schema": 183.2,  "sql": 24.8, "synthesize": 218.5
  }
}
```

```env
OBSERVABILITY_TRACE_ENABLED=true
```

---

## 13. MCP Server

```python
server = PAMMCPServer()
result = await server.handle_tool_call("search", {"query": "...", "top_k": 5})
result = await server.handle_tool_call("structured_query", {"question": "...", "query_type": "comparison"})
```

---

## 14. Benchmark & Kiểm thử

```bash
python3 scripts/benchmark_ingest.py data/test_docs/SYSTEM_DESIGN.md
python3 scripts/benchmark_ingest.py data/test_docs/SYSTEM_DESIGN.md --embed
python3 scripts/benchmark_retrieval.py data/benchmarks/retrieval_baseline.json --top-k 5
python3 scripts/benchmark_agent.py data/benchmarks/agent_baseline.json --repeat 1
```

### Kiểm tra nhanh

```bash
# Semantic chat
curl -X POST http://127.0.0.1:8000/chat \
  -d '{"question": "Tính năng chính?", "document_title": "my_doc"}' \
  -H "Content-Type: application/json"

# Structured chat
curl -X POST http://127.0.0.1:8000/chat \
  -d '{"question": "So sánh module A và B về hiệu suất", "document_title": "my_doc"}' \
  -H "Content-Type: application/json"

# Hybrid+KG search
curl -X POST http://127.0.0.1:8000/search \
  -d '{"query": "quan hệ phụ thuộc giữa các module", "mode": "hybrid_kg", "top_k": 5}' \
  -H "Content-Type: application/json"
```

**Kiểm tra response `/chat`:**
- `reasoning_path` → `"structured"` hoặc `"semantic"`
- `tool_trace` → ít nhất 1 retrieval step
- `citations` → có `document_title` + `content_hash`
- Hit sources trong `tool_trace` → `"structmem"` và/hoặc `"synthesis"` (không còn `"graph"`)

---

## 15. Reset môi trường

```bash
docker compose down -v --remove-orphans
rm -rf .cache/pam/graph

docker compose up -d
uv sync
uv run alembic upgrade head
uv run uvicorn main:app --reload --port 8000
```

**Lỗi thường gặp:**

| Lỗi | Nguyên nhân | Xử lý |
|---|---|---|
| `Connection refused :9200` | Elasticsearch chưa sẵn sàng | Chờ 30s sau `docker compose up` |
| `unsupported value: NaN` | Ollama embedding không ổn định | Đổi sang `nomic-embed-text` |
| Structured path luôn fallback | Model quá nhỏ | Dùng model ≥7B, `EXTRACTION_TEMPERATURE=0.0` |
| `pam_entries` rỗng sau ingest | `STRUCTMEM_ENABLED=false` hoặc extraction lỗi | Kiểm tra logs worker, bật `GRAPH_ENABLE_CACHE=false` để debug |

---

## 16. Cấu trúc thư mục

```
AgentRag/
├── main.py                              # FastAPI app + lifespan (graph + consolidation workers)
├── docker-compose.yml                   # PostgreSQL, Elasticsearch, Redis
├── migrations/                          # Alembic
├── data/
│   ├── docs/                            # Tài liệu để ingest
│   └── benchmarks/                      # Baseline JSON cho benchmark
├── scripts/                             # Benchmark scripts
└── src/pam/
    ├── config.py                        # Pydantic Settings
    ├── config_validation.py
    │
    ├── agent/                           # Semantic agent loop
    │   ├── service.py                   # AgentService — orchestrator
    │   ├── context.py                   # ContextAssembler (4-stage)
    │   ├── llm.py                       # AgentLLM (multi-provider)
    │   └── tools.py                     # Tool registry + executor
    │
    ├── agents/                          # Multi-agent workers
    │   ├── data_agent.py
    │   ├── insight_agent.py
    │   └── report_agent.py
    │
    ├── structured/                      # SQL Reasoning Pipeline
    │   ├── pipeline.py
    │   ├── query_classifier.py          # L1 regex + L2 LLM
    │   ├── schema_discovery.py
    │   ├── extractor.py                 # CLEAR A+B validation
    │   ├── sql_engine.py                # SQLite in-memory
    │   └── synthesizer.py
    │
    ├── graph/                           # StructMem knowledge extraction
    │   ├── structmem_service.py         # Dual-perspective extraction (2 parallel LLM calls)
    │   ├── structmem_sync.py            # Build + index entry docs
    │   ├── graph_jobs.py                # Async ingest queue worker
    │   └── consolidation_jobs.py        # Cross-chunk consolidation worker
    │
    ├── retrieval/
    │   ├── elasticsearch_retriever.py   # Hybrid retrieval (BM25+kNN+StructMem)
    │   └── reranker.py                  # LLM / CrossEncoder reranker
    │
    ├── ingestion/
    │   ├── pipeline.py                  # ingest_folder() entry point
    │   ├── chunkers/hybrid_chunker.py
    │   ├── connectors/                  # FolderConnector, MarkdownConnector
    │   ├── parsers/                     # DoclingParser, ExcelParser, ImageDescriber
    │   ├── embedders/                   # OpenAI / Gemini / HF / Ollama embedders
    │   └── stores/
    │       ├── elasticsearch_store.py   # pam_segments + pam_entries + pam_synthesis
    │       └── postgres_store.py
    │
    ├── services/
    │   ├── knowledge_service.py         # Retrieval facade
    │   ├── llm_gateway.py               # LLM routing + cost tracking
    │   ├── security_service.py          # Query-time policy gate
    │   └── context_assembly_service.py
    │
    ├── chat/history.py                  # ConversationStore (Redis + PG)
    ├── database/                        # ORM models + AsyncSessionLocal
    ├── common/                          # StageTracer, SecurityPolicy
    ├── mcp/server.py                    # MCP server
    └── health/providers.py              # Provider health checks
```

---

## 17. Module READMEs

Chi tiết từng module xem tại:

| Module | README |
|---|---|
| Ingestion Pipeline | [src/pam/ingestion/README.md](src/pam/ingestion/README.md) |
| StructMem (Graph) | [src/pam/graph/README.md](src/pam/graph/README.md) |
| Retrieval | [src/pam/retrieval/README.md](src/pam/retrieval/README.md) |
| Agent (Semantic Loop) | [src/pam/agent/README.md](src/pam/agent/README.md) |
| Structured SQL Reasoning | [src/pam/structured/README.md](src/pam/structured/README.md) |
| Services | [src/pam/services/README.md](src/pam/services/README.md) |
| Multi-Agent Workers | [src/pam/agents/README.md](src/pam/agents/README.md) |
| Chat & Conversation | [src/pam/chat/README.md](src/pam/chat/README.md) |
| Common Utilities | [src/pam/common/README.md](src/pam/common/README.md) |
