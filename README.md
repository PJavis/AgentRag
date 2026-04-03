# PAM — Personal AI Memory

PAM là nền tảng RAG (Retrieval-Augmented Generation) với hai luồng suy luận song song:

- **Semantic path** — Hybrid retrieval (BM25 + vector + knowledge graph) + LLM synthesis
- **Structured path** — SQL reasoning trên bảng dữ liệu trích xuất từ văn bản (DocSage-inspired)

---

## Mục lục

1. [Kiến trúc tổng quan](#1-kiến-trúc-tổng-quan)
2. [Yêu cầu hệ thống](#2-yêu-cầu-hệ-thống)
3. [Cài đặt & Khởi động](#3-cài-đặt--khởi-động)
4. [Cấu hình (`.env`)](#4-cấu-hình-env)
5. [API Reference](#5-api-reference)
6. [Structured SQL Reasoning](#6-structured-sql-reasoning)
7. [LLM Routing](#7-llm-routing)
8. [Security Policy](#8-security-policy)
9. [Observability & Tracing](#9-observability--tracing)
10. [Multi-Agent Workers](#10-multi-agent-workers)
11. [MCP Server](#11-mcp-server)
12. [Benchmark & Kiểm thử](#12-benchmark--kiểm-thử)
13. [Reset môi trường](#13-reset-môi-trường)
14. [Cấu trúc thư mục](#14-cấu-trúc-thư-mục)

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
            ├── Agent loop: decide → tool → context assembly
            └── LLMGateway.json_response(task="answer")
```

**Storage:**
| Store | Vai trò |
|---|---|
| PostgreSQL | Source of truth: documents, segments, conversations |
| Elasticsearch | Search index: BM25 + kNN (segments, entities, relationships) |
| Neo4j | Knowledge graph: entities, relationships, episodes (Graphiti) |
| Redis | Chat history cache (TTL-based) |

---

## 2. Yêu cầu hệ thống

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (package manager)
- Docker + Docker Compose
- Ít nhất một LLM provider: Ollama (local), OpenAI, Gemini, hoặc HuggingFace Inference

---

## 3. Cài đặt & Khởi động

```bash
# 1. Sao chép config
cp .env.example .env
# Chỉnh sửa .env theo provider bạn dùng (xem Section 4)

# 2. Khởi động infra (PostgreSQL, Elasticsearch, Redis, Neo4j)
docker compose up -d

# 3. Cài dependencies
uv sync

# 4. Chạy migration database
uv run alembic upgrade head

# 5. Khởi động server
uv run uvicorn main:app --reload --port 8000
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
# Ingest toàn bộ thư mục Markdown
curl -X POST http://127.0.0.1:8000/ingest/folder \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "data/docs", "graph_ingest_mode": "async"}'

# graph_ingest_mode:
#   "async" (mặc định) — non-blocking, graph được xây dựng background
#   "sync"             — blocking, chờ graph xong mới trả về
```

Theo dõi tiến độ graph ingest:
```bash
curl http://127.0.0.1:8000/documents/<document_id>/graph-status
```

---

## 4. Cấu hình (`.env`)

### 4.0 Chọn nhanh theo phần cứng

Sao chép block `.env` tương ứng, điền API key (nếu dùng cloud), rồi chạy.

#### Tier 1 — CPU Only (RAM ≥ 16 GB, không có GPU)

Dùng khi chỉ có máy tính thông thường. Tốc độ chậm, phù hợp để thử nghiệm.

```env
# Provider
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text          # ~274MB, chạy tốt trên CPU
EXTRACTION_PROVIDER=ollama
EXTRACTION_MODEL=llama3.2:3b              # ~2GB RAM, nhanh trên CPU
AGENT_PROVIDER=
AGENT_MODEL=
GRAPH_EMBEDDING_PROVIDER=ollama
GRAPH_EMBEDDING_MODEL=nomic-embed-text

# Retrieval — tắt rerank để tiết kiệm CPU
RETRIEVAL_TOP_K=8
RETRIEVAL_RERANK_ENABLED=false

# Structured reasoning — tắt (model 3B không đủ JSON output)
STRUCTURED_REASONING_ENABLED=false

# Graph — giảm concurrency
GRAPH_MAX_CONCURRENCY=1
GRAPH_CHUNK_TIMEOUT_SECONDS=600
```

> **Lưu ý:** Model 3B thường không đủ khả năng sinh JSON schema chính xác. Tắt `STRUCTURED_REASONING_ENABLED` để tránh lỗi.

---

#### Tier 2 — GPU 6–8 GB VRAM (GTX 1660 / RTX 3060 6GB / RTX 4060 8GB)

Cân bằng giữa chất lượng và tốc độ. Đủ cho production nhỏ.

```env
# Provider
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text          # ~274MB VRAM
EXTRACTION_PROVIDER=ollama
EXTRACTION_MODEL=qwen2.5:7b-instruct      # ~4.7GB VRAM (Q4_K_M)
AGENT_PROVIDER=
AGENT_MODEL=
GRAPH_EMBEDDING_PROVIDER=ollama
GRAPH_EMBEDDING_MODEL=nomic-embed-text

# Retrieval
RETRIEVAL_TOP_K=10
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_BACKEND=local_cross_encoder
RETRIEVAL_RERANK_MODEL=dengcao/bge-reranker-v2-m3  # ~580MB VRAM

# Structured reasoning
STRUCTURED_REASONING_ENABLED=true
STRUCTURED_CLASSIFIER_METHOD=rule+llm
STRUCTURED_MAX_CHUNKS_FOR_SCHEMA=10
STRUCTURED_MAX_CHUNKS_FOR_EXTRACT=20
STRUCTURED_SQL_MAX_RETRIES=2

# LLM Routing — dùng model nhỏ cho task nhẹ
LLM_ROUTING_ENABLED=true
LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","decide":"llama3.2:3b","schema_discovery":"qwen2.5:7b-instruct","sql_compile":"qwen2.5:7b-instruct","synthesize":"qwen2.5:7b-instruct","answer":"qwen2.5:7b-instruct"}

# Graph
GRAPH_MAX_CONCURRENCY=2
GRAPH_CHUNK_TIMEOUT_SECONDS=300
```

---

#### Tier 3 — GPU 16–24 GB VRAM (RTX 3090 / RTX 4090 / RTX 4080)

Chất lượng cao, đủ cho production thực tế.

```env
# Provider
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=mxbai-embed-large         # ~670MB VRAM, tốt hơn nomic cho tiếng Anh
EXTRACTION_PROVIDER=ollama
EXTRACTION_MODEL=qwen2.5:14b-instruct     # ~9GB VRAM (Q4_K_M)
AGENT_PROVIDER=ollama
AGENT_MODEL=qwen2.5:32b-instruct          # ~18GB VRAM (Q4_K_M) — dùng cho answer
AGENT_TEMPERATURE=0.1
GRAPH_EMBEDDING_PROVIDER=ollama
GRAPH_EMBEDDING_MODEL=mxbai-embed-large

# Retrieval
RETRIEVAL_TOP_K=15
RETRIEVAL_NUM_CANDIDATES=60
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_BACKEND=local_cross_encoder
RETRIEVAL_RERANK_MODEL=dengcao/bge-reranker-v2-m3
RETRIEVAL_RERANK_TOP_N=30

# Structured reasoning
STRUCTURED_REASONING_ENABLED=true
STRUCTURED_CLASSIFIER_METHOD=rule+llm
STRUCTURED_MAX_CHUNKS_FOR_SCHEMA=15
STRUCTURED_MAX_CHUNKS_FOR_EXTRACT=30
STRUCTURED_SQL_MAX_RETRIES=3

# LLM Routing — tách model nhỏ/lớn theo task
LLM_ROUTING_ENABLED=true
LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","decide":"llama3.2:3b","schema_discovery":"qwen2.5:14b-instruct","sql_compile":"qwen2.5:14b-instruct","synthesize":"qwen2.5:32b-instruct","answer":"qwen2.5:32b-instruct","insight":"qwen2.5:14b-instruct","report":"qwen2.5:32b-instruct"}
LLM_COST_TRACKING_ENABLED=true

# Graph
GRAPH_MAX_CONCURRENCY=4
GRAPH_CHUNK_MAX_TOKENS=1536
```

---

#### Tier 4 — Server / Multi-GPU (VRAM ≥ 48 GB, hoặc A100/H100)

Chất lượng tối đa, production quy mô lớn.

```env
# Provider
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct  # tốt nhất cho đa ngôn ngữ
EXTRACTION_PROVIDER=ollama
EXTRACTION_MODEL=qwen2.5:72b-instruct     # ~40GB VRAM (Q4_K_M)
AGENT_PROVIDER=ollama
AGENT_MODEL=qwen2.5:72b-instruct
GRAPH_EMBEDDING_PROVIDER=ollama
GRAPH_EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct

# Retrieval
RETRIEVAL_TOP_K=20
RETRIEVAL_NUM_CANDIDATES=100
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_BACKEND=local_cross_encoder
RETRIEVAL_RERANK_MODEL=dengcao/bge-reranker-v2-m3
RETRIEVAL_RERANK_TOP_N=40

# Structured reasoning
STRUCTURED_REASONING_ENABLED=true
STRUCTURED_CLASSIFIER_METHOD=rule+llm
STRUCTURED_MAX_CHUNKS_FOR_SCHEMA=20
STRUCTURED_MAX_CHUNKS_FOR_EXTRACT=40
STRUCTURED_SQL_MAX_RETRIES=3

# LLM Routing
LLM_ROUTING_ENABLED=true
LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","decide":"llama3.2:3b","schema_discovery":"qwen2.5:14b-instruct","sql_compile":"qwen2.5:14b-instruct","synthesize":"qwen2.5:72b-instruct","answer":"qwen2.5:72b-instruct","insight":"qwen2.5:72b-instruct","report":"qwen2.5:72b-instruct"}
LLM_COST_TRACKING_ENABLED=true

# Graph
GRAPH_MAX_CONCURRENCY=8
GRAPH_CHUNK_MAX_TOKENS=2048
GRAPH_CHUNK_RETRIES=3
```

---

#### Tier 5 — Cloud API (OpenAI / Gemini)

Không cần GPU, trả phí theo token. Tốt nhất cho prototype nhanh.

```env
# API Keys
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...               # tùy chọn, nếu dùng Gemini

# Provider — OpenAI
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small    # rẻ hơn, đủ tốt
EXTRACTION_PROVIDER=openai
EXTRACTION_MODEL=gpt-4o-mini              # rẻ, nhanh, đủ cho extraction
AGENT_PROVIDER=openai
AGENT_MODEL=gpt-4o                        # tốt hơn cho reasoning
GRAPH_EMBEDDING_PROVIDER=openai
GRAPH_EMBEDDING_MODEL=text-embedding-3-small

# Retrieval
RETRIEVAL_TOP_K=10
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_BACKEND=llm_chat
RETRIEVAL_RERANK_PROVIDER=openai
RETRIEVAL_RERANK_MODEL=gpt-4o-mini

# Structured reasoning
STRUCTURED_REASONING_ENABLED=true
STRUCTURED_CLASSIFIER_METHOD=rule+llm

# LLM Routing — tiết kiệm chi phí
LLM_ROUTING_ENABLED=true
LLM_TASK_MODEL_MAP={"classify":"gpt-4o-mini","decide":"gpt-4o-mini","schema_discovery":"gpt-4o-mini","sql_compile":"gpt-4o-mini","synthesize":"gpt-4o","answer":"gpt-4o","insight":"gpt-4o-mini","report":"gpt-4o"}
LLM_COST_TRACKING_ENABLED=true
```

---

#### Tier 6 — Hybrid (HuggingFace Inference + Ollama local)

Embed qua HF API (không cần GPU cho embedding), LLM chạy local.

```env
# API Keys
HF_TOKEN=hf_...

# Provider
EMBEDDING_PROVIDER=hf_inference
EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct  # tốt nhất cho tiếng Việt
EXTRACTION_PROVIDER=ollama
EXTRACTION_MODEL=qwen2.5:7b-instruct
AGENT_PROVIDER=
AGENT_MODEL=
GRAPH_EMBEDDING_PROVIDER=hf_inference
GRAPH_EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct

# Retrieval
RETRIEVAL_TOP_K=10
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_BACKEND=local_cross_encoder
RETRIEVAL_RERANK_MODEL=dengcao/bge-reranker-v2-m3

# Structured reasoning
STRUCTURED_REASONING_ENABLED=true
STRUCTURED_CLASSIFIER_METHOD=rule+llm
```

---

### Bảng so sánh model

#### Embedding models

| Model | VRAM | Dim | Đa ngôn ngữ | Ghi chú |
|---|---|---|---|---|
| `nomic-embed-text` | ~274 MB | 768 | Hạn chế | Nhẹ nhất, tốt cho tiếng Anh |
| `mxbai-embed-large` | ~670 MB | 1024 | Tốt | Tốt cho tiếng Anh, nhanh |
| `intfloat/multilingual-e5-large-instruct` | ~1.2 GB | 1024 | Rất tốt | **Khuyến nghị** cho tiếng Việt |
| `text-embedding-3-small` | cloud | 1536 | Tốt | OpenAI, rẻ |
| `text-embedding-3-large` | cloud | 3072 | Rất tốt | OpenAI, tốt nhất |

#### LLM models (Ollama)

| Model | VRAM (Q4) | JSON output | Cấu trúc hóa | Ghi chú |
|---|---|---|---|---|
| `llama3.2:3b` | ~2 GB | Trung bình | Kém | Chỉ dùng cho classify/decide |
| `phi3.5:3.8b` | ~2.3 GB | Tốt | Trung bình | Thay thế llama3.2 trên CPU |
| `qwen2.5:7b-instruct` | ~4.7 GB | Rất tốt | Tốt | **Tier 2 minimum** |
| `llama3.1:8b-instruct` | ~5.5 GB | Tốt | Tốt | Thay thế qwen2.5:7b |
| `qwen2.5:14b-instruct` | ~9 GB | Xuất sắc | Rất tốt | **Khuyến nghị** cho structured |
| `qwen2.5:32b-instruct` | ~18 GB | Xuất sắc | Xuất sắc | Tốt nhất trong 24GB VRAM |
| `llama3.1:70b-instruct` | ~40 GB | Rất tốt | Tốt | Cần multi-GPU |
| `qwen2.5:72b-instruct` | ~42 GB | Xuất sắc | Xuất sắc | Tốt nhất overall |

> **Lưu ý Structured Reasoning:** Cần model với JSON output tốt. Tối thiểu `qwen2.5:7b-instruct`. Dưới 7B dễ lỗi schema → fallback semantic.

#### Reranker models

| Model | VRAM | Đa ngôn ngữ | Ghi chú |
|---|---|---|---|
| `dengcao/bge-reranker-v2-m3` | ~580 MB | Rất tốt | **Khuyến nghị** local |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~120 MB | Kém | Nhẹ, chỉ tiếng Anh |
| LLM rerank (llm_chat) | dùng LLM | Tốt | Chậm hơn, không cần model riêng |

---

### 4.1 API Keys

```env
# Chỉ cần điền provider nào bạn dùng
OPENAI_API_KEY=
GEMINI_API_KEY=
HF_TOKEN=
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1/
OLLAMA_API_KEY=ollama
```

---

### 4.2 Database & Cache

```env
# PostgreSQL — source of truth
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=rag
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5433

# Elasticsearch — BM25 + vector search
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX_NAME=pam_segments
ELASTICSEARCH_ENTITY_INDEX_NAME=pam_entities
ELASTICSEARCH_RELATIONSHIP_INDEX_NAME=pam_relationships

# Redis — chat history cache
REDIS_URL=redis://127.0.0.1:6379/0

# Neo4j — knowledge graph
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j123456
```

---

### 4.3 LLM Providers

Provider được cấu hình riêng cho từng chức năng. Nếu `AGENT_*` để trống, fallback về `EXTRACTION_*`.

```env
# Embedding (dùng cho search chunks)
EMBEDDING_PROVIDER=ollama           # openai | gemini | hf_inference | ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BASE_URL=                 # để trống = URL mặc định của provider
EMBEDDING_BATCH_SIZE=32

# Extraction & default LLM (dùng cho schema, SQL, synthesis, agent loop)
EXTRACTION_PROVIDER=ollama
EXTRACTION_MODEL=qwen2.5:7b-instruct
EXTRACTION_BASE_URL=
EXTRACTION_TEMPERATURE=0.0

# Agent runtime — override EXTRACTION nếu muốn model mạnh hơn cho answer
AGENT_PROVIDER=
AGENT_MODEL=
AGENT_BASE_URL=
AGENT_TEMPERATURE=

# Graph embedding (dùng cho entity/relationship vectors trong Neo4j)
GRAPH_EMBEDDING_PROVIDER=ollama
GRAPH_EMBEDDING_MODEL=nomic-embed-text
GRAPH_EMBEDDING_BASE_URL=
```

---

### 4.4 Chunking

```env
# Search chunks — nhỏ, overlap nhiều → recall tốt
SEARCH_CHUNK_MAX_TOKENS=512
SEARCH_CHUNK_OVERLAP_TOKENS=64
SEARCH_CHUNK_BY_PARAGRAPH=true      # tách ưu tiên theo \n\n

# Graph chunks — lớn hơn → ít episode, ít LLM call
GRAPH_CHUNK_MAX_TOKENS=1200
GRAPH_CHUNK_OVERLAP_TOKENS=32

CHUNK_TOKENIZER_MODEL=text-embedding-3-large  # tokenizer đếm token (offline)
```

---

### 4.5 Retrieval & Reranking

```env
RETRIEVAL_TOP_K=10                  # kết quả trả về cuối cùng
RETRIEVAL_NUM_CANDIDATES=50         # ứng viên trước khi fusion + rerank
RETRIEVAL_RRF_K=60                  # hệ số RRF (60 là chuẩn học thuật)

# Reranking
RETRIEVAL_RERANK_ENABLED=false
RETRIEVAL_RERANK_TOP_N=20           # số ứng viên đưa vào reranker
RETRIEVAL_RERANK_BACKEND=llm_chat   # llm_chat | local_cross_encoder
RETRIEVAL_RERANK_PROVIDER=          # fallback: AGENT_* → EXTRACTION_*
RETRIEVAL_RERANK_MODEL=
RETRIEVAL_RERANK_BASE_URL=
RETRIEVAL_RERANK_TEMPERATURE=0.0
```

---

### 4.6 Agent & Chat

```env
AGENT_MAX_STEPS=4                   # số vòng tool-use tối đa
AGENT_TOOL_TOP_K=5                  # top-k mỗi lần gọi tool
AGENT_MAX_CONTEXT_CHUNKS=6          # số chunk đưa vào LLM context
CHAT_HISTORY_WINDOW=10              # số message lịch sử giữ trong prompt
CHAT_REDIS_TTL_SECONDS=300          # TTL cache Redis (giây)
```

---

### 4.7 Structured SQL Reasoning

```env
STRUCTURED_REASONING_ENABLED=true       # bật/tắt toàn bộ SQL path
STRUCTURED_CLASSIFIER_METHOD=rule+llm   # rule | llm | rule+llm
STRUCTURED_MAX_CHUNKS_FOR_SCHEMA=10     # chunk đưa vào schema discovery
STRUCTURED_MAX_CHUNKS_FOR_EXTRACT=20    # chunk đưa vào extraction
STRUCTURED_SQL_MAX_RETRIES=2            # số lần retry khi SQL lỗi
STRUCTURED_CONFIDENCE_THRESHOLD=0.7    # ngưỡng tin cậy để chọn structured
```

---

### 4.8 LLM Routing

```env
LLM_ROUTING_ENABLED=false
# JSON: task_name → model_name (dùng model của EXTRACTION_PROVIDER)
LLM_TASK_MODEL_MAP={}
# Ví dụ với Ollama Tier 2:
# LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","decide":"llama3.2:3b","answer":"qwen2.5:7b-instruct"}
LLM_COST_TRACKING_ENABLED=false
```

**Task keys có thể route:** `classify`, `decide`, `schema_discovery`, `sql_compile`, `synthesize`, `answer`, `insight`, `report`

---

### 4.9 Observability

```env
OBSERVABILITY_TRACE_ENABLED=true    # ghi timings_ms trong response /chat
```

---

### 4.10 Graph Ingest

```env
GRAPH_INGEST_MODE=async             # sync | async (async không block /ingest)
ENABLE_DOCLING_PARSE=false          # bật để parse PDF/DOCX (cần pip install docling)
GRAPH_MAX_CONCURRENCY=3             # số chunk Graphiti chạy song song
GRAPH_CHUNK_TIMEOUT_SECONDS=180     # timeout mỗi chunk (giây)
GRAPH_CHUNK_RETRIES=2               # retry khi chunk lỗi
GRAPH_ENABLE_CACHE=true             # cache kết quả để tránh re-extract
GRAPH_CACHE_DIR=.cache/pam/graph
```

---

## 5. API Reference

### `GET /config/validate`

Kiểm tra cấu hình `.env` hợp lệ.

```bash
curl http://127.0.0.1:8000/config/validate
# {"ok": true, "providers": {"embedding": "ollama", ...}}
```

### `GET /health/providers`

Kiểm tra kết nối thực tế đến tất cả provider.

```bash
curl http://127.0.0.1:8000/health/providers
```

### `POST /ingest/folder`

Ingest tất cả file Markdown trong thư mục.

```bash
curl -X POST http://127.0.0.1:8000/ingest/folder \
  -H "Content-Type: application/json" \
  -d '{
    "folder_path": "data/docs",
    "graph_ingest_mode": "async"
  }'
```

### `POST /search`

Tìm kiếm trực tiếp (không qua agent loop).

```bash
# Hybrid + Knowledge Graph
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "kiến trúc hệ thống",
    "mode": "hybrid_kg",
    "top_k": 5,
    "rerank": true,
    "document_title": "my_doc"
  }'
```

| Tham số | Mô tả | Mặc định |
|---|---|---|
| `query` | Câu truy vấn | bắt buộc |
| `mode` | `sparse` / `dense` / `hybrid` / `hybrid_kg` / `graph_lookup` | `hybrid_kg` |
| `top_k` | Số kết quả | `RETRIEVAL_TOP_K` |
| `document_title` | Lọc theo tài liệu | tất cả |
| `rerank` | Bật reranking | `RETRIEVAL_RERANK_ENABLED` |

### `POST /chat`

Chat với agent (semantic + structured reasoning).

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "So sánh hiệu suất module A và B",
    "document_title": "system_report",
    "conversation_id": "<uuid>"
  }'
```

**Response:**
```json
{
  "answer": "Module A đạt..., trong khi Module B...",
  "citations": [
    {
      "document_title": "system_report",
      "section_path": "Performance/Benchmarks",
      "content_hash": "abc123",
      "position": 3
    }
  ],
  "sql_query": "SELECT module, metric FROM perf WHERE year='2024'",
  "reasoning_path": "structured",
  "tool_trace": [...],
  "timings_ms": {"classify": 12, "schema": 180, "extract": 340, "sql": 25, "synthesize": 210},
  "conversation_id": "<uuid>"
}
```

`reasoning_path` cho biết luồng nào được dùng: `"structured"` hoặc `"semantic"`.

### `POST /conversations`

Tạo conversation mới.

```bash
curl -X POST http://127.0.0.1:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{"title": "Phân tích báo cáo Q1"}'
# {"conversation_id": "<uuid>", "title": "...", "created_at": "..."}
```

### `GET /conversations`

Liệt kê conversations.

```bash
curl http://127.0.0.1:8000/conversations?limit=20
```

### `GET /conversations/{id}/messages`

Xem lịch sử tin nhắn.

```bash
curl http://127.0.0.1:8000/conversations/<uuid>/messages
```

### `GET /documents/{id}/graph-status`

Theo dõi tiến độ graph ingest của tài liệu.

```bash
curl http://127.0.0.1:8000/documents/<document_id>/graph-status
# {"graph_status": "done", "graph_progress": 1.0, "graph_processed_chunks": 24, ...}
```

---

## 6. Structured SQL Reasoning

Tự động kích hoạt khi phát hiện câu hỏi dạng so sánh, thống kê, xếp hạng.

### Các dạng câu hỏi được nhận diện

| Loại (`query_type`) | Ví dụ | Pattern kích hoạt |
|---|---|---|
| `comparison` | "So sánh A và B" | so sánh, compare, vs, versus |
| `aggregation` | "Tổng doanh thu là bao nhiêu?" | bao nhiêu, how many, tổng, average |
| `ranking` | "Top 5 sản phẩm bán chạy nhất" | top N, best, lớn nhất, xếp hạng |
| `multi_filter` | "Tất cả sản phẩm loại A và giá > 100" | tất cả...và, find all |
| `multi_hop` | "Qua quan hệ nào A kết nối với B?" | qua, through, via, quan hệ |

### Pipeline 5 bước

```
[1] Retrieve     → bootstrap_search (hybrid mode, expand query)
[2] Schema       → LLM infers minimal relational schema from top-10 chunks
[3] Extract      → Parallel LLM extraction per (chunk × table), CLEAR A+B validation
[4] SQL Compile  → LLM generates SELECT, execute SQLite in-memory, retry on error
[5] Synthesize   → LLM formats result (table/list/statement) + provenance citations
```

### CLEAR Validation

**Level A (per-row):**
- Drop rows có Primary Key null/rỗng
- Coerce numeric strings: `"42"` → `42`, `"3.14"` → `3.14`

**Level B (cross-row):**
- Dedup: cùng PK + cùng giá trị → merge
- Conflict: cùng PK + khác giá trị → giữ row từ chunk có `_source_position` nhỏ hơn (xuất hiện trước trong tài liệu)

### Fallback

Nếu bất kỳ bước nào thất bại, pipeline tự động fallback về semantic path:
```json
{"_structured_fallback": true, "_fallback_reason": "empty_schema"}
```

### Cấu hình classifier

```env
# Chỉ dùng regex (nhanh, không cần LLM call):
STRUCTURED_CLASSIFIER_METHOD=rule

# Chỉ dùng LLM (chính xác hơn, chậm hơn):
STRUCTURED_CLASSIFIER_METHOD=llm

# Dùng cả hai (rule trước, LLM nếu không match):
STRUCTURED_CLASSIFIER_METHOD=rule+llm
```

---

## 7. LLM Routing

Dùng model khác nhau cho từng task để tối ưu chi phí/tốc độ.

```env
LLM_ROUTING_ENABLED=true
LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","schema_discovery":"llama3.1:8b","sql_compile":"llama3.1:8b","synthesize":"qwen2.5:72b","answer":"qwen2.5:72b","insight":"qwen2.5:7b","report":"qwen2.5:72b"}
```

**Các task có thể route:**

| Task key | Mô tả | Gợi ý model |
|---|---|---|
| `classify` | Phân loại intent câu hỏi | model nhỏ, nhanh (3-7B) |
| `schema_discovery` | Suy luận schema từ chunks | model trung (7-8B) |
| `sql_compile` | Sinh SQL từ schema + câu hỏi | model trung (7-8B) |
| `synthesize` | Tổng hợp kết quả SQL thành câu trả lời | model lớn (32-72B) |
| `answer` | Trả lời semantic (cuối pipeline) | model lớn |
| `insight` | Rút ra business insights | model trung |
| `report` | Tổng hợp multi-section report | model lớn |
| `decide` | Quyết định tool tiếp theo trong agent loop | model nhỏ-trung |

**Cost tracking:**
```env
LLM_COST_TRACKING_ENABLED=true
```
Kết quả ghi trong memory, truy cập qua `LLMGateway.cost_summary()`.

---

## 8. Security Policy

Kiểm soát truy cập tài liệu/section ở query-time, không cần thay đổi DB schema.

### Định nghĩa policy

```python
from src.pam.common.security_policy import PolicyRegistry, DocumentPolicy

registry = PolicyRegistry()
registry.load_from_list([
    {
        "document_title": "internal_report",
        "denied_section_prefixes": ["Confidential/", "HR/"],
        "denied_section_patterns": [".*salary.*", ".*personal.*"],
        "denied_segment_types": ["footnote"],
        "max_results": 5,
    }
])
```

### Cách hoạt động

- `denied_section_prefixes`: Drop chunks có `section_path` bắt đầu bằng prefix này
- `denied_section_patterns`: Drop chunks khớp regex (case-insensitive)
- `denied_segment_types`: Drop chunks có `segment_type` trong danh sách
- `max_results`: Giới hạn số kết quả trả về cho tài liệu này

Policy được áp dụng tại `SecurityService.filter_tool_results()` sau mỗi lần retrieval, trước khi trả về cho user.

---

## 9. Observability & Tracing

Mỗi request tạo một `StageTracer` ghi lại thời gian từng giai đoạn.

**Response field `timings_ms`:**
```json
{
  "timings_ms": {
    "classify":   12.3,
    "retrieve":   45.7,
    "schema":    183.2,
    "extract":   341.0,
    "sql":        24.8,
    "synthesize": 218.5
  }
}
```

Bật/tắt:
```env
OBSERVABILITY_TRACE_ENABLED=true
```

**Dùng trực tiếp trong code:**
```python
from src.pam.common.tracing import StageTracer

tracer = StageTracer()
tracer.start("my_stage", service="my_service")
# ... logic ...
tracer.end("my_stage", rows_processed=42)

timings = tracer.as_timings_dict()   # {"my_stage": 123.4}
total   = tracer.total_elapsed_ms()  # tổng thời gian
```

---

## 10. Multi-Agent Workers

Dành cho phân tích phức tạp: chia câu hỏi lớn thành nhiều sub-questions, chạy song song.

```python
import asyncio
from src.pam.agents.data_agent import DataAgent, DataTask
from src.pam.agents.insight_agent import InsightAgent
from src.pam.agents.report_agent import ReportAgent
from src.pam.services.knowledge_service import KnowledgeService
from src.pam.services.llm_gateway import LLMGateway

# Setup
knowledge_service = KnowledgeService()
llm_gateway = LLMGateway()

data_agent    = DataAgent(knowledge_service, llm_gateway)
insight_agent = InsightAgent(llm_gateway)
report_agent  = ReportAgent(llm_gateway)

# Step 1: Thu thập dữ liệu song song
sub_questions = [
    DataTask(question="Doanh thu Q1 2024?",    document_title="finance"),
    DataTask(question="Doanh thu Q2 2024?",    document_title="finance"),
    DataTask(question="Chi phí vận hành 2024?", document_title="finance"),
]
data_results = await asyncio.gather(
    *[data_agent.run(task) for task in sub_questions]
)

# Step 2: Rút ra insights song song
insight_results = await asyncio.gather(
    *[insight_agent.run(dr) for dr in data_results]
)

# Step 3: Tổng hợp report
report = await report_agent.run(insight_results)
print(report.title)
print(report.summary)
for section in report.sections:
    print(f"## {section['heading']}")
    print(section['content'])
```

**DataAgent** tự động chọn luồng:
- `intent == structured` → `StructuredReasoningPipeline`
- `intent == semantic` → `KnowledgeService.bootstrap_search` + LLM

---

## 11. MCP Server

PAM có thể đóng vai MCP (Model Context Protocol) server, cho phép Claude Desktop hoặc client khác gọi PAM như một tool.

```python
from src.pam.mcp.server import PAMMCPServer

server = PAMMCPServer()

# Liệt kê tools
tools = server.list_tools()
# [{"name": "search", ...}, {"name": "structured_query", ...}]

# Gọi tool search
result = await server.handle_tool_call("search", {
    "query": "kiến trúc hệ thống",
    "document_title": "design_doc",
    "top_k": 5,
})

# Gọi tool structured_query
result = await server.handle_tool_call("structured_query", {
    "question": "So sánh hiệu suất module A và B",
    "document_title": "benchmark_report",
    "query_type": "comparison",
})
```

**Tools được expose:**

| Tool | Input | Output |
|---|---|---|
| `search` | `query`, `document_title?`, `top_k?` | `{results: [{content, score, section_path, ...}]}` |
| `structured_query` | `question`, `document_title?`, `query_type?` | `{answer, sql_query, citations}` |

Tất cả kết quả đều đi qua `SecurityService.filter_tool_results()` trước khi trả về.

---

## 12. Benchmark & Kiểm thử

```bash
# Benchmark ingest pipeline (chunking + parsing)
python3 scripts/benchmark_ingest.py data/test_docs/SYSTEM_DESIGN.md

# Benchmark ingest + embedding
python3 scripts/benchmark_ingest.py data/test_docs/SYSTEM_DESIGN.md --embed

# Benchmark graph extraction (chạy thử Graphiti với N chunks)
python3 scripts/benchmark_graph.py data/test_docs/SYSTEM_DESIGN.md --max-chunks 5

# Benchmark retrieval (đo accuracy + latency từ baseline JSON)
python3 scripts/benchmark_retrieval.py data/benchmarks/retrieval_baseline.json --top-k 5

# Benchmark agent (đo end-to-end chat response)
python3 scripts/benchmark_agent.py data/benchmarks/agent_baseline.json --repeat 1
```

### Kiểm tra nhanh từng tính năng

```bash
# Semantic chat
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Tính năng chính của hệ thống là gì?", "document_title": "my_doc"}'

# Structured chat (so sánh)
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "So sánh module A và module B về hiệu suất", "document_title": "my_doc"}'

# Structured chat (top-N)
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Top 3 module có latency thấp nhất?", "document_title": "benchmark"}'

# Search trực tiếp với rerank
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "retrieval architecture", "mode": "hybrid_kg", "top_k": 5, "rerank": true}'
```

### Kiểm tra response

Với `/chat`, kiểm tra các field sau:

```
reasoning_path  →  "structured" hoặc "semantic"
sql_query       →  câu SQL được chạy (chỉ có nếu structured)
timings_ms      →  latency từng giai đoạn
tool_trace      →  ít nhất 1 retrieval step
context         →  không rỗng
citations       →  có document_title + content_hash
```

Nếu `reranked: false` trong kết quả search, xem `rerank_reason` để biết lý do:
- `disabled` — `RETRIEVAL_RERANK_ENABLED=false`
- `invalid_schema` — model trả về JSON sai format
- `provider_error` — lỗi kết nối provider

---

## 13. Reset môi trường

```bash
# Dừng và xóa toàn bộ data
docker compose down -v --remove-orphans
rm -rf data/neo4j_data data/es_data data/postgres_data
rm -rf .cache/pam/graph

# Khởi động lại từ đầu
docker compose up -d
uv sync
uv run alembic upgrade head
uv run uvicorn main:app --reload --port 8000
```

**Lỗi thường gặp:**

| Lỗi | Nguyên nhân | Xử lý |
|---|---|---|
| `unsupported value: NaN` | Ollama embedding model không ổn định | Đổi sang `nomic-embed-text` hoặc `mxbai-embed-large` |
| `Connection refused :9200` | Elasticsearch chưa sẵn sàng | Chờ 30s sau `docker compose up` |
| `Auth failure Neo4j` | Sai password | Kiểm tra `NEO4J_PASSWORD` trong `.env` |
| `RuntimeWarning: module found in sys.modules` | Chạy module trực tiếp `python -m` | Benign warning, không ảnh hưởng |
| Structured path luôn fallback | Model quá nhỏ không theo JSON schema | Dùng model ≥7B, đặt `EXTRACTION_TEMPERATURE=0.0` |

---

## 14. Cấu trúc thư mục

```
PAM/
├── main.py                          # FastAPI app + endpoints
├── .env / .env.example              # Cấu hình
├── docker-compose.yml               # PostgreSQL, ES, Redis, Neo4j
├── migrations/                      # Alembic migrations
├── data/
│   ├── docs/                        # Markdown tài liệu để ingest
│   └── benchmarks/                  # Baseline JSON cho benchmark
├── scripts/                         # Benchmark scripts
├── docs/adr/
│   ├── 0001-target-architecture-rollout.md
│   └── 0002-structured-sql-reasoning.md
└── src/pam/
    ├── config.py                    # Pydantic Settings
    ├── config_validation.py         # Startup validation
    ├── main.py                      # FastAPI lifespan
    │
    ├── common/
    │   ├── tracing.py               # StageTracer
    │   └── security_policy.py       # DocumentPolicy, PolicyRegistry
    │
    ├── agent/
    │   ├── service.py               # AgentService — orchestrator chính
    │   ├── context.py               # ContextAssembler (4-stage pipeline)
    │   ├── llm.py                   # AgentLLM (multi-provider client)
    │   └── tools.py                 # AgentTools (tool registry + executor)
    │
    ├── services/
    │   ├── knowledge_service.py     # Retrieval facade, intent-aware
    │   ├── llm_gateway.py           # LLM routing + cost tracking
    │   ├── security_service.py      # Query-time policy gate
    │   └── context_assembly_service.py
    │
    ├── structured/                  # SQL Reasoning Pipeline (ADR 0002)
    │   ├── pipeline.py              # StructuredReasoningPipeline
    │   ├── query_classifier.py      # QueryIntentClassifier (L1+L2)
    │   ├── schema_discovery.py      # SchemaDiscoveryModule
    │   ├── extractor.py             # StructuredExtractor (CLEAR A+B)
    │   ├── sql_engine.py            # SQLReasoningEngine (SQLite)
    │   └── synthesizer.py           # AnswerSynthesizer
    │
    ├── agents/                      # Multi-Agent Workers
    │   ├── data_agent.py            # DataAgent
    │   ├── insight_agent.py         # InsightAgent
    │   └── report_agent.py          # ReportAgent
    │
    ├── mcp/
    │   └── server.py                # PAMMCPServer (search + structured_query)
    │
    ├── retrieval/
    │   ├── elasticsearch_retriever.py  # Hybrid retrieval (BM25+kNN+graph)
    │   └── reranker.py                 # LLMReranker / CrossEncoder
    │
    ├── ingestion/
    │   ├── pipeline.py              # ingest_folder()
    │   ├── chunkers/                # HybridChunker
    │   ├── connectors/              # MarkdownConnector
    │   ├── parsers/                 # DoclingParser
    │   ├── embedders/               # OpenAI/Gemini/HF/Ollama embedders
    │   └── stores/                  # PostgresStore, ElasticsearchStore
    │
    ├── graph/
    │   ├── graphiti_service.py      # Neo4j entity/relationship extraction
    │   ├── entity_sync.py           # ES ↔ Neo4j entity sync
    │   ├── hf_embedder.py           # HF embedder for graph
    │   └── graph_jobs.py            # Async graph ingest queue
    │
    ├── database/
    │   ├── models.py                # ORM: Document, Segment, Conversation
    │   └── __init__.py              # AsyncSessionLocal
    │
    ├── chat/
    │   └── history.py               # ConversationStore (Redis + PG)
    │
    └── health/
        └── providers.py             # Provider health checks
```
