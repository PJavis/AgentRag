# AgentRag

Nền tảng RAG (Retrieval-Augmented Generation) cho học liệu, đặc biệt phục vụ sinh viên y khoa. Hai luồng suy luận song song, bộ nhớ phân cấp, CLI tương tác, và UI tương thích open-notebook.

- **Semantic path** — Hybrid retrieval (BM25 + vector + StructMem entries) + LLM synthesis
- **Structured path** — SQL reasoning trên dữ liệu trích xuất từ văn bản
- **Chat StructMem** — Bộ nhớ hội thoại semantic thay thế sliding-window history
- **Page-aware citations** — Trích dẫn chỉ rõ số trang (NotebookLM-style) cho tài liệu PDF
- **Vision LLM** — Mô tả ảnh y tế (giải phẫu, X-quang, ECG…) trong PDF + ảnh standalone
- **Mindmap & Summary** — Sinh sơ đồ tư duy Mermaid + tóm tắt cấu trúc theo template y khoa
- **Open-notebook adapter** — Mount UI Next.js qua `/on`; admin panel reasoning trace tại `/admin`

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
14. [Page-Aware Citations & Vision](#14-page-aware-citations--vision)
15. [Mindmap & Structured Summary](#15-mindmap--structured-summary)
16. [Open-Notebook Adapter & Admin Panel](#16-open-notebook-adapter--admin-panel)
17. [Authentication (JWT + Google OAuth)](#17-authentication-jwt--google-oauth)
18. [Security Policy](#18-security-policy)
19. [Benchmark & Kiểm thử](#19-benchmark--kiểm-thử)
20. [Reset môi trường](#20-reset-môi-trường)
21. [Cấu trúc thư mục](#21-cấu-trúc-thư-mục)
22. [Module READMEs](#22-module-readmes)

---

## 1. Kiến trúc tổng quan

> S4 — codebase chia thành **Reasoning Plane** (agentic decision-making) và
> **Execution Plane** (IO workers + service facades) với
> `ServiceContainer` singleton là entry duy nhất cho Reasoning code lấy
> service. Chi tiết: [`ARCHITECTURE.md`](./ARCHITECTURE.md).

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
    ├── Parse:
    │     .pdf  → PDFParser (PyMuPDF) — page-aware + extract images
    │     .docx/.pptx/.html → MarkItDownParser
    │     .jpg/.png/...     → ImageParser → vision_response()
    │     .md/.txt          → MarkdownConnector
    │     .xlsx/.csv        → ExcelParser (markdown | sql mode)
    ├── Chunk (search 512 tok + graph 1536 tok)
    │     — strip page markers + assign page_start/page_end per chunk
    ├── Embed → PostgresStore + ElasticsearchStore (agentrag_segments)
    │     — text segments + image segments (segment_type="image")
    └── ARQ: enqueue graph_ingest job
            ├── StructMemService.sync_chunks()
            │       └── Per chunk: asyncio.gather(factual_call, relational_call)
            ├── index_structmem_views() → agentrag_entries
            └── [if chunks ≥ threshold] ARQ: enqueue consolidate job → agentrag_synthesis

POST /generate/mindmap | /generate/summary
    │
    ├── ES sparse_search (top_k=30 chunks per document_title)
    └── LLMGateway.json_response (single-shot, no agent loop)

GET /on/api/* (Next.js frontend)
    │
    └── adapter sub-app
        ├── notebook/source CRUD
        ├── /chat/execute (full agent) | source chat (direct RAG, isolated)
        └── search

GET /admin
    └── admin reasoning panel (HTML) + /admin/api/* (JSON traces)
```

---

## 2. Storage Layer

| Store | Vai trò | Indices / Tables |
|---|---|---|
| **PostgreSQL** | Source of truth: documents, segments, conversations + adapter notebooks/notes | `documents`, `segments`, `conversations`, `chat_messages`, `adapter_notebooks`, `adapter_notes`, `adapter_notebook_sources` |
| **Elasticsearch** | BM25 + kNN hybrid search + StructMem knowledge | `agentrag_segments` (có `page_start`, `page_end`, `segment_type`), `agentrag_entries`, `agentrag_synthesis`, `agentrag_chat_entries`, `agentrag_chat_synthesis` |
| **Redis** | Chat history cache (TTL) + ARQ job queue | key-value + sorted sets |
| **Filesystem** | Ảnh extract từ PDF + ảnh standalone | `IMAGE_STORAGE_DIR` (mặc định `data/images/`), serve qua `/images/*` static mount |

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

### Quick start (Makefile)

```bash
make install        # docker up + uv sync + npm install + alembic upgrade
make dev            # api + worker + frontend song song (Ctrl+C tắt tất cả)
```

UI mở tại http://localhost:3000, API tại http://localhost:8000.

### Make targets

`make help` để liệt kê tại runtime. Targets nhóm theo mục đích:

#### Setup & install

| Target | Mô tả |
|---|---|
| `make install` | One-shot setup: `docker-up` + tạo `.env` + `uv sync` + `npm install` + `migrate` |
| `make env` | Tạo `.env` từ `.env.example` + `frontend/.env.local` nếu chưa có |
| `make uv-sync` | Cài Python deps qua uv |
| `make frontend-install` | `npm install` trong `frontend/` |
| `make migrate` | `alembic upgrade head` |

#### Docker infra

| Target | Mô tả |
|---|---|
| `make docker-up` | Start postgres + elasticsearch + redis (default services) |
| `make docker-down` | Stop infra services |
| `make docker-up-llm` | + Ollama container (GPU profile), pull every model referenced in `.env` (`EMBEDDING_MODEL`, `EXTRACTION_MODEL`, `AGENT_MODEL`, `RETRIEVAL_RERANK_MODEL`, `VISION_MODEL` when provider=ollama, plus `OLLAMA_EXTRA_MODELS`) |
| `make ollama-pull` | Pull models from `.env` without restarting compose |
| `make ollama-pull-dry` | Preview which models would be pulled |
| `make docker-up-app` | Build + start full stack trong docker (api + worker + frontend) |
| `make docker-up-edge` | + Nginx reverse proxy tại port 80 |
| `make docker-down-app` | Stop tất cả app/edge services |

#### Run (dev)

| Target | Mô tả |
|---|---|
| `make dev` | Chạy api + worker + frontend song song foreground (Ctrl+C tắt tất cả) |
| `make api` | Chỉ Uvicorn dev (`--reload`) |
| `make api-prod` | Gunicorn multi-worker (đọc `UVICORN_WORKERS` từ env) |
| `make frontend` | Chỉ Next.js dev server |
| `make worker` | 1 ARQ worker |
| `make scaler` | ARQ auto-scaling worker pool |
| `make cli` | Interactive CLI chat |

#### Run (background)

Mỗi target ghi log vào `.run/<name>.log` + `.run/<name>.pid`.

| Target | Mô tả |
|---|---|
| `make api-bg` / `make worker-bg` / `make frontend-bg` / `make scaler-bg` | Chạy nền từng service |
| `make up-bg` | Chạy nền 3 services (api + worker + frontend) |
| `make logs` | Tail 30 dòng cuối từ tất cả log files |
| `make stop` | Kill tất cả background processes (đọc PID từ `.run/`) |

#### Vision LLM (Ollama)

```bash
# default model llava:13b
make vision-pull

# hoặc model nhẹ hơn nếu VRAM hạn chế
make vision-pull VISION_MODEL_TAG=llava:7b
```

Sau pull, set `.env`:
```env
VISION_PROVIDER=ollama
VISION_MODEL=llava:13b
VISION_BASE_URL=http://127.0.0.1:11434/v1/
```

#### Maintenance

| Target | Mô tả |
|---|---|
| `make health` | Probe `/config/validate` + `/on/api/auth/status` |
| `make test` | `pytest -q` |
| `make bench-ingest` | Benchmark ingest pipeline |
| `make clean` | Xoá `.cache`, `__pycache__`, `.next` |
| `make deepclean` | `clean` + `node_modules` + `.venv` |

#### Reset (3 mức độ)

| Mức | Target | Xoá những gì | Giữ lại |
|---|---|---|---|
| 🟡 **Soft** | `make reset` | DB volumes (postgres + ES + redis + ollama), `.cache/agentrag`. Restart infra + migrate. | Code, deps, `.venv`, `node_modules`, ảnh đã extract, logs |
| 🟠 **Data** | `make reset-data` | Tất cả của Soft + `data/images/*` + `.run/` (logs) | Code, deps, `.venv`, `node_modules` |
| 🔴 **Nuke** | `make nuke` | Tất cả của Data + `.venv` + `frontend/node_modules` + `.next` + tất cả docker containers (kể cả Ollama) | `.env` |

Sau khi nuke chạy `make install` để dựng lại từ đầu.

**Custom finetuned Ollama models** (`qwen-agentrag`, `agentrag-embed-v1`):

| Tình huống | Hành vi |
|---|---|
| `models/<name>*/Modelfile` còn (đã `make convert-llm`) | `make reseed-models` re-register exact GGUF, chất lượng finetune giữ |
| Chỉ HF dir còn (chưa convert) | Cần `make convert-llm` để build GGUF + Modelfile, rồi reseed |
| Không artifact nào | `make reseed-models` tự động create alias `FROM qwen2.5:7b-instruct` (chất lượng base). Script sẽ pull base model nếu chưa có, rồi tạo alias. |

**Cách alias hoạt động:**
```bash
# scripts/ensure_ollama_model.sh quản lý 5 bước resolution:
# 1. Kiểm tra model đã registered? → done
# 2. Pull từ registry (public model)?  → done
# 3. Local Modelfile từ finetune? → done
# 4. Local Modelfile matching pattern? → done
# 5. Fallback: tạo alias FROM base model
#    - Đảm bảo base model đã pulled trước
#    - Viết Modelfile vào /tmp, rồi ollama create
#    - Tạo lightweight alias tag
```

Recovery sau nuke đầy đủ:
```bash
make nuke
make install                  # uv + npm + alembic
make docker-up-llm            # auto reseed via ensure_ollama_model.sh
# Nếu muốn restore finetune quality:
make convert-llm              # rebuild GGUF từ models/qwen-agentrag-7b/
make reseed-models            # re-register exact
```

```bash
# Database hỏng / muốn ingest lại từ đầu
make reset

# Reset hoàn toàn data + ảnh + logs (giữ deps)
make reset-data

# Lỗi nặng / muốn build sạch hoàn toàn
make nuke && make install
```

#### Cấu hình runtime (override)

```bash
make api API_PORT=9000                   # đổi port API
make api UVICORN_RELOAD=                 # tắt --reload
make api-prod UVICORN_WORKERS=4          # multi-worker production
make vision-pull VISION_MODEL_TAG=...    # custom vision model
```

### Workflow điển hình

```bash
# Lần đầu setup
make install
# (review .env, set OPENAI_API_KEY hoặc bật docker-up-llm)
make dev                                # Ctrl+C để tắt

# Daily dev (background, free terminal)
make up-bg                              # api + worker + frontend chạy nền
make logs                               # xem logs
make stop                               # dừng

# Production-like local test
make docker-up-app                      # tất cả trong docker
curl localhost:8000/on/api/config

# Reset khi schema/data hỏng
make reset

### Manual (nếu không dùng Makefile)

```bash
cp .env.example .env
docker compose up -d
docker compose --profile local-llm up -d   # tuỳ chọn Ollama+GPU
uv sync
uv run alembic upgrade head
cd frontend && npm install && cd ..
# 4 process song song:
uv run uvicorn main:app --reload --port 8000
uv run arq src.agentrag.worker.settings.WorkerSettings
uv run python scaler.py                    # thay cho ARQ nếu muốn tự scale
cd frontend && npm run dev
```

Kiểm tra:
```bash
curl http://127.0.0.1:8000/config/validate
curl http://127.0.0.1:8000/health/providers
```

### Production deployment (docker compose)

Mọi service chạy trong docker, behind nginx:

```bash
# 1. Build + start (api, worker, frontend) — profile "app"
docker compose --profile app up -d --build

# 2. Optional: front bằng nginx tại port 80 — profile "edge"
docker compose --profile app --profile edge up -d --build

# 3. Pull vision model nếu dùng Ollama
make vision-pull               # pulls llava:13b vào agentrag-ollama
```

Nginx config: `deploy/nginx.conf` (proxy `/on/*`, `/admin`, `/images/*`, `/chat`, `/ingest`, ... → API; còn lại → Next.js).

Multi-worker: set `UVICORN_WORKERS=4` trong `.env` (dùng cho `make api-prod` hoặc Dockerfile CMD).

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

### Tier 3a — GPU 16 GB VRAM (full feature set, qwen 7B + vision)

Recommended for laptop/workstation users who want every feature on (vision,
StructMem, mindmap, summary, transformations) without OOM. Concurrent loaded:
qwen2.5:7b (~5GB) + llava:7b (~4.5GB) + llama3.2:3b (~2GB) + mxbai-embed-large (~0.7GB).

```env
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=mxbai-embed-large
EXTRACTION_PROVIDER=ollama
EXTRACTION_MODEL=qwen2.5:7b-instruct
AGENT_PROVIDER=ollama
AGENT_MODEL=qwen2.5:7b-instruct
VISION_PROVIDER=ollama
VISION_MODEL=llava:7b
VISION_INGEST_MODE=async                 # vital — vision blocks ingest otherwise
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_BACKEND=local_cross_encoder
STRUCTURED_REASONING_ENABLED=true
LLM_ROUTING_ENABLED=true
LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","decide":"llama3.2:3b","schema_discovery":"qwen2.5:7b-instruct","sql_compile":"qwen2.5:7b-instruct","answer":"qwen2.5:7b-instruct","mindmap":"qwen2.5:7b-instruct","summary":"qwen2.5:7b-instruct"}
STRUCTMEM_MAX_CONCURRENCY=2
```

Also bump `OLLAMA_MAX_LOADED_MODELS=3` in `docker-compose.yml` so 3 LLMs stay
hot together.

### Tier 3b — GPU 24 GB VRAM (qwen 14B/32B)

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

### Tier 5 — 6 GB VRAM (laptop) + Gemini cloud

Embedding stays local on Ollama (`nomic-embed-text` ≈ 300 MB VRAM). Everything
else routes to Gemini. Tested on Windows/WSL2 with 6 GB GPU.

```env
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EXTRACTION_PROVIDER=gemini
EXTRACTION_MODEL=gemini-2.5-flash
AGENT_PROVIDER=gemini
AGENT_MODEL=gemini-2.5-pro
VISION_PROVIDER=gemini
VISION_MODEL=gemini-2.5-flash
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_PROVIDER=gemini
RETRIEVAL_RERANK_MODEL=gemini-2.5-flash-lite
LLM_ROUTING_ENABLED=true
LLM_TASK_MODEL_MAP={"classify":"gemini-2.5-flash-lite","decide":"gemini-2.5-flash-lite","schema_discovery":"gemini-2.5-flash","sql_compile":"gemini-2.5-flash","synthesize":"gemini-2.5-pro","answer":"gemini-2.5-pro"}
LLM_COST_TRACKING_ENABLED=true
# Free Gemini tier: 10 RPM for 2.5-flash; bump to 1000 on paid tier.
VISION_MAX_RPM=10
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

### 5.6 LLM Routing & Cost Tracking

```env
LLM_ROUTING_ENABLED=false
LLM_TASK_MODEL_MAP={}
LLM_COST_TRACKING_ENABLED=false     # flip true → GET /on/api/metrics/cost

# Auto-route to large-context model when prompt > threshold tokens
# (open-notebook provision_langchain_model pattern). Bỏ trống = disabled.
# Ví dụ: gemini-2.5-pro (1M ctx), qwen2.5:32b (128k ctx)
LLM_LARGE_CONTEXT_MODEL=
LLM_LARGE_CONTEXT_THRESHOLD=100000
```

Task keys: `classify`, `decide`, `schema_discovery`, `sql_compile`, `synthesize`, `answer`, `mindmap`, `summary`, `transformation`

When cost tracking is on, every LLM call is logged in a process-local ring
buffer (last 5000 calls) with estimated USD via public Gemini / OpenAI pricing.

- `GET /on/api/metrics/cost` — per-task + per-model breakdown (`calls`, `in_tokens`, `out_tokens`, `usd`, `avg_latency_ms`)
- `POST /on/api/metrics/cost/reset` — clear ledger

### 5.6b Agent Harness (Context + Plan + Critique)

```env
# Token-aware context budget (replaces chunk-count cap when >0)
AGENT_MAX_CONTEXT_TOKENS=6000
# Lost-in-the-middle reorder: best chunks at start + end of packed context
AGENT_LOST_IN_MIDDLE_REORDER=true

# Self-critique 2nd pass — verifies draft against context. Fires only when
# retrieval is thin (top RRF < threshold). +1 LLM call/turn.
AGENT_SELF_CRITIQUE_ENABLED=false
AGENT_SELF_CRITIQUE_RRF_THRESHOLD=0.05

# Plan-then-execute: planner decomposes multi-hop questions into sub-queries,
# parallel retrieval, then single answer pass. Skipped for short questions.
AGENT_PLAN_THEN_EXECUTE_ENABLED=true
AGENT_PLAN_TRIGGER_MIN_CHARS=60
AGENT_PLAN_MAX_SUBQUERIES=4

# Orchestrator backend:
#   loop      = hand-rolled chat() loop (battle-tested)
#   langgraph = StateGraph with 13 nodes (checkpoint + replay)
AGENT_BACKEND=loop
```

**LangGraph backend** (`AGENT_BACKEND=langgraph`): same nodes as legacy loop
(validate → memory → chitchat_check → classify → structured/semantic →
plan → bootstrap → decide ⇄ tool_exec → assemble → answer → ground) but
orchestrated as a `StateGraph` with `InMemorySaver` checkpointer. Each
turn's state is persisted by `thread_id = conversation_id` → resume from
any node, inspect state via `_GRAPH.aget_state(config)`.

**Chit-chat fast-path**: short messages with greeting/thanks tokens
(`hi`, `chào`, `thanks`, `cảm ơn`, `how are you`, ...) skip retrieval and
answer via the cheap routing model. Rule-based — no env knob.

**User feedback**: every assistant message in the UI has thumbs-up/down
buttons. Ratings persisted to `adapter_chat_feedback` table for later
prompt tuning / preference-pair dataset.

### 5.7 Auto-scaler

```env
SCALER_MIN_WORKERS=1
SCALER_MAX_WORKERS=4
SCALER_SCALE_UP_AT=5        # +1 worker mỗi 5 jobs trong queue
SCALER_POLL_SECONDS=5
SCALER_COOLDOWN_SECONDS=30
```

### 5.8 PDF & Vision

```env
PDF_PARSER_BACKEND=pymupdf       # pymupdf (page-aware) | markitdown (legacy)

# Vision LLM — bỏ qua image parsing nếu không set
VISION_PROVIDER=openai           # openai | gemini | ollama
VISION_MODEL=gpt-4o              # gpt-4o | gemini-1.5-flash | llava:13b
VISION_BASE_URL=                 # override endpoint (cho Ollama)
IMAGE_STORAGE_DIR=data/images
IMAGE_MIN_SIZE_BYTES=5000        # bỏ qua icon nhỏ

# Vision worker concurrency + RPM cap (used by vision_extract ARQ job).
# Free Gemini tier (2.5-flash): 10 RPM. Paid: ~1000.
# Ollama (llava): keep concurrency=1 to avoid GPU thrashing.
VISION_MAX_CONCURRENCY=4
VISION_MAX_RPM=10                # 0 = disable RPM cap
VISION_PER_IMAGE_RETRIES=3
VISION_FLUSH_BATCH_SIZE=10       # PG+ES commit every N described images (progress visibility)
VISION_TIMEOUT_SECONDS=180       # llava cold-start có thể > 60s
# sync: describe inline (blocks pipeline). async: queue ARQ vision_extract
# → text retrieval ready ngay; image segments lấp dần khi vision xong.
VISION_INGEST_MODE=async

# Persist original uploaded bytes so UI 'Open original' button works.
# Empty/None = bytes discarded after ingest.
ORIGINALS_DIR=data/originals
```

Image-heavy PDFs (e.g. 100-page scanned thesis) are processed by the
`vision_extract` ARQ worker job. Describes each page with the vision LLM,
embeds, then upserts segments to Postgres + Elasticsearch in batches —
ES `docs.count` climbs every batch instead of one bulk write at the end.

### 5.9 Open-Notebook Adapter

```env
OPEN_NOTEBOOK_PASSWORD=demo123        # bỏ trống = no auth
ADAPTER_ADMIN_TOKEN=admin_secret_123  # bỏ trống = admin disabled
ADAPTER_VERSION=0.7.0
```

### 5.10 Ontology & Domain Routing (S5)

Knowledge base chia theo **hệ cơ quan × chuyên khoa lâm sàng** với shared
ontology + cross-domain federation. Mọi chunk được gắn `system_tag`
(15 hệ: `tim_mach`, `ho_hap`, `tieu_hoa`, …) và `specialty_tag`
(14 chuyên khoa: `noi`, `ngoai`, `san`, …). Mỗi câu hỏi đi qua
`DomainRouter` (SLM, JSON-only) chấm điểm domain rồi `FederatedRetriever`
truy hồi top-1 (confidence ≥ 0.7) hoặc top-K khi mơ hồ.

```env
TAGGING_ENABLED=true
DOMAIN_FILTER_ENABLED=true
DOMAIN_ROUTER_CONFIDENCE_THRESHOLD=0.7
DOMAIN_ROUTER_TOP_K=3
```

Tagging dùng **section-primary với content fallback**: `SectionTagger`
quét `section_path` qua ontology trước (strict, no fuzzy), nếu không bắt
được hệ thì rơi xuống `find_in_text` trên chunk content. UI override:
dropdown `DomainFilter` trên ChatPanel (Hệ cơ quan + Chuyên khoa) — khi
user chọn thủ công thì bỏ qua router.

**Seed ontology** (idempotent, upsert theo `(canonical_norm, source)`):

```bash
make migrate            # tạo bảng + bật pg_trgm
python scripts/seed_ontology.py \
  --yaml data/ontology/custom_terms.yaml \
  --icd10 data/ontology/icd10_vn.csv
python scripts/backfill_tags.py     # re-tag ES segments
```

**Adapter taxonomy endpoints** (public, không cần token):

```
GET /on/api/ontology/systems       → [{value, label}, …]   # 15 hệ
GET /on/api/ontology/specialties   → [{value, label}, …]   # 14 chuyên khoa
```

**Chat domain filter** (notebook chat):

```json
POST /on/api/chat/execute
{
  "session_id": "…",
  "message": "…",
  "context": {…},
  "domain_filter": {"system": "tim_mach", "specialties": ["noi"]}
}
```

Chi tiết kiến trúc: xem `src/agentrag/ontology/README.md`.

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

### Adapter endpoints (`/on/api/*`)

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/on/api/metrics/cost` | Per-task + per-model LLM cost ledger summary |
| `POST` | `/on/api/metrics/cost/reset` | Clear in-memory cost ledger |
| `POST` | `/on/api/chat/feedback` | Upsert thumbs-up/down on assistant turn (body: `{turn_id, rating: +1\|-1, session_id?, comment?}`) |
| `GET` | `/on/api/models/defaults` | Effective model defaults (`.env` + JSON overrides) |
| `PUT` | `/on/api/models/defaults` | Persist defaults to `data/model_overrides.json` (live-applied) |
| `GET` | `/on/api/images/{path}` | Static-serve extracted PDF images (no auth) |

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

### `POST /generate/mindmap`

```bash
curl -X POST http://127.0.0.1:8000/generate/mindmap \
  -H "Content-Type: application/json" \
  -d '{"document_title": "giai_phau", "focus_topic": null, "max_depth": 3}'
# {"mermaid": "mindmap\n  root((Title))\n    Branch...", "concepts": [...], "cached": false}
```

### `POST /generate/summary`

```bash
curl -X POST http://127.0.0.1:8000/generate/summary \
  -d '{"document_title": "giai_phau", "style": "clinical"}'
# style: study_note | clinical | quick_review
# {"title", "overview", "sections": [{heading, summary, key_points, important_terms}]}
```

### Open-notebook adapter (`/on/api/*`)

| Endpoint | Mô tả |
|---|---|
| `GET /on/api/config` | Version + DB status (public, dùng cho frontend bootstrap) |
| `GET /on/api/notebooks` · `POST` · `PUT/{id}` · `DELETE/{id}` | Notebook CRUD |
| `POST /on/api/sources` (multipart) | Upload file → `ingest_folder()` → trả `SourceResponse` |
| `POST /on/api/chat/execute` | Notebook chat (full agent) |
| `POST /on/api/sources/{id}/chat/sessions/{sid}/messages` | Source chat SSE (direct RAG, document-isolated) |
| `POST /on/api/search` | Search sources/notes |

Auth: `Authorization: Bearer ${OPEN_NOTEBOOK_PASSWORD}`. Xem [adapter README](src/agentrag/adapter/README.md).

### Admin reasoning panel

```
GET /admin                                    # HTML inspector
GET /admin/api/conversations                  # cần X-Admin-Token
GET /admin/api/conversations/{id}/trace       # tool_trace per assistant message
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

## 14. Page-Aware Citations & Vision

### Citations với số trang (NotebookLM-style)

Khi `PDF_PARSER_BACKEND=pymupdf` (mặc định), mỗi chunk được tag `page_start` / `page_end`:

```json
{
  "document_title": "giai_phau_lam_sang",
  "section_path": "Chương 3 / Hệ tim mạch / Tim",
  "page_start": 47,
  "page_end": 48,
  "excerpt": "Van hai lá nằm giữa tâm nhĩ trái...",
  "content_hash": "abc123"
}
```

Cơ chế: `PDFParser` chèn marker `\x00P{N}\x00` vào full text (vô hình với LLM/chunker), `HybridChunker` strip marker và assign page range cho mỗi chunk.

Markdown / DOCX không có page info → `page_start = page_end = null`.

### Vision LLM cho ảnh y tế

Khi `VISION_PROVIDER` được set:
- **PDF**: `PDFParser.extract_images()` lưu ảnh vào `IMAGE_STORAGE_DIR/{slug(title)}/p{page}_{idx}.{ext}`, `ImageParser` mô tả qua vision LLM, tạo image segment có `segment_type="image"` + `image_url` + `page`
- **Standalone**: upload `.jpg/.png/...` → `ImageParser` xử lý trực tiếp

System prompt được tune cho ngữ cảnh y khoa: identify image type, anatomical structures, labels, pathological findings.

```env
VISION_PROVIDER=openai
VISION_MODEL=gpt-4o
# hoặc local:
VISION_PROVIDER=ollama
VISION_MODEL=llava:13b
```

Image chunks trong API response:
```json
{
  "segment_type": "image",
  "image_url": "/images/giai_phau/p47_0.jpg",
  "content": "Hình 3.2: Van hai lá nhìn từ trên xuống. Mũi tên chỉ hai lá van...",
  "page": 47
}
```

Ảnh được serve qua FastAPI static mount `/images/*`.

---

## 15. Mindmap & Structured Summary

Hai service trong `src/agentrag/generation/`. Đều retrieve top chunks từ ES rồi gọi LLM single-shot (không qua agent loop).

### Mindmap (`POST /generate/mindmap`)

Sinh Mermaid mindmap + concept hierarchy:

```bash
curl -X POST http://localhost:8000/generate/mindmap \
  -d '{"document_title": "giai_phau", "max_depth": 3}'
```

```json
{
  "mermaid": "mindmap\n  root((Giải phẫu))\n    Hệ tuần hoàn\n      Tim\n      Mạch máu",
  "concepts": [{"name": "Tim", "parent": "Hệ tuần hoàn", "level": 2}],
  "cached": false
}
```

In-process cache TTL 24h (key = `title|focus_topic|depth`).

### Summary (`POST /generate/summary`)

3 styles:

| Style | Pipeline | Khi dùng |
|---|---|---|
| `study_note` | Iterate 9 sections y khoa song song | Ghi chú học tập đầy đủ |
| `clinical` | Iterate 9 sections, prompt thiên về lâm sàng | Tóm tắt cho phòng khám |
| `quick_review` | Single LLM call, output gọn | Cheat sheet ôn nhanh |

**Medical template** (Vietnamese):
```
Định nghĩa & Phân loại  →  Dịch tễ học  →  Nguyên nhân & Yếu tố nguy cơ
Sinh lý bệnh  →  Triệu chứng lâm sàng  →  Cận lâm sàng & Chẩn đoán
Điều trị  →  Biến chứng  →  Tiên lượng & Theo dõi
```

Mỗi section trả `summary`, `key_points` (3-6 bullets), `important_terms` (term + definition).

### Highlights trong chat

`AgentService.chat()` và adapter `_direct_rag()` đều trả thêm `highlights: list[str]` — 3-5 điểm quan trọng nhất từ câu trả lời. Câu trả lời cũng dùng `**bold**` cho thuật ngữ.

Xem [generation README](src/agentrag/generation/README.md).

---

## 16. Open-Notebook Adapter & Admin Panel

AgentRag mount sub-app `/on` tương thích với [open-notebook](https://github.com/lfnovo/open-notebook) Next.js frontend — không cần sửa frontend.

### Setup frontend

Frontend được vendor sẵn tại `frontend/` (fork từ [lfnovo/open-notebook](https://github.com/lfnovo/open-notebook), tinh chỉnh `LoginForm` để hỗ trợ signup + Google).

```bash
make frontend         # auto-load frontend/.env.local, http://localhost:3000
```

Hoặc thủ công:

```bash
cd frontend
cp .env.local.example .env.local      # API_URL=http://localhost:8000/on
npm install
npm run dev
```

Đăng nhập:
- **Email/password** — đăng ký trực tiếp ở trang `/login` (tab "Đăng ký").
- **Google** — set `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET` trong `.env`, redirect URI `http://localhost:8000/on/api/auth/google/callback`.
- **Legacy bearer** — `OPEN_NOTEBOOK_PASSWORD` trong `.env` vẫn hoạt động (backward compat).

### Hai mode chat

| Mode | Endpoint | Strategy |
|---|---|---|
| Notebook chat | `POST /on/api/chat/execute` | Full `AgentService.chat()` — có thể cross-document, dùng StructMem |
| Source chat | `POST /on/api/sources/{id}/chat/sessions/{sid}/messages` (SSE) | `_direct_rag()` — strict isolation theo document |

Source chat dùng client-side filter trên `document_title` để **tránh leak context từ document khác** qua graph memory.

### Admin reasoning panel — `/admin`

LangGraph-style HTML inspector. Sidebar list conversations, main hiển thị:
- Messages (user + assistant)
- Flow diagram (START → tool steps → ANSWER)
- Step details: tool_name, tool_input, tool_output, duration_ms

Bảo vệ bằng `ADAPTER_ADMIN_TOKEN` (nhập trong UI), độc lập với `OPEN_NOTEBOOK_PASSWORD`.

Xem [adapter README](src/agentrag/adapter/README.md).

---

## 17. Authentication (JWT + Google OAuth)

Khi `AUTH_ENABLED=true`, mọi `/on/api/*` (trừ public) yêu cầu Bearer token. Hai cách lấy token:

### Email + password

```bash
# Signup
curl -X POST http://localhost:8000/on/api/auth/signup \
  -d '{"email": "user@example.com", "password": "...", "name": "..."}'
# → {"token": "<jwt>", "user": {...}}

# Login
curl -X POST http://localhost:8000/on/api/auth/login \
  -d '{"email": "user@example.com", "password": "..."}'
```

JWT TTL = `JWT_TTL_DAYS` (mặc định 7 ngày). Password hash bằng bcrypt.

### Google OAuth

```
GET /on/api/auth/google/start          → redirect tới Google consent screen
GET /on/api/auth/google/callback?code  → exchange code → JWT, set cookie, redirect FRONTEND_URL
```

Cấu hình:
```env
AUTH_ENABLED=true
AUTH_ALLOW_SIGNUP=true                    # cho phép signup mở (false: chỉ admin tạo user)
JWT_SECRET=<long-random-string>           # auto-derived ở dev nếu để trống
JWT_TTL_DAYS=7

GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=http://localhost:8000/on/api/auth/google/callback
FRONTEND_URL=http://localhost:3000
```

### Endpoints

| Endpoint | Mô tả |
|---|---|
| `POST /on/api/auth/signup` | Tạo user mới |
| `POST /on/api/auth/login` | Đổi email+password lấy JWT |
| `GET  /on/api/auth/me` | Profile user hiện tại (cần Bearer) |
| `GET  /on/api/auth/status` | Public — báo `auth_enabled` + `signup_enabled` để frontend hiện form |
| `GET  /on/api/auth/google/start` | Bắt đầu OAuth flow |
| `GET  /on/api/auth/google/callback` | OAuth callback |
| `POST /on/api/auth/logout` | Clear cookie |

### Rate limit + upload hardening

Khi `RATE_LIMIT_ENABLED=true`:
- `RATE_LIMIT_PER_MIN_DEFAULT=120` per-user/min cho chat + search
- `RATE_LIMIT_UPLOAD_PER_MIN=20` per-user/min cho upload
- `UPLOAD_MAX_BYTES=104857600` (100 MB)
- `UPLOAD_DEDUPE_BY_HASH=true` — skip re-ingest nếu đã thấy bytes

Implementation: `src/agentrag/adapter/rate_limit.py` + `upload_dedupe.py` (Redis INCR + EXPIRE 60s).

### Legacy shared password

Nếu cả `AUTH_ENABLED=false` và `OPEN_NOTEBOOK_PASSWORD` được set, middleware vẫn check Bearer token = password (tương thích deployment cũ).

---

## 18. Security Policy

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

## 19. Benchmark & Kiểm thử

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

## 20. Reset môi trường

3 mức độ reset, từ nhẹ đến mạnh:

```bash
# 🟡 Soft — xoá DB + cache, giữ deps & ảnh & logs
make reset

# 🟠 Data — soft + ảnh + logs (giữ deps)
make reset-data

# 🔴 Nuke — xoá sạch luôn cả .venv, node_modules, ollama, .next
make nuke && make install
```

Manual (không Makefile):

```bash
docker compose --profile app --profile edge down -v --remove-orphans
rm -rf .cache/agentrag data/images/* .run

docker compose up -d
uv sync
uv run alembic upgrade head
make dev
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

## 21. Cấu trúc thư mục

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
    │   ├── functions.py                 # graph_ingest, consolidate, chat_memory, vision_extract
    │   ├── pool.py                      # ARQ pool singleton (init/get/close)
    │   └── settings.py                  # WorkerSettings cho arq CLI
    │
    ├── observability/
    │   └── cost.py                      # Process-global LLM cost ledger
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
    │   ├── chunkers/                    # HybridChunker (page-aware)
    │   ├── connectors/
    │   ├── parsers/
    │   │   ├── pdf_parser.py            # PyMuPDF — page-aware + extract_images()
    │   │   ├── image_parser.py          # Vision LLM mô tả ảnh
    │   │   ├── markitdown_parser.py     # DOCX/PPTX/HTML
    │   │   └── excel_parser.py
    │   ├── embedders/
    │   └── stores/
    │
    ├── services/
    │   ├── llm_gateway.py               # json_response + vision_response, routing, cost tracking
    │   ├── knowledge_service.py
    │   ├── security_service.py
    │   └── context_assembly_service.py
    │
    ├── generation/                      # Học liệu artifacts
    │   ├── mindmap_service.py           # Mermaid mindmap + concept hierarchy
    │   └── summary_service.py           # Structured medical summary
    │
    ├── adapter/                         # Open-notebook compatible API + admin
    │   ├── app.py                       # FastAPI sub-app mount tại /on
    │   ├── auth.py                      # OpenNotebookAuthMiddleware
    │   ├── db.py                        # AdapterNotebook, AdapterNote tables
    │   ├── models.py                    # Pydantic schemas
    │   ├── admin.py                     # /admin reasoning inspector
    │   └── routers/                     # notebooks, sources, chat, notes, search, config
    │
    ├── database/                        # ORM models + AsyncSessionLocal
    ├── common/                          # StageTracer, SecurityPolicy
    └── health/                          # Provider health checks
```

---

## 22. Module READMEs

| Module | README |
|---|---|
| Ingestion Pipeline | [src/agentrag/ingestion/README.md](src/agentrag/ingestion/README.md) |
| StructMem (doc) | [src/agentrag/graph/README.md](src/agentrag/graph/README.md) |
| Chat & StructMem | [src/agentrag/chat/README.md](src/agentrag/chat/README.md) |
| Retrieval | [src/agentrag/retrieval/README.md](src/agentrag/retrieval/README.md) |
| Agent (Semantic Loop) | [src/agentrag/agent/README.md](src/agentrag/agent/README.md) |
| Structured SQL | [src/agentrag/structured/README.md](src/agentrag/structured/README.md) |
| Services | [src/agentrag/services/README.md](src/agentrag/services/README.md) |
| Generation (Mindmap/Summary) | [src/agentrag/generation/README.md](src/agentrag/generation/README.md) |
| Open-Notebook Adapter & Admin | [src/agentrag/adapter/README.md](src/agentrag/adapter/README.md) |
| Background Worker | [src/agentrag/worker/README.md](src/agentrag/worker/README.md) |
| CLI | [src/agentrag/cli/README.md](src/agentrag/cli/README.md) |
| MCP Server | [src/agentrag/mcp/README.md](src/agentrag/mcp/README.md) |
| Common Utilities | [src/agentrag/common/README.md](src/agentrag/common/README.md) |
