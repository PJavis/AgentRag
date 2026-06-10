# AgentRag

Nền tảng RAG cho học liệu y khoa Việt Nam. Hai luồng suy luận song song, bộ
nhớ phân cấp, CLI tương tác, UI Next.js tương thích open-notebook, có
domain-aware retrieval + reasoning trace + cost dashboard.

### Tính năng

- **Reasoning Plane / Execution Plane** (S4) — Layered architecture với
  `ServiceContainer` DI; reasoning fetch service qua Protocol, không tự
  instantiate. Xem [`ARCHITECTURE.md`](../ARCHITECTURE.md).
- **Domain partition** (S5) — KB chia theo `hệ cơ quan × chuyên khoa` (15×14).
  Shared ontology + `pg_trgm` fuzzy + `DomainRouter` SLM + UI override dropdown
  trên ChatPanel. Xem `src/agentrag/ontology/README.md`.
- **Cost & token dashboard** (S1) — `/cost` page, per-task / per-model summary
  với p50/p95 latency, recent-calls feed.
- **Reasoning trace** (S2) — Nút "Trace" trên mỗi AI bubble → LangGraph-style
  node graph (`plan → decide → tool → assemble → answer → critique`) +
  expandable tool I/O + plan sub-queries + SQL.
- **Embedding cache** (S3) — TTL 600s cho query path; ES result cache 60s đã có.
- **Semantic + Structured paths** — Hybrid (BM25 + kNN + RRF + KG) hoặc SQL
  reasoning trên rows trích xuất.
- **Chat StructMem** — Semantic conversation memory thay sliding-window.
- **Page-aware citations** — Số trang chính xác cho PDF (NotebookLM-style).
- **Vision LLM** — Mô tả ảnh y tế trong PDF + ảnh standalone.
- **MinerU backend (opt-in)** — Layout + OCR + formula → LaTeX + table → HTML một lượt. PPTX cũng có thể đi qua libreoffice → PDF → MinerU.
- **Mindmap & Summary** — Mermaid mindmap + cấu trúc tóm tắt y khoa.
- **RAG enhancement (2026-06, mặc định OFF)** — bật từng cờ để A/B
  ([§5.5](#55-retrieval--reranking)):
  - **Contextual Retrieval** — LLM thêm câu ngữ cảnh vào mỗi chunk trước khi
    embed/BM25 (cite vẫn dùng `content` gốc).
  - **RAPTOR** — lớp tóm tắt đa tầng (cluster + summarize đệ quy) trong cùng index.
  - **CRAG critique + multi-hop** — node kiểm tra grounding/relevance,
    re-retrieve sửa lỗi có giới hạn + chuỗi multi-hop.
  - **Adaptive routing** — fast-path bỏ qua vòng lặp agent cho câu hỏi đơn giản.
  - **Semantic retrieval cache** — cache kết quả theo độ tương đồng embedding.
- **UI signal surfacing** — MessageSignals chips, TraceDialog nâng cấp,
  RAPTOR/contextual citation hover, cost-dashboard recharts charts làm lộ các
  tín hiệu RAG enhancement trên UI.

---

## Mục lục

> **TL;DR**: §3 → §4 → §5.10 (S5 seed). Dashboard ở §5.x cost / §16 adapter.

1. [Kiến trúc tổng quan](#1-kiến-trúc-tổng-quan) — bao gồm S4 plane split
2. [Storage Layer](#2-storage-layer)
3. [Yêu cầu hệ thống](#3-yêu-cầu-hệ-thống)
4. [Cài đặt & Khởi động](#4-cài-đặt--khởi-động) — quick start 5 lệnh
5. [Cấu hình `.env`](#5-cấu-hình-env) — bao gồm §5.10 Ontology & Domain Routing (S5)
6. [API Reference](#6-api-reference) — bao gồm `/metrics/cost` (S1), `/ontology/*` (S5)
7. [CLI](#7-cli)
8. [StructMem — Knowledge Extraction](#8-structmem--knowledge-extraction)
9. [Chat StructMem — Bộ nhớ hội thoại](#9-chat-structmem--bộ-nhớ-hội-thoại)
10. [Background Workers & Auto-scaler](#10-background-workers--auto-scaler)
11. [Structured SQL Reasoning](#11-structured-sql-reasoning)
12. [LLM Routing](#12-llm-routing)
13. [MCP Server](#13-mcp-server)
14. [Page-Aware Citations & Vision](#14-page-aware-citations--vision)
15. [Mindmap & Structured Summary](#15-mindmap--structured-summary)
16. [Open-Notebook Adapter & Admin Panel](#16-open-notebook-adapter--admin-panel) — bao gồm Cost dashboard (S1) + Trace dialog (S2)
17. [Authentication (JWT + Google OAuth)](#17-authentication-jwt--google-oauth)
18. [Security Policy](#18-security-policy)
19. [Benchmark & Kiểm thử](#19-benchmark--kiểm-thử)
20. [Reset môi trường](#20-reset-môi-trường)
21. [Cấu trúc thư mục](#21-cấu-trúc-thư-mục)
22. [Module READMEs](#22-module-readmes) — bao gồm tag map sub-projects

---

## 1. Kiến trúc tổng quan

> S4 — codebase chia thành **Reasoning Plane** (agentic decision-making) và
> **Execution Plane** (IO workers + service facades) với
> `ServiceContainer` singleton là entry duy nhất cho Reasoning code lấy
> service. Chi tiết: [`ARCHITECTURE.md`](../ARCHITECTURE.md).

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
    │     .pdf  → PDFParser — text-layer first; tiered escalation per page:
    │              hybrid backend  → Tesseract → Vision LLM (fallback)
    │              mineru backend  → MinerU (layout + OCR + formula + table, one pass)
    │     .pptx → MarkItDownParser  (or libreoffice → PDF → MinerU when INGEST_USE_MINERU_FOR_PPTX)
    │     .docx/.html → MarkItDownParser
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
            ├── index_structmem_views() → agentrag_memory_doc (kind=entry)
            └── [if chunks ≥ threshold] ARQ: enqueue consolidate job → agentrag_memory_doc (kind=synthesis)

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
| **Elasticsearch** | BM25 + kNN hybrid search + StructMem knowledge | `agentrag_segments` (có `page_start`, `page_end`, `segment_type`), `agentrag_memory_doc` (doc entries + synthesis, `kind` discriminator), `agentrag_memory_chat` (chat entries + synthesis, `kind` discriminator) |
| **Valkey** | Chat history cache (TTL) + ARQ job queue + cost ledger stream + mindmap cache | key-value, sorted sets, streams (RESP — Redis client compatible) |
| **Filesystem** | Ảnh extract từ PDF + ảnh standalone | `IMAGE_STORAGE_DIR` (mặc định `data/images/`), serve qua `/images/*` static mount |

**ES Indices:**

| Index | Nội dung | Dùng cho |
|---|---|---|
| `agentrag_segments` | Chunks gốc từ tài liệu | Hybrid search (BM25 + kNN) |
| `agentrag_memory_doc` | Doc memory — `kind=entry` (factual + relational) AND `kind=synthesis` (cross-chunk hypotheses) | Knowledge retrieval + multi-hop reasoning |
| `agentrag_memory_chat` | Chat memory — `kind=entry` (per-turn) AND `kind=synthesis` (cross-turn) | Chat memory retrieval |

---

## 3. Yêu cầu hệ thống

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Docker + Docker Compose
- Ít nhất một LLM provider: Ollama (local/container), OpenAI, Gemini, HuggingFace Inference

---

## 4. Cài đặt & Khởi động

### Quick start (5 lệnh)

```bash
# 1. Clone + cd
git clone <repo> && cd AgentRag

# 2. One-shot install: docker compose up + uv sync + npm install + alembic migrate
make install

# 3. (S5) Seed medical ontology + backfill ES tags — chỉ cần lần đầu
make seed-ontology

# 4. Review .env (set OPENAI_API_KEY / GEMINI_API_KEY hoặc bật Ollama)
$EDITOR .env

# 5. Run everything
make dev                  # api + worker + frontend foreground (Ctrl+C all)
```

| URL | Mô tả |
|---|---|
| http://localhost:3000 | Next.js UI |
| http://localhost:3000/cost | **S1 — Cost & token dashboard** |
| http://localhost:3000/notebooks | Notebook chat (Trace button per AI turn — **S2**) |
| http://localhost:8000 | FastAPI root |
| http://localhost:8000/docs | Swagger UI |
| http://localhost:8000/admin | Admin reasoning panel |

### Verify install

```bash
make health                                          # /config/validate + /on/api/auth/status
make test-fast                                       # tests độc lập với Postgres
curl http://localhost:8000/on/api/ontology/systems   # S5 — 15 medical systems
curl http://localhost:8000/on/api/metrics/cost       # S1 — ledger
```

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
| `make seed-ontology` | **S5** — `scripts/seed_ontology.py` + `backfill_tags`. Idempotent |
| `make backfill-tags` | **S5** — re-tag ES segments only (no seed) |
| `make backfill-tags-dry` | Preview backfill changes |

#### Docker infra

| Target | Mô tả |
|---|---|
| `make docker-up` | Start postgres + elasticsearch + valkey (default services) |
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
| `make test` | `pytest -q` (full — cần Postgres + ES) |
| `make test-fast` | `pytest` minus `tests/ontology` + `tests/ingestion` (no Postgres) |
| `make bench-ingest` | Benchmark ingest pipeline |
| `make cost-reset` | **S1** — clear in-memory LLM cost ledger |
| `make dashboard-open` | **S1** — open `/cost` dashboard in browser |
| `make clean` | Xoá `.cache`, `__pycache__`, `.next` |
| `make deepclean` | `clean` + `node_modules` + `.venv` |

#### Reset (3 mức độ)

| Mức | Target | Xoá những gì | Giữ lại |
|---|---|---|---|
| 🟡 **Soft** | `make reset` | DB volumes (postgres + ES + valkey + ollama), `.cache/agentrag`. Restart infra + migrate. | Code, deps, `.venv`, `node_modules`, ảnh đã extract, logs |
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

### Tier presets (one-shot)

Tier-specific overlays sống tại [`presets/`](../presets/). Áp dụng:

```bash
make use-preset TIER=3a       # tier-1 | tier-2 | tier-3a | tier-3b | tier-4 | tier-5
# Backup .env hiện tại tự động (.env.bak-YYYYMMDD-HHMMSS)
```

| Tier | Phù hợp | File |
|---|---|---|
| 1 | CPU only, RAM ≥ 16 GB | `presets/tier-1.env` |
| 2 | GPU 6–8 GB VRAM | `presets/tier-2.env` |
| 3a | GPU 16 GB (full feature, qwen 7B + vision) — *recommended laptop/workstation* | `presets/tier-3a.env` |
| 3b | GPU 24 GB (qwen 14B/32B) | `presets/tier-3b.env` |
| 4 | Cloud API (OpenAI/Gemini) | `presets/tier-4.env` |
| 5 | 6 GB VRAM laptop + Gemini cloud | `presets/tier-5.env` |

Tier 3a cũng cần `OLLAMA_MAX_LOADED_MODELS=3` trong `docker-compose.yml` để
giữ 3 LLM hot cùng lúc. Sau khi `make use-preset`, append secrets vào `.env`
(`OPENAI_API_KEY`, `GEMINI_API_KEY`, …).

### 5.1 API Keys

```env
OPENAI_API_KEY=
GEMINI_API_KEY=
HF_TOKEN=
DEEPSEEK_API_KEY=                     # DeepSeek (OpenAI-compatible); falls back to OPENAI_API_KEY
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1/
```

### 5.1a Embedding serving (local Ollama / TEI / cloud)

```env
EMBEDDING_PROVIDER=ollama             # ollama | openai | gemini | hf_inference
EMBEDDING_MODEL=nomic-embed-text      # use bge-m3 via TEI (below) for best VN quality
EMBEDDING_BASE_URL=                   # blank = OLLAMA_BASE_URL
```

| Mode | Config | Notes |
|---|---|---|
| **Ollama (default)** | `EMBEDDING_PROVIDER=ollama`, `EMBEDDING_MODEL=nomic-embed-text` | Free local, 768-dim, stable. ⚠ Ollama's `bge-m3` GGUF can emit NaN at batch scale — serve bge-m3 via TEI instead. |
| **TEI — bge-m3 (recommended quality)** | `make serve-embed` → `EMBEDDING_PROVIDER=openai`, `EMBEDDING_MODEL=BAAI/bge-m3`, `EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1/` | Local GPU server, 1024-dim, strong VN. |
| **Cloud** | `EMBEDDING_PROVIDER=gemini` (or `hf_inference`) + key | No local GPU. |

**TEI (Text Embeddings Inference) for bge-m3:**

```bash
make serve-embed        # docker compose -f deploy/tei.compose.yml --profile gpu up -d  → :8080
make stop-embed
```

- Serves `BAAI/bge-m3` from `models/bge-m3`. Pre-download once:
  `uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='models/bge-m3')"`
- **Blackwell GPUs (RTX 50xx, compute cap sm_120):** pinned TEI images are sm_80-only and fail with `Runtime compute cap 120 is not compatible`. The compose uses `text-embeddings-inference:cuda-latest` (all-arch PTX) — required for sm_120, fine on older GPUs.
- TEI ignores the bearer key, so `OPENAI_API_KEY` holding a DeepSeek key is harmless.
- **Switching embed model changes the vector dimension** (bge-m3 1024 vs nomic 768) → delete + re-ingest the index.

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
CHAT_STRUCTMEM_ENABLED=true          # default; replaces sliding-window history (false → fall back to last-N turns)
CHAT_MEMORY_CONSOLIDATION_THRESHOLD=10   # số turns trước khi consolidate
CHAT_MEMORY_TOP_K=8
```

### 5.5 Retrieval & Reranking

```env
RETRIEVAL_TOP_K=10
RETRIEVAL_NUM_CANDIDATES=50
RETRIEVAL_RRF_K=60
RETRIEVAL_RERANK_ENABLED=false
RETRIEVAL_RERANK_BACKEND=local_cross_encoder   # llm_chat | local_cross_encoder
RETRIEVAL_RERANK_MODEL=dengcao/bge-reranker-v2-m3
```

Rerank backends:

| Backend | Model | Deps | Cost | Notes |
|---|---|---|---|---|
| `local_cross_encoder` | `dengcao/bge-reranker-v2-m3` | bundled (`sentence-transformers`) | free | **Default.** Local CPU/GPU, no API. Lifts relevant chunks to the top. |
| `llm_chat` | any chat model | none | varies | Rank via a chat LLM (slow); fallback when no cross-encoder available. |

#### RAG enhancement (2026-06)

Năm workstream nâng cao chất lượng/độ trễ retrieval. **Tất cả mặc định OFF** —
bật từng cờ để A/B. Thiết kế chi tiết / Design specs (2026-06):
`docs/superpowers/specs/2026-06-10-rag-enhancement-design.md` (RAG backend) +
`docs/superpowers/specs/2026-06-10-ui-enhancement-design.md` (UI signal surfacing:
MessageSignals chips, TraceDialog, RAPTOR/contextual citation hovers, cost charts).

- **Contextual Retrieval** (`CONTEXTUAL_RETRIEVAL_ENABLED`) — mỗi chunk được LLM
  thêm câu ngữ cảnh trước khi embed/BM25 (cite vẫn dùng `content` gốc). Cần
  re-ingest; route task `contextualize` sang DeepSeek qua `LLM_TASK_MODEL_MAP`
  để tận dụng doc-prefix cache rẻ.
- **RAPTOR** (`RAPTOR_ENABLED`) — lớp tóm tắt đa tầng (cluster + summarize đệ
  quy), node `node_level>=1` nằm trong cùng index. Cần re-ingest.
- **CRAG critique** (`CRAG_ENABLED`) — node kiểm tra grounding/relevance sau
  answer, re-retrieve sửa lỗi có giới hạn (`AGENT_CRITIQUE_MAX_RETRIES`); kèm
  multi-hop chaining (`AGENT_MULTIHOP_ENABLED`).
- **Adaptive routing** (`ADAPTIVE_ROUTING_ENABLED`) — câu hỏi đơn giản, một
  domain, độ tự tin cao → fast-path bỏ qua vòng lặp agent (giảm latency).
- **Semantic cache** (`SEMANTIC_CACHE_ENABLED`) — cache kết quả retrieval theo
  độ tương đồng embedding (chỉ cho truy vấn không filter).

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

Task keys: `classify`, `decide`, `schema_discovery`, `sql_compile`, `synthesize`, `answer`, `mindmap`, `summary`, `transformation`, `domain_router` (S5), `followup` (S7).

> **DeepSeek — local + cloud cost mix.** DeepSeek is OpenAI-compatible and
> auto-routes by **model-name prefix**: any `deepseek-*` value in
> `LLM_TASK_MODEL_MAP` is sent to `https://api.deepseek.com` using
> `DEEPSEEK_API_KEY` (→ `OPENAI_API_KEY` fallback). Route cheap tasks to local
> Ollama and quality tasks to DeepSeek, e.g.:
>
> ```env
> LLM_ROUTING_ENABLED=true
> DEEPSEEK_API_KEY=sk-...
> LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","decide":"llama3.2:3b","domain_router":"llama3.2:3b","followup":"llama3.2:3b","schema_discovery":"deepseek-v4-flash","sql_compile":"deepseek-v4-flash","synthesize":"deepseek-v4-pro","answer":"deepseek-v4-pro"}
> ```
>
> Prefix → provider: `gemini-`/`gemma-` → Gemini, `gpt-`/`o1`/`o3` → OpenAI,
> `deepseek*` → DeepSeek, otherwise the default `AGENT_PROVIDER` (Ollama).

> **Speed tuning (Phase C).** The default `LLM_TASK_MODEL_MAP` routes
> the fast tasks (`decide` / `classify` / `domain_router` / `followup`)
> to a cheap small model (`llama3.2:3b`) and reserves the finetuned
> `qwen-agentrag` (or its fallback `qwen2.5:7b-instruct`) for the
> `answer` step only. Pair this with `AGENT_MAX_STEPS=2` and
> `AGENT_PLAN_TRIGGER_MIN_CHARS=120` to cut the decide-loop overhead
> from ~30s → ~10s on local Ollama. Pull both tags up front:
>
> ```bash
> docker exec agentrag-ollama ollama pull llama3.2:3b
> docker exec agentrag-ollama ollama pull qwen2.5:7b-instruct
> ```

When cost tracking is on, every LLM call is logged in a process-local ring
buffer (last 5000 calls) with estimated USD via public Gemini / OpenAI pricing.

- `GET /on/api/metrics/cost` — per-task + per-model breakdown (`calls`, `in_tokens`, `out_tokens`, `usd`, `avg_latency_ms`)
- `POST /on/api/metrics/cost/reset` — clear ledger

### 5.6c Langfuse tracing (optional)

Full LLM trace UI (prompts, latency, token usage per call) alongside the cost
ledger. Every OpenAI client is built through one factory
(`common/langfuse_client.py::make_async_openai`), so flipping `LANGFUSE_ENABLED`
on traces all agent / vision / reranker calls via the `langfuse.openai` drop-in —
no per-call instrumentation.

```bash
uv sync --extra observability
docker compose --profile observability up -d     # self-hosted Langfuse + its Postgres
# open http://localhost:3000 → create project → copy keys into .env
```

```env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3002
```

Disabled by default → plain `openai.AsyncOpenAI`, zero overhead. The cost ledger
(§5.6) stays independent; Langfuse adds the trace/latency UI.

### 5.6d RAGAS answer-quality eval (optional, two-step)

Score the agent against a golden dataset with RAGAS (faithfulness,
context_precision, context_recall; optional answer_relevancy).

**Why two steps:** RAGAS hard-requires `langchain-core <0.3`, which conflicts
with this project's `langchain-core 1.x` (pulled by `langgraph`) — they cannot
share one venv. So the agent dumps eval rows in the app venv, and RAGAS scores
them in an isolated env.

```bash
# once, if no golden set
python scripts/eval/generate_dataset.py achievement-system

# STEP 1 (app venv): run agent → rows JSON. No extra deps.
python scripts/eval/run_ragas.py achievement-system --limit 5
#   → data/eval/achievement-system_ragas_rows.json

# STEP 2 (isolated venv): score with a dedicated Gemini judge
GEMINI_API_KEY=$GEMINI_API_KEY uv run --no-project \
    --with "ragas>=0.2,<0.3" --with "langchain-openai<0.3" \
    python scripts/eval/score_ragas.py data/eval/achievement-system_ragas_rows.json
#   → console table + ..._ragas_rows.scored.json
```

Step 1 reads the agent's `context` field (retrieved passages) from `chat()` as
RAGAS `retrieved_contexts`. The judge is decoupled from the agent runtime
(`--judge-model`, default `gemini-2.5-flash`), so a slow local agent is still
graded by a fast cloud judge.

Default metrics are LLM-only: **faithfulness, context_precision,
context_recall**. Add `--with-relevancy` to also run `answer_relevancy` — but
that needs embeddings, and Gemini's OpenAI-compat `/embeddings` returns
`501 UNIMPLEMENTED`, so pass `--embedding-provider openai` (OpenAI embeddings)
for it. Failed metrics are reported `FAILED` and omitted from the JSON.

Verified e2e on a 5-question hypertension dataset (real agent answers):
`faithfulness 1.00 · context_precision 0.85 · context_recall 1.00`.

### 5.6e Full RAG benchmark — 9 metrics (DeepEval + HF datasets)

End-to-end benchmark that runs **our** pipeline (ingest gold docs → retrieve →
agent answer) over public RAG datasets and gates against targets. Unlike RAGAS,
DeepEval has no langchain conflict — installs in the app venv.

```bash
uv sync --extra deepeval
LLM_COST_TRACKING_ENABLED=true ELASTICSEARCH_INDEX_NAME=agentrag_bench \
  python scripts/eval/run_benchmark.py --suite vn --n 30      # or --suite en | both
```

Datasets: VN `sailor2/Vietnamese_RAG` (BKAI_RAG, LegalRAG) · EN `galileo-ai/ragbench`
(covidqa, pubmedqa — medical). Judge = Gemini 2.5 Flash (`--judge-provider deepseek|openai` to swap).

| # | Metric | Source | Target |
|---|--------|--------|--------|
| 1 | Retrieval recall@k | `ContextualRecallMetric` | ≥ 0.70 |
| 2 | Context precision | `ContextualPrecisionMetric` | ≥ 0.70 |
| 3 | Faithfulness | `FaithfulnessMetric` | ≥ 0.80 |
| 4 | Answer correctness | `GEval` vs expected | ≥ 0.70 |
| 5 | Citation accuracy | `GEval` vs context | ≥ 0.70 |
| 6 | Latency p50/p95/p99 | per-question wall time | report |
| 7 | Cost / query | LLM ledger USD ÷ N | report |
| 8 | Failure rate | empty / exception | < 5% |
| 9 | Freshness | `run_freshness_check` | pass |

Use an isolated `ELASTICSEARCH_INDEX_NAME` so the benchmark corpus doesn't mix
with your real index. Report → `data/eval/benchmark_<suite>.json`.

> **Note — citation_accuracy:** the GEval citation metric expects inline `[N]`
> markers in the answer. The agent currently returns citations as a structured
> field (not inline), so this metric reads low until inline citation markers are
> added to the answer prompt. Treat it as a known gap, not a regression.

### 5.6f Phoenix tracing (optional)

Arize Phoenix — local OTEL trace + eval UI, runs alongside Langfuse. OpenInference
auto-instruments the OpenAI client, so every LLM call is traced with no per-call code.

```bash
uv sync --extra phoenix
docker compose --profile observability up -d phoenix    # UI http://localhost:6006
```

```env
PHOENIX_ENABLED=true
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
```

### 5.6b Agent Harness (Context + Plan + Critique)

```env
# Token-aware context budget (replaces chunk-count cap when >0)
AGENT_MAX_CONTEXT_TOKENS=6000
# Lost-in-the-middle reorder: best chunks at start + end of packed context
AGENT_LOST_IN_MIDDLE_REORDER=true

# Plan-then-execute: planner decomposes multi-hop questions into sub-queries,
# parallel retrieval, then single answer pass. Skipped for short questions.
AGENT_PLAN_THEN_EXECUTE_ENABLED=true
AGENT_PLAN_TRIGGER_MIN_CHARS=60
AGENT_PLAN_MAX_SUBQUERIES=4
```

**Orchestrator** — single backend: `GraphAgentService` (LangGraph `StateGraph`,
13 nodes: validate → memory → chitchat_check → classify → structured/semantic →
plan → bootstrap → decide ⇄ tool_exec → assemble → answer → ground) with
`InMemorySaver` checkpoint. Each turn's state is persisted by
`thread_id = conversation_id` → resume from any node, inspect state via
`_GRAPH.aget_state(config)`. (Self-critique pass from the deleted hand-rolled
loop is not currently ported — re-introduce as a graph node if needed.)

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

### 5.8 PDF, OCR & MinerU

Two parser backends — pick via `PDF_PARSER_BACKEND`.

| Backend | Tiers | When |
|---|---|---|
| `hybrid` (default) | PyMuPDF text-layer → Tesseract OCR → Vision LLM fallback (per page, escalates only when thin) | No GPU / no MinerU install. Vietnamese ⇒ set `PDF_OCR_LANG=vie+eng` |
| `mineru` | PyMuPDF text-layer → **MinerU** single-pass (layout + OCR + formula→LaTeX + table→HTML) — replaces Tesseract + vision tiers | GPU available, want formulas/tables/layout preserved, fewer LLM calls |

```env
PDF_PARSER_BACKEND=hybrid        # hybrid | mineru

# Tesseract path (hybrid backend only)
PDF_OCR_LANG=vie+eng
PDF_OCR_MIN_TEXT_CHARS=50
PDF_OCR_DPI=300
PDF_OCR_VISION_FALLBACK=true     # tier-3 escalation when Tesseract output still thin
PDF_OCR_VISION_THRESHOLD=30

# MinerU (mineru backend). Auto-picks GPU when local engine selected.
MINERU_BACKEND=vlm-auto-engine   # pipeline | vlm-auto-engine (default) | hybrid-auto-engine | *-http-client
MINERU_OUTPUT_DIR=.cache/agentrag/mineru
MINERU_LANG=latin                # Vietnamese uses Latin script → 'latin'
                                 # Allowed: ch | en | korean | japan | chinese_cht | latin | arabic | east_slavic | cyrillic | devanagari
MINERU_DEVICE=cuda               # legacy, ignored by new CLI (device chosen via backend)

# Route PPTX → libreoffice → PDF → MinerU (preserves slide layout + formulas).
# Falls back to MarkItDown when off or libreoffice/mineru missing.
INGEST_USE_MINERU_FOR_PPTX=false
```

**Picking `MINERU_BACKEND`:**

| Value | Notes |
|---|---|
| `pipeline` | Classic; lightweight; no VLM; runs CPU-OK |
| `vlm-auto-engine` | **Default.** Qwen2-VL multilingual; preserves Vietnamese diacritics natively (no OCR fallback to `latin`). Needs GPU |
| `hybrid-auto-engine` | Faster than `vlm-auto-engine` but uses paddleocr `latin` for some pages → **drops Vietnamese accents**. Pick only if corpus is English/CJK and latency matters more than accents |
| `*-http-client` | Remote OpenAI-compatible VLM (set `MINERU_URL`) |

**Install MinerU:**

```bash
pip install -U mineru          # CPU only
pip install -U "mineru[all]"   # + GPU + table-rec + formula models
mineru --version               # confirm CLI on PATH
```

Models download lazily on first run (~3-5 GB). Subsequent calls cache to `MINERU_OUTPUT_DIR`.

`INGEST_USE_MINERU_FOR_PPTX=true` also requires `libreoffice` on PATH (`apt install libreoffice-impress`).

```env
# Vision LLM (used by hybrid backend tier-3 + standalone image ingest).
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

Image-heavy PDFs (e.g. 100-page scanned thesis):
- `PDF_PARSER_BACKEND=mineru` → one MinerU pass extracts text + layout + tables + formulas; no vision-LLM tier needed.
- `PDF_PARSER_BACKEND=hybrid` → `vision_extract` ARQ job describes each thin page with the vision LLM, embeds, then upserts segments to Postgres + Elasticsearch in batches. ES `docs.count` climbs every batch instead of one bulk write at the end.

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

Auth: `Authorization: Bearer ${OPEN_NOTEBOOK_PASSWORD}`. Xem [adapter README](../src/agentrag/adapter/README.md).

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
  ├──▶ index vào agentrag_memory_doc (kind=synthesis)
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
          ├── embed + index → agentrag_memory_chat (kind=entry)
          └── [if count ≥ threshold] consolidate() → agentrag_memory_chat (kind=synthesis)

Next question
  └──▶ ChatMemoryService.retrieve(conversation_id, question)
          ├── KNN search trên agentrag_memory_chat (filter kind=entry)
          ├── KNN search trên agentrag_memory_chat (filter kind=synthesis)
          └── inject conversation_memory vào _decide() + _answer() prompts
```

### Khi nào nên bật

- Conversation dài (> 10 turns)
- Cần nhớ thông tin cụ thể từ nhiều turns trước
- Sliding-window history không đủ context

---

## 10. Background Workers & Auto-scaler

Jobs chạy nền qua **ARQ** (Valkey/Redis-backed task queue, RESP protocol) — survive process restart, scalable.

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

Bất kể backend (`hybrid` hay `mineru`), `PDFParser` chèn marker page trong text khi parse; mỗi chunk được tag `page_start` / `page_end`:

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

Xem [generation README](../src/agentrag/generation/README.md).

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

### S1 — Cost & Token Dashboard (`/cost`)

Frontend page tổng hợp LLM ledger. Auto-refresh 5s.

| Khu vực | Nội dung |
|---|---|
| Summary cards | Total calls / input tokens / output tokens / estimated USD |
| Per-task tab | calls, in tok, out tok, avg ms, **p50, p95**, USD |
| Per-model tab | giống per-task nhưng group by model |
| Recent tab | Newest-first list (20/50/100/200 toggle) — time, task, model, latency, tokens, USD, source |
| Reset button | `POST /on/api/metrics/cost/reset` |

Cần bật `LLM_COST_TRACKING_ENABLED=true` trong `.env`.

### S2 — Reasoning Trace per AI bubble

Mỗi assistant message trong notebook chat hiện nút **"Trace"** (nếu có
`tool_trace` / `timings_ms`). Click mở dialog gồm:

- Pipeline graph: `plan → decide → tool → assemble → answer → critique`
  với latency mỗi stage
- Sub-queries từ planner
- Generated SQL (structured path)
- Tool calls list (expandable): input + truncated output JSON
- Citations list

Trace lấy từ `ChatMessage` model — `reasoning_path`, `timings_ms`,
`tool_trace`, `plan_subqueries`, `sql_query` đều persist trên assistant turn.

### S5 — Domain Filter dropdown

Trên ChatPanel có nút **"Lĩnh vực"** (Filter icon) — popover chọn
`Hệ cơ quan` × `Chuyên khoa`. Khi user pick, request gắn
`domain_filter: {system, specialties}` → backend bỏ qua DomainRouter, lọc
trực tiếp. Để trống → router chạy bình thường.

Xem [adapter README](../src/agentrag/adapter/README.md).

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

Implementation: `src/agentrag/adapter/rate_limit.py` + `upload_dedupe.py` (Valkey INCR + EXPIRE 60s).

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

### Full RAG benchmark (9 metrics)

See §5.6e. Quick run (needs `uv sync --extra deepeval`, live ES/PG/Valkey, an isolated index):

```bash
ELASTICSEARCH_INDEX_NAME=agentrag_benchmark LLM_COST_TRACKING_ENABLED=true \
  uv run python scripts/eval/run_benchmark.py --suite both --n 30 \
  --judge-provider deepseek --judge-model deepseek-v4-flash   # or --judge-provider gemini
# → data/eval/benchmark_<suite>.json
```

### Frontend e2e (Playwright)

Drives the real UI end-to-end: login → every route → notebook chat (streamed
reply) → source ingest → retrieval-grounded answer.

```bash
# Prereqs: API (:8000) + frontend running, and a test user (sign up in the UI
# or POST /on/api/auth/signup). First time only: install browsers.
cd frontend
npm i -D @playwright/test && npx playwright install chromium
npm run e2e            # = playwright test
```

- `e2e/auth.setup.ts` logs in once → reuses the session (storageState).
- `e2e/full-ui.spec.ts` smokes all routes; `e2e/deep.spec.ts` drives chat + RAG.
- The login credentials + `baseURL` (`:3001` in config) are set for the local dev
  stack — adjust to your ports/user.
- ⚠ The burst of page loads + activity/cost polling can trip
  `RATE_LIMIT_PER_MIN_DEFAULT` (429 → logout); lower the test burst or raise the
  limit for the test session to get a clean pass.

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
| `ARQ pool not initialized` | Chạy app trước khi Valkey sẵn sàng | Đảm bảo Valkey đang chạy |
| `unsupported value: NaN` | Embedding không ổn định | Đổi sang `nomic-embed-text` |
| Structured path luôn fallback | Model quá nhỏ | Dùng model ≥7B |
| `agentrag_memory_doc` rỗng (kind=entry) | `STRUCTMEM_ENABLED=false` hoặc worker chưa chạy | Chạy `arq worker` hoặc `python scaler.py` |

---

## 21. Cấu trúc thư mục

```
AgentRag/
├── main.py                              # FastAPI app + lifespan (ARQ pool)
├── cli.py                               # CLI entry point
├── scaler.py                            # ARQ worker auto-scaler
├── docker-compose.yml                   # PostgreSQL, Elasticsearch, Valkey, Ollama
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
    │   ├── history.py                   # ConversationStore (Valkey + PG)
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
    │   │   ├── pdf_parser.py            # PyMuPDF — page-aware; dispatches to Tesseract/Vision/MinerU
    │   │   ├── mineru_parser.py         # MinerU CLI shim (layout + OCR + formula + table, one pass)
    │   │   ├── pptx_via_mineru.py       # PPTX → libreoffice PDF → MinerU (opt-in)
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

Architecture overview: [`ARCHITECTURE.md`](../ARCHITECTURE.md) (S4 plane split).

| Module | Plane | README |
|---|---|---|
| Services (Execution Plane) | E | [src/agentrag/services/README.md](../src/agentrag/services/README.md) |
| Ontology / Domain partition (S5) | mixed | [src/agentrag/ontology/README.md](../src/agentrag/ontology/README.md) |
| Agent (Semantic Loop) | R | [src/agentrag/agent/README.md](../src/agentrag/agent/README.md) |
| Structured SQL Reasoning | R | [src/agentrag/structured/README.md](../src/agentrag/structured/README.md) |
| Retrieval | E | [src/agentrag/retrieval/README.md](../src/agentrag/retrieval/README.md) |
| Ingestion Pipeline | E | [src/agentrag/ingestion/README.md](../src/agentrag/ingestion/README.md) |
| StructMem (doc) | E | [src/agentrag/graph/README.md](../src/agentrag/graph/README.md) |
| Chat & Chat StructMem | mixed | [src/agentrag/chat/README.md](../src/agentrag/chat/README.md) |
| Generation (Mindmap/Summary) | R | [src/agentrag/generation/README.md](../src/agentrag/generation/README.md) |
| Open-Notebook Adapter & Admin | — | [src/agentrag/adapter/README.md](../src/agentrag/adapter/README.md) |
| Background Worker (ARQ jobs) | E | [src/agentrag/worker/README.md](../src/agentrag/worker/README.md) |
| CLI | — | [src/agentrag/cli/README.md](../src/agentrag/cli/README.md) |
| MCP Server | — | [src/agentrag/mcp/README.md](../src/agentrag/mcp/README.md) |
| Common Utilities | — | [src/agentrag/common/README.md](../src/agentrag/common/README.md) |
| Domain Router (orchestration) | R | [src/agentrag/orchestration/README.md](../src/agentrag/orchestration/README.md) |
| Database (engine + ORM models) | E | [src/agentrag/database/README.md](../src/agentrag/database/README.md) |
| Health diagnostics | — | [src/agentrag/health/README.md](../src/agentrag/health/README.md) |
| Observability (cost ledger + event log) | E | [src/agentrag/observability/README.md](../src/agentrag/observability/README.md) |
| Eval (offline RAG quality harness) | — | [src/agentrag/eval/README.md](../src/agentrag/eval/README.md) |

R = Reasoning Plane, E = Execution Plane.

### Tag map (sub-projects)

| Tag | Sub-project |
|---|---|
| `s5-complete` | Medical KB domain partition (ontology + federated retrieval + UI override) |
| `s4-complete` | Reasoning / Execution Plane split + ServiceContainer + Protocols |
| `s1-complete` | LLM cost & token dashboard |
| `s2-complete` | Per-turn LangGraph-style reasoning trace UI |
| `s3-complete` | Embedding cache + p50/p95 latency surface |
