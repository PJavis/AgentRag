# AgentRag

Nền tảng RAG cho học liệu y khoa Việt Nam. Một luồng suy luận semantic
(hybrid BM25 + kNN + RRF + StructMem KG), bộ nhớ phân cấp (StructMem),
domain-aware retrieval, reasoning trace + cost dashboard, UI Next.js
tương thích open-notebook.

> 📖 **Tài liệu vận hành đầy đủ** (cài đặt chi tiết, mọi flag `.env`, API
> reference, CLI, benchmark, reset…): **[`docs/README-full.md`](./docs/README-full.md)**.
> Kiến trúc: [`ARCHITECTURE.md`](./ARCHITECTURE.md). Mỗi module có README riêng
> (xem [bên dưới](#module-docs)).
>
> 🚀 **Chạy ở nhà (app + fine-tune):** [`docs/HOME-RUN.md`](./docs/HOME-RUN.md) ·
> 🛠️ **Triển khai/vận hành:** [`docs/DEPLOY-RUNBOOK.md`](./docs/DEPLOY-RUNBOOK.md) ·
> 📝 **Changelog:** [`docs/CHANGELOG.md`](./docs/CHANGELOG.md) ·
> 🔐 **AuthZ audit:** [`docs/security/authz-audit-2026-06-25.md`](./docs/security/authz-audit-2026-06-25.md)

---

## Tính năng chính

- **Reasoning / Execution Plane** — kiến trúc phân tầng, `ServiceContainer` DI;
  Reasoning lấy service qua Protocol.
- **Semantic hybrid retrieval** — BM25 + kNN + RRF + StructMem KG, rerank
  cross-encoder, abstain-on-thin-context khi thiếu căn cứ.
- **Domain partition** — KB chia theo `hệ cơ quan × chuyên khoa` (15×14), shared
  ontology + `pg_trgm` fuzzy + `DomainRouter` SLM.
- **StructMem** — bộ nhớ tri thức (factual + relational + synthesis) cho document
  và hội thoại, thay sliding-window.
- **Page-aware citations · Vision LLM · Mindmap & Summary** —
  trích dẫn đúng trang, mô tả ảnh y tế, layout/OCR/formula, mermaid mindmap.
- **Cost dashboard · Reasoning trace** — `/cost` page (p50/p95, recent calls);
  nút Trace → node graph `plan → decide → tool → assemble → answer → critique`.
- **RAG enhancement (2026-06, mặc định OFF)** — Contextual Retrieval · RAPTOR
  summary layer · CRAG critique + multi-hop · adaptive fast-path · semantic cache.
  Bật từng cờ để A/B; UI tự lộ tín hiệu (chips + trace + citation hover + charts).
  Chi tiết: [`docs/README-full.md` §5.5](./docs/README-full.md) · thiết kế:
  [`docs/superpowers/specs/2026-06-10-rag-enhancement-design.md`](./docs/superpowers/specs/2026-06-10-rag-enhancement-design.md),
  [`…ui-enhancement-design.md`](./docs/superpowers/specs/2026-06-10-ui-enhancement-design.md).
  Hướng dẫn test: [`docs/TEST-GUIDE-2026-06-10.md`](./docs/TEST-GUIDE-2026-06-10.md).

## Kiến trúc (tóm tắt)

Hai mặt phẳng, nối qua `ServiceContainer` singleton:
- **Reasoning Plane** — quyết định: semantic agent loop → answer + grounding.
- **Execution Plane** — IO: LLM gateway, embedding, retrieval (ES hybrid), storage,
  vision. Không branch theo prompt.

| Store | Vai trò |
|---|---|
| **PostgreSQL** | source of truth — documents, segments, conversations, notebooks/notes |
| **Elasticsearch** | hybrid search `agentrag_segments` + StructMem `agentrag_memory_doc` / `_chat` |
| **Valkey** (Redis-compat) | chat cache + ARQ queue + cost ledger stream |
| **Filesystem** | ảnh extract (`data/images/`, serve `/images/*`) |

Sơ đồ luồng `POST /chat` / `/ingest` đầy đủ: [`docs/README-full.md` §1](./docs/README-full.md), [`ARCHITECTURE.md`](./ARCHITECTURE.md).

## Quick start

```bash
git clone <repo> && cd AgentRag
make install        # docker compose up + uv sync + npm install + alembic migrate
make seed-ontology  # seed medical ontology + backfill ES tags (lần đầu)
$EDITOR .env        # set OPENAI_API_KEY / GEMINI_API_KEY hoặc bật Ollama
make dev            # api + worker + frontend (Ctrl+C dừng tất cả)
```

| URL | Mô tả |
|---|---|
| http://localhost:3000 | Next.js UI (notebook chat, Trace per AI turn) |
| http://localhost:3000/cost | Cost & token dashboard |
| http://localhost:8000/docs | FastAPI Swagger |
| http://localhost:8000/admin | Admin reasoning panel |

Day-to-day: `make up-bg` (chạy nền) · `make logs` · `make stop`.
Verify: `make health` · `make test-fast`. Toàn bộ target: `make help`.
Chi tiết cài đặt / production / `.env` / troubleshooting → [`docs/README-full.md`](./docs/README-full.md).

## Cấu trúc thư mục

```
src/agentrag/        backend (xem README mỗi module bên dưới)
frontend/            Next.js UI (React 19, Radix, Tailwind)
scripts/eval/        benchmark + ablation runners
docs/                README-full.md (manual đầy đủ) + specs/plans + eval reports
main.py · cli.py     FastAPI app · interactive CLI
Makefile             toàn bộ lệnh dev/ops
```

## <a id="module-docs"></a>Module docs

Mỗi module backend có README chi tiết (purpose · plane · key files · public
interface · data flow · config · gotchas):

| Module | Mô tả |
|---|---|
| [agent](./src/agentrag/agent/README.md) | LangGraph semantic loop (classify→plan→retrieve→answer→critique) |
| [retrieval](./src/agentrag/retrieval/README.md) | hybrid BM25+kNN+RRF + rerank + RAPTOR + semantic cache |
| [ingestion](./src/agentrag/ingestion/README.md) | parse→chunk→[contextualize]→embed→index + RAPTOR |
| [graph](./src/agentrag/graph/README.md) | StructMem extraction + consolidation + vision jobs |
| [services](./src/agentrag/services/README.md) | ServiceContainer DI + LLM/embedding/retrieval facades |
| [ontology](./src/agentrag/ontology/README.md) | medical taxonomy + TermResolver |
| [orchestration](./src/agentrag/orchestration/README.md) | DomainRouter (system/specialty routing) |
| [chat](./src/agentrag/chat/README.md) | conversation persistence + chat StructMem |
| [generation](./src/agentrag/generation/README.md) | mindmap + structured summary |
| [adapter](./src/agentrag/adapter/README.md) | open-notebook FastAPI edge (auth, routers, admin) |
| [mcp](./src/agentrag/mcp/README.md) | MCP tools (hybrid search) |
| [worker](./src/agentrag/worker/README.md) | ARQ background-job runtime |
| [database](./src/agentrag/database/README.md) | async SQLAlchemy engine + ORM models |
| [eval](./src/agentrag/eval/README.md) | offline RAG quality harness (DeepEval/RAGAS) |
| [observability](./src/agentrag/observability/README.md) | cost/latency ledger + activity log |
| [common](./src/agentrag/common/README.md) | tracing, progress pub/sub, access policy |
| [cli](./src/agentrag/cli/README.md) | interactive Typer/Rich CLI |
| [health](./src/agentrag/health/README.md) | `/health/providers` diagnostics |
