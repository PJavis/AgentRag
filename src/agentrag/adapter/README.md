# Module: `adapter` — Open-Notebook Compatible API + Admin Panel

**Vị trí:** `src/agentrag/adapter/`

Sub-application FastAPI mount tại `/on` của main app, expose API tương thích với [open-notebook](https://github.com/lfnovo/open-notebook) Next.js frontend. Cho phép dùng UI có sẵn (notebook, sources, chat, search) mà không cần sửa frontend. Kèm panel `/admin` để xem reasoning trace của agent (LangGraph-style).

---

## Files

| File | Mô tả |
|---|---|
| `app.py` | `adapter` FastAPI sub-app — mount tất cả routers + middleware |
| `auth.py` | `OpenNotebookAuthMiddleware` (Bearer password), `is_admin()` helper |
| `db.py` | `AdapterNotebook`, `AdapterNote`, `AdapterTransformation`, `AdapterSourceInsight`, `AdapterChatFeedback`, `adapter_notebook_sources` (SQLAlchemy) |
| `models.py` | Pydantic schemas khớp open-notebook contract |
| `admin.py` | Admin reasoning inspector (HTML + JSON API) — mount tại `/admin` của main app |
| `routers/notebooks.py` | CRUD notebooks + add/remove sources |
| `routers/sources.py` | Upload, list, delete documents (mapped sang `Document` table) |
| `routers/notes.py` | CRUD notes per notebook |
| `routers/chat.py` | Notebook chat + source-based chat (SSE streaming) |
| `routers/search.py` | Tìm kiếm sources/notes |
| `routers/config.py` | `/api/config`, `/api/auth/status` cho client bootstrap |
| `routers/stubs.py` | Endpoints không quan trọng (transformations, models, episodes...) trả stub |

---

## Mount points (`main.py`)

```python
app.mount("/on", adapter)                   # tất cả /on/api/* và /on/health
app.include_router(adapter_admin.router)    # /admin (HTML) + /admin/api/* (JSON)
```

Inside the adapter, stored images are also re-mounted under
`/on/api/images/*` (StaticFiles → `IMAGE_STORAGE_DIR`) so the frontend can
resolve markdown `<img src>` URLs through the same `/api/*` Next.js rewrite
it uses for the rest of the API.

---

## Auth model

Hai layer độc lập:

| Header | Bảo vệ | Khi nào set |
|---|---|---|
| `Authorization: Bearer ${OPEN_NOTEBOOK_PASSWORD}` | Tất cả `/on/api/*` (trừ public list) | User nhập password trong UI login form |
| `X-Admin-Token: ${ADAPTER_ADMIN_TOKEN}` | `/admin/api/*` | Admin nhập token trong `/admin` page |

**Public prefixes** (không cần Bearer): `/api/config`, `/api/auth/status`, `/api/auth/login`, `/api/auth/signup`, `/api/auth/google/*`, `/api/health`, `/api/images`, `/health`, `/docs`, `/admin`.

OPTIONS preflight luôn pass-through để CORS hoạt động.

> ⚠️ Vì sub-app mount tại `/on`, `request.url.path` mà middleware nhận là `/on/api/config` (không phải `/api/config`). `_is_public()` xử lý cả prefix-match, suffix-match, và mid-path-component để robust với mọi mount point.

---

## Database tables (auto-create on startup)

| Table | Vai trò | Cột chính |
|---|---|---|
| `adapter_notebooks` | Open-notebook notebook entity | `id`, `name`, `description`, `archived`, `created_at`, `updated_at` |
| `adapter_notes` | Notes thuộc notebook | `id`, `notebook_id`, `title`, `content`, `note_type` |
| `adapter_notebook_sources` | M-N notebook ↔ document | `notebook_id`, `document_id` |
| `adapter_transformations` | User-defined transformation prompts | `id`, `name`, `prompt`, `apply_default` |
| `adapter_source_insights` | LLM-generated insights per source | `id`, `source_id`, `insight_type`, `content` |
| `adapter_chat_feedback` | Thumbs-up/down ratings per assistant turn | `user_id`, `turn_id`, `rating ∈ {+1,-1}`, `question`, `answer`, `reasoning_path` |

`Document` (table gốc của AgentRag) được dùng làm "source" — không tạo table riêng. Frontend gửi `source_id` dạng `source:<uuid>`; `_parse_source_id()` strip prefix trước khi UUID parse.

---

## Chat flows

### Notebook chat (`POST /api/chat/execute`)

```
POST /on/api/chat/execute
  body: {session_id, message, context, model_override}
  │
  ├── ConversationStore.append_message(role=user)
  ├── AgentService.chat(question, document_title=None, chat_history)  ← full agent
  ├── ConversationStore.append_message(role=assistant, citations, tool_trace)
  └── return {session_id, messages}
```

Gọi full agent → có thể cross-document, dùng StructMem.

### Source-based chat (`POST /api/sources/{id}/chat/sessions/{sid}/messages`)

```
SSE stream:
  ├── yield {"type": "user", "content": ...}
  ├── _direct_rag()                              ← không qua agent
  │     ├── ElasticsearchRetriever.search(document_title=...)
  │     ├── client-side filter theo document_title
  │     └── LLMGateway.json_response → {answer, highlights}
  ├── yield {"type": "ai", "content": <word>}    ← word-by-word fake stream
  ├── yield {"type": "context", "sources": [...], "highlights": [...]}
  ├── [admin] yield {"type": "reasoning", "tool_trace": [...]}
  └── yield {"type": "complete"}
```

Dùng `_direct_rag` thay vì agent để **strict isolation** theo document — tránh leak context từ document khác qua graph memory.

---

## Model defaults — runtime override

`PUT /on/api/models/defaults` persists user-selected models to
`data/model_overrides.json` (key/value of `EMBEDDING_PROVIDER`,
`EMBEDDING_MODEL`, `EXTRACTION_*`, `AGENT_*`, `VISION_*`). Overrides are
re-applied to the in-memory `settings` singleton at every process startup
(see `src/agentrag/config_overrides.py`).

API process picks up changes immediately. ARQ worker picks up on its next
restart — restart worker after switching the extraction/agent model.

`GET /on/api/models/defaults` returns the currently effective defaults
(env values merged with overrides).

---

## Feedback collection

`POST /on/api/chat/feedback` — upsert per `(user_id, turn_id)`:

```json
{
  "turn_id": "<assistant message id>",
  "session_id": "<conversation id>",
  "rating": 1,                      // +1 or -1
  "question": "...",
  "answer": "...",
  "reasoning_path": "semantic|structured|chitchat",
  "comment": "optional"
}
```

Frontend wires this to thumbs-up/down buttons (`FeedbackButtons.tsx`)
rendered next to every assistant message. The accumulated ledger in
`adapter_chat_feedback` is the seed asset for future preference-pair
training (DPO) or prompt tuning.

---

## Admin reasoning panel

`GET /admin` → HTML inspector (vanilla JS). Hiển thị:
- Sidebar: list conversations
- Main: messages + LangGraph-style flow diagram + step details (input/output/duration)

API riêng:

| Endpoint | Trả về |
|---|---|
| `GET /admin/api/conversations` | `[{id, title, message_count, created_at}, ...]` |
| `GET /admin/api/conversations/{id}/trace` | `{title, messages, tool_traces}` |

JS detect base path từ `window.location` → hoạt động dù mount ở `/admin` hay `/on/admin`.

---

## Tương tác

| Module | Vai trò |
|---|---|
| `agent.AgentService` | Chat notebook (full agent loop) |
| `chat.ConversationStore` | Lưu sessions + messages (Redis + PG) |
| `database.Document` / `Segment` | Document entity dùng làm source |
| `ingestion.pipeline.ingest_folder` | Chạy khi user upload file qua `POST /api/sources` |
| `retrieval.ElasticsearchRetriever` | Direct RAG cho source chat |
| `services.LLMGateway` | Single LLM call cho `_direct_rag` |
| `main.py` | Mount adapter + admin |

---

## Config liên quan

| Key | Default | Mô tả |
|---|---|---|
| `OPEN_NOTEBOOK_PASSWORD` | `None` | Bearer password cho `/on/api/*`. `None` = no auth. |
| `ADAPTER_ADMIN_TOKEN` | `None` | Token cho `/admin/api/*`. `None` = admin disabled. |
| `ADAPTER_VERSION` | `"0.7.0"` | Trả về cho `/api/config` để client check update |

---

## Frontend setup

```bash
git clone https://github.com/lfnovo/open-notebook
cd open-notebook/frontend

# Trỏ Next.js về AgentRag adapter
cat > .env.local << 'EOF'
API_URL=http://localhost:8000/on
INTERNAL_API_URL=http://localhost:8000/on
EOF

npm install
npm run dev    # http://localhost:3000
```

Login với password = giá trị `OPEN_NOTEBOOK_PASSWORD` trong AgentRag `.env`.
