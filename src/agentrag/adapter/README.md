# adapter — Open-Notebook-compatible HTTP API + auth + admin trace viewer

## Mục đích / Purpose
This module is the **HTTP-facing edge** of AgentRag. It is a self-contained
FastAPI sub-application (`adapter`) that re-implements the
[open-notebook](https://github.com/lfnovo/open-notebook) Next.js API contract,
so the existing open-notebook frontend (notebooks, sources, chat, search,
insights) can drive AgentRag unchanged. On top of that contract it adds the
pieces AgentRag needs in production: per-user accounts (JWT + Google OAuth,
with a legacy shared password still accepted), per-user Redis rate limiting,
upload dedupe, an activity/usage feed, a thumbs-up/down feedback ledger, and a
LangGraph-style admin reasoning inspector. It owns no RAG logic of its own — it
maps each request onto AgentRag's `AgentService`, retrieval, ingestion, and
`ConversationStore`.

## Plane
**Infrastructure** (delivery/edge layer). It is the transport + auth + session
boundary: it parses HTTP, authenticates the actor, enforces quotas, persists
adapter-owned tables, then delegates the actual decisions to the Reasoning
Plane (`AgentService`, `DomainRouter`) and the work to the Execution Plane
(`ElasticsearchRetriever`, `LLMGateway`, `ingest_folder`, `ConversationStore`)
via `agent.factory.get_agent_service()` and `services.container.get_container()`.

## Key files
| File | Responsibility |
|---|---|
| `app.py` | Builds the `adapter` FastAPI sub-app: CORS → RateLimit → Auth middleware, includes every router under `/api`, mounts stored images at `/api/images`, exposes `/` and `/health`. |
| `auth.py` | `OpenNotebookAuthMiddleware` — resolves `Authorization: Bearer …` to an `AuthIdentity` (JWT first, legacy password fallback), attaches it to `request.state.auth_identity`. Public-prefix allowlist + `is_admin()` / `get_identity()` helpers. |
| `auth_service.py` | Auth domain logic: bcrypt password hashing, JWT issue/decode (`HS256`), `authenticate()`, signup/`create_user`, Google OAuth (`google_auth_url`/`google_exchange`/`upsert_google_user`), `ensure_user_row` self-heal, `ADMIN_EMAILS` auto-promotion. |
| `rate_limit.py` | `RateLimitMiddleware` — per-user (or per-IP) fixed-window counter in Redis. Two buckets: `upload` vs `default`. Fails **open** on any Redis error. |
| `db.py` | SQLAlchemy models auto-created on startup: `AdapterNotebook`, `AdapterNote`, `AdapterTransformation`, `AdapterSourceInsight`, `AdapterChatFeedback`, the `adapter_notebook_sources` M-N table, and `create_adapter_tables()`. |
| `models.py` | Pydantic request/response schemas matching the open-notebook contract (notebooks, notes, sources, chat, search, activity). |
| `upload_dedupe.py` | `hash_bytes()` + `find_existing_document()` — SHA-256 content-hash lookup to skip re-ingesting identical uploads. |
| `account_deletion.py` | `delete_user_data(user_id)` — P4 right-to-delete: wipes all Postgres rows owned by the user (documents, segments, conversations, messages, feedback, events, user row), then best-effort ES + image-file purge. Called by `DELETE /chat/account`. |
| `admin.py` | `/admin` HTML reasoning inspector (inline vanilla-JS SPA) + `/admin/api/conversations[/{id}/trace]` JSON, grouping `ConversationStore` messages into per-turn question→trace→answer blocks. |
| `routers/notebooks.py` | Notebook CRUD + add/remove source links (`adapter_notebooks` / `adapter_notebook_sources`). |
| `routers/sources.py` | Upload → background `ingest_folder`, list/get/update/delete sources (mapped to `Document`), original-file download, dedupe, per-user ingest semaphore, SSE ingest-progress stream. |
| `routers/notes.py` | Notes CRUD backed by `adapter_notes`. |
| `routers/chat.py` | The heart: notebook chat (`/chat/execute`, `/chat/execute-stream`, `/chat/regenerate`), source-isolated chat (`_direct_rag`), image chat, feedback upsert (`POST /chat/feedback`), account deletion (`DELETE /chat/account` → `account_deletion.delete_user_data`), chat starters, follow-up generation. |
| `routers/search.py` | `/search` (hybrid/sparse retrieval) + `/search/ask[/simple]` (agent answer, SSE). |
| `routers/insights.py` | Per-source LLM insights — `run_transformation()` over a source's full text, stored in `adapter_source_insights`; save-as-note. |
| `routers/transformations.py` | User-defined transformation prompts CRUD + execute (seeds defaults). |
| `routers/models.py` | Synthesizes a model picker list from settings; `GET/PUT /models/defaults` persists runtime model overrides via `config_overrides`. |
| `routers/activity.py` | S6 usage feed — SQL aggregation over the `event_log` table: personal (`/activity/*`) + admin-global (`/admin/activity/*`) summaries, heatmap, per-user breakdown. |
| `routers/ontology.py` | Static taxonomy lookups (`/ontology/systems`, `/ontology/specialties`) for the S5 domain-filter dropdowns. |
| `routers/config.py` | `/config`, `/settings`, `/languages`, `/health`, `/metrics/cost*` (LLM cost ledger). |
| `routers/auth.py` | `/auth/status|signup|login|me|logout` + Google `/auth/google/start|callback`. Issues JWTs as the bearer token. |
| `routers/stubs.py` | No-op stubs for open-notebook features AgentRag doesn't implement (credentials, podcasts, command jobs, embeddings rebuild, speaker profiles). |

## Public interface
The module is consumed **only** by the root app (`/home/nguyenquocdung/AgentRag/main.py`),
not imported by other backend modules. Mount wiring:

```python
# main.py
from src.agentrag.adapter.db import create_adapter_tables
from src.agentrag.adapter.app import adapter
from src.agentrag.adapter import admin as adapter_admin

await create_adapter_tables()          # startup — create adapter_* tables
app.include_router(adapter_admin.router)  # /admin (HTML) + /admin/api/* (JSON)
app.mount("/on", adapter)                 # all /on/api/* and /on/health
```

Everything in `routers/*` is registered under `/api` inside the sub-app, so the
public URLs are `/on/api/...` (e.g. `POST /on/api/chat/execute`). Images are
re-mounted twice: `/images` (root app) and `/on/api/images` (sub-app) → both
serve `settings.IMAGE_STORAGE_DIR`.

Inbound the adapter calls into:
- `agent.factory.get_agent_service()` → `AgentService.chat()` / `.chat_stream()`
- `services.container.get_container()` → `.llm` (LLMGateway), `.domain_router`
- `chat.history.ConversationStore` (sessions + messages, Redis + Postgres)
- `retrieval.elasticsearch_retriever.ElasticsearchRetriever`
- `ingestion.pipeline.ingest_folder`
- `agent.followups.generate_followups`, `agent.starters.generate_starters`
- `generation.summary_service.SummaryService`
- `observability.activity.record_event`, `observability.cost`

## Data flow

**Notebook chat** (`POST /on/api/chat/execute` and `/execute-stream`):
1. Look up the session in `ConversationStore`; read `notebook_id` from metadata.
2. `_resolve_document_hint()` — cheap ILIKE scan of the user message for
   filename hints (`lec10`, `chương 3`, `file …`) to pin a single `Document.title`.
3. Append the user message.
4. Auto domain routing: if `DOMAIN_FILTER_ENABLED` and the UI sent no
   `domain_filter`, call `container.domain_router.classify()` and forward the
   pick to retrieval via `domain_filter`.
5. Verbosity intent: `_is_verbose_followup()` / `_is_summary_request()` can force
   the agent's DETAILED path; whole-doc summaries route to
   `SummaryService.iter_sections()` (map-reduce over every page, streamed live).
6. Scope isolation: `retrieval.context.set_document_scope(...)` restricts
   retrieval to the notebook's (or ticked `source_ids`) documents —
   NotebookLM-style; prevents cross-notebook leakage.
7. `agent.chat()` / `agent.chat_stream()`; append the assistant turn with
   `citations`, `tool_trace`, `timings_ms`, `reasoning_path`, etc.; generate +
   persist `follow_ups`; emit a `chat_turn` activity event.

**Source-isolated chat** (`POST /on/api/sources/{id}/chat/sessions/{sid}/messages`):
Bypasses the agent and uses `_direct_rag()` — `ElasticsearchRetriever.search()`
(mode `hybrid_kg` when `STRUCTMEM_ENABLED` else `hybrid`) with strict
client-side filter to the source's `document_title`, numbered `[n]` context
blocks, then a single `LLMGateway.json_response` returning `{answer, highlights}`.
Streamed word-by-word over NDJSON; results cached 5 min in a `TTLCache`. Strict
per-document isolation is the whole point of this path.

**Upload** (`POST /on/api/sources`): returns a skeleton `Document`
(`graph_status="pending"`) immediately, persists the original under
`ORIGINALS_DIR`, then runs `ingest_folder` in a `BackgroundTask` under a
per-user `asyncio.Semaphore(2)`. SHA-256 dedupe links to the existing doc when
the bytes already exist. `GET /sources/progress/stream` (SSE, Redis pub/sub)
pushes ingest-stage transitions for live status chips.

## Config
Read directly by this module (real names from `src/agentrag/config.py`):

| Key | Default | Used by |
|---|---|---|
| `AUTH_ENABLED` | `True` | `auth.py` — `False` ⇒ anonymous identity, no token required |
| `AUTH_ALLOW_SIGNUP` | `True` | `routers/auth.py` signup gate |
| `OPEN_NOTEBOOK_PASSWORD` | `None` | legacy shared bearer password (still accepted) |
| `JWT_SECRET` | `None` | JWT signing (auto-derived from PG creds in dev if unset) |
| `JWT_TTL_DAYS` | `7` | token lifetime |
| `ADMIN_EMAILS` | `""` | comma-list auto-promoted to admin on signup/login |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` / `GOOGLE_REDIRECT_URI` | `None` | Google OAuth |
| `FRONTEND_URL` | `http://localhost:3000` | OAuth post-login redirect target |
| `ADAPTER_ADMIN_TOKEN` | `None` | `X-Admin-Token` for `/admin/api/*` and admin activity |
| `ADAPTER_VERSION` | `"0.7.0"` | reported to `/api/config` |
| `RATE_LIMIT_ENABLED` | `True` | `rate_limit.py` master switch |
| `RATE_LIMIT_PER_MIN_DEFAULT` | `120` | default bucket / user / min |
| `RATE_LIMIT_UPLOAD_PER_MIN` | `20` | upload bucket / user / min |
| `REDIS_URL` | `redis://…` | rate-limit counters + ingest-progress pub/sub |
| `IMAGE_STORAGE_DIR` | `data/images` | static image mount + chat-upload + per-doc image purge on delete |
| `ORIGINALS_DIR` | `data/originals` | original-file persist + `/sources/{id}/download` |
| `UPLOAD_MAX_BYTES` | `100 MB` | upload size cap |
| `UPLOAD_DEDUPE_BY_HASH` | `True` | content-hash dedupe on upload |
| `PUBLIC_BASE_URL` | (opt) | absolute URL for chat-uploaded images |
| `DOMAIN_FILTER_ENABLED` | `True` | auto domain routing in chat |
| `STRUCTMEM_ENABLED` | `True` | picks `hybrid_kg` vs `hybrid` in `_direct_rag` |
| `RETRIEVAL_RERANK_ENABLED` | `False` | rerank toggle passed to retriever |
| `AGENT_MAX_CONTEXT_CHUNKS` | `8` | context cap in `_direct_rag` |

## Recent additions (2026-06)
The recent RAG-enhancement workstreams (Contextual Retrieval, RAPTOR, CRAG
critique/multi-hop, adaptive fast-path routing, semantic retrieval cache) live
in `ingestion/`, `agent/`, `structured/`, and `services/` — **this module does
not gate them directly.** The adapter participates only as the UI-signal carrier
the orchestrator described:
- Notebook chat turns surface whatever the agent returns — `citations` (which
  may carry `node_level`/`context_text` from RAPTOR/Contextual), plus
  `reasoning_path`, `timings_ms` (incl. `critique` when CRAG ran),
  `plan_subqueries`, and `sql_query` — persisted on `extra_metadata` and mapped
  into `ChatMessage` (`models.py`) for the `/trace` dialog and admin inspector.
- `_direct_rag` (source chat) reads `STRUCTMEM_ENABLED` to switch between
  `hybrid_kg` and `hybrid`, and `RETRIEVAL_RERANK_ENABLED` for reranking.
- Independent of those flags, the adapter itself shipped: JWT + Google OAuth
  auth (replacing shared-password-only), Redis rate limiting, the S6 activity
  feed (`routers/activity.py`), thumbs feedback (`adapter_chat_feedback`),
  chat regenerate, image chat, NotebookLM-style notebook scope isolation, and
  live map-reduce summary streaming.

## Gotchas
- **Mount-prefix paths in middleware.** Because the sub-app is mounted at `/on`,
  `request.url.path` seen by the middlewares is `/on/api/config`, not
  `/api/config`. `auth._is_public()` matches by prefix **and** suffix **and**
  mid-path component to stay robust to the mount point. New public endpoints
  must be added to `_PUBLIC_PREFIXES`.
- **Middleware order is inverted.** Starlette runs later-added middleware first;
  `app.py` adds RateLimit **before** Auth so that Auth runs first and RateLimit
  can key on the resolved per-user identity. Don't reorder.
- **`OPTIONS` always passes** both Auth and RateLimit (CORS preflight).
- **Rate limiting fails open.** Any Redis error (or `RATE_LIMIT_ENABLED=False`,
  or no `REDIS_URL`) ⇒ requests pass unthrottled.
- **`db.py` vs `models.py`.** `db.py` = SQLAlchemy ORM (real tables);
  `models.py` = Pydantic API schemas. Don't confuse them.
- **Source-id prefix.** The frontend sends `source:<uuid>`; `_parse_source_id()`
  / `_parse_source_uuid()` strip the `source:` prefix before `uuid.UUID()`.
- **Two admin gates.** `/admin/api/*` accepts either `X-Admin-Token ==
  ADAPTER_ADMIN_TOKEN` **or** a JWT with `admin: true`. `is_admin()` checks both.
- **`ensure_user_row` self-heal.** A JWT outlives a `make reset-data` that wipes
  the `users` table; the middleware recreates the user row idempotently on the
  first authed request so FK constraints (`documents.user_id`,
  `conversations.user_id`) keep working.
- **Model-override persistence is process-startup, not live.** `PUT
  /models/defaults` writes overrides; they re-apply to the in-memory `settings`
  singleton at process start (`config_overrides`). The API picks up the change
  on its next restart; the ARQ worker on **its** next restart — restart the
  worker after switching the extraction/agent model.
- **Ingest is fire-and-forget.** `POST /sources` returns a `pending` skeleton
  before ingestion finishes; the real row is populated in place by the pipeline.
  Poll `/sources/{id}/status` or subscribe to `/sources/progress/stream`.
- **`create_adapter_tables()` uses `Base.metadata.create_all` only** — it never
  runs migrations. Column changes to `adapter_*` tables need an Alembic
  migration (see `migrations/versions/d7e2a4b9c1f0_add_adapter_tables.py` for the
  core tables and `migrations/versions/2026062501_add_adapter_chat_feedback.py` for
  `adapter_chat_feedback`).
- **`_direct_rag` 5-minute cache** keys on `(question, document_title, last-2
  turns)` — UI re-fetches return the cached answer; a content change in the
  source won't reflect until the TTL expires.
