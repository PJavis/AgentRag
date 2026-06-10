# database — Async SQLAlchemy engine, session factory & ORM models (Postgres)

## Mục đích / Purpose
Module nền tảng (foundational) định nghĩa kết nối Postgres của AgentRag: một
async `engine`, một session factory `AsyncSessionLocal`, lớp `Base` (declarative
metadata) và toàn bộ ORM models cốt lõi (User, Project, Document, Segment,
SyncLog, Conversation, ChatMessage, EventLog). Mọi worker IO trong hệ thống mở
session từ đây để đọc/ghi trạng thái persistent. This is the single source of
truth for the relational schema; it does NOT contain business logic.

## Plane
**Infrastructure.** Đây là tầng IO thuần (raw DB primitives) mà cả Reasoning
Plane lẫn Execution Plane dùng tới. Bản thân module không tham gia DI
`ServiceContainer` — nó được import trực tiếp như một singleton process-wide
(`engine` + `AsyncSessionLocal` được tạo một lần khi module load).

## Key files
| File | Responsibility |
| --- | --- |
| `__init__.py` | Tạo async `engine` từ `settings.DATABASE_URL`, tạo session factory `AsyncSessionLocal` (`async_sessionmaker`, `expire_on_commit=False`), re-export `Base` + 6 model (`Project, Document, Segment, SyncLog, Conversation, ChatMessage`). `User` và `EventLog` **không** re-export ở đây — import qua `src.agentrag.database.models`. |
| `base.py` | `class Base(DeclarativeBase)` — gốc declarative metadata mà mọi bảng (kể cả `ontology/models.py`, `adapter/db.py`) kế thừa và đăng ký vào `Base.metadata`. |
| `models.py` | Định nghĩa các ORM model cốt lõi: `User`, `Project`, `Document`, `Segment`, `SyncLog`, `Conversation`, `ChatMessage`, `EventLog`. |

## Public interface
Truy cập bằng **import trực tiếp** (không qua `ServiceContainer`, không qua
Protocol trong `services/protocols.py`):

```python
from src.agentrag.database import AsyncSessionLocal, engine, Base
from src.agentrag.database.models import (
    User, Project, Document, Segment, SyncLog, Conversation, ChatMessage, EventLog,
)
```

- `engine` — `AsyncEngine` (psycopg async driver), tạo một lần ở module-level, `echo=False`.
- `AsyncSessionLocal` — `async_sessionmaker[AsyncSession]`; mẫu dùng chuẩn ở mọi
  caller:
  ```python
  async with AsyncSessionLocal() as session:
      ...
      await session.commit()
  ```
  `expire_on_commit=False` nên object vẫn đọc được attribute sau `commit()`
  (không phát thêm SELECT / không lỗi `DetachedInstanceError`).
- `Base` — chỉ dùng metadata; `Base.metadata.create_all` được gọi bởi
  `adapter/db.py::create_adapter_tables()` lúc startup (idempotent,
  `checkfirst=True`). Schema thật được quản lý bằng **Alembic** (xem Gotchas).

### Model quan hệ (relationships)
- `Project 1—* Document` (`Project.documents` ↔ `Document.project`).
- `Document 1—* Segment` (`Document.segments` ↔ `Segment.document`), cascade
  `all, delete-orphan` — `session.delete(document)` phát DELETE tường minh cho
  Segment con (tránh vi phạm NOT NULL `segments.document_id`).
- `Conversation 1—* ChatMessage` (`Conversation.messages` ↔ `ChatMessage.conversation`).
- `Document.user_id`, `Conversation.user_id`, `EventLog.user_id` → `users.id`
  (nullable; bản ghi pre-auth có thể NULL).

## Data flow
Inputs (callers mở session): ingestion (`ingestion/pipeline.py`,
`ingestion/stores/postgres_store.py`), graph jobs (`graph/graph_jobs.py`,
`graph/vision_jobs.py`), agent tools (`agent/tools.py`, `agent/service.py`),
auth (`adapter/auth_service.py`), API routers (`adapter/routers/*` —
chat/sources/notebooks/insights/transformations), ontology resolver
(`ontology/resolver.py`), upload dedupe (`adapter/upload_dedupe.py`), activity
log (`observability/activity.py`), storage (`services/storage_service.py`).

Flow: caller `async with AsyncSessionLocal()` → đọc/ghi các model trong
`models.py` → Postgres. Module này phụ thuộc downstream chỉ vào
`src.agentrag.config.settings` (để lấy `DATABASE_URL`); không phụ thuộc module
AgentRag nào khác — nên nó nằm dưới đáy đồ thị phụ thuộc.

## Schema notes (models.py)
- Khóa chính ở mọi bảng là `PG_UUID(as_uuid=True)` default `uuid.uuid4`.
- `Document` mang trạng thái ingestion/graph: `graph_status`
  (`queued | parsing | searchable | enriching | done | done_partial | failed`,
  NULL = bản ghi cũ trước migration; `searchable`+ mới dùng được cho chat),
  `graph_synced`, `graph_total/processed/failed_chunks`,
  `parse_total_pages` / `parse_done_pages` (tiến độ parse phase; per-page live đi
  qua SSE), `content_hash` (dùng cho dedupe by hash).
- `Segment.extra_metadata` (JSON), `segment_type`, `section_path`, `position`,
  `version` — đơn vị retrieval; cột `content` là text gốc của chunk.
- `ChatMessage` lưu các tín hiệu của một lượt chat: `citations` (JSON),
  `tool_trace` (JSON), `timings_ms` (JSON), `extra_metadata` (JSON). Đây là nơi
  persist các UI-signal của vòng RAG mới (xem Recent additions).
- `EventLog` (S6) — activity feed per-user (`event_type`, `target_kind`,
  `target_id`, `payload` JSON).
- `User` có index unique `ix_users_email` và `ix_users_google_id`; hỗ trợ cả
  password (`password_hash`) lẫn Google OAuth (`google_id`).

## Config
Module chỉ đọc một giá trị: `settings.DATABASE_URL` (property tổng hợp trong
`config.py` từ `POSTGRES_USER`, `POSTGRES_PASSWORD` (URL-quoted),
`POSTGRES_HOST`, `POSTGRES_PORT` mặc định `5433`, `POSTGRES_DB`). Driver:
`postgresql+psycopg://...` (psycopg3, async). Không có feature flag riêng cho
module này.

## Recent additions (2026-06)
Schema thay đổi (qua Alembic), KHÔNG phải logic trong module này:
- `migrations/versions/2026060501_add_parse_page_columns.py` —
  `Document.parse_total_pages` / `parse_done_pages`.
- `ChatMessage.citations` / `timings_ms` (đã có sẵn) là chỗ ngồi của các
  UI-signal mới do vòng RAG `feat/ragas-langfuse-reranker` sinh ra: citations
  mang `node_level` / `context_text` (RAPTOR / Contextual Retrieval), và
  `timings_ms.critique` (CRAG). Module này chỉ **lưu** các trường JSON đó nguyên
  trạng — không sinh, không validate. Các flag default-OFF gating những tính
  năng ấy nằm ở module sinh ra chúng (`agent/`, `ingestion/`,
  `services/semantic_cache.py`), không nằm ở đây.

## Gotchas
- **Alembic là nguồn sự thật của schema, không phải `create_all`.** Bảng được
  quản lý bằng migrations trong `migrations/versions/` (ví dụ
  `aa5dd4a77554_initial.py`, `c3f9f8b1d2e7_add_conversation_tables.py`,
  `d7e2a4b9c1f0_add_adapter_tables.py`, `2026051601_add_user_id_and_event_log.py`,
  `2026051502_enable_pg_trgm.py`). `create_adapter_tables()` (`adapter/db.py`)
  gọi `Base.metadata.create_all(checkfirst=True)` chỉ như lưới an toàn lúc
  startup — thêm/đổi cột vẫn phải viết migration, đừng chỉ sửa `models.py`.
- `Base` được nhiều module mở rộng: `ontology/models.py` (`OntologyTerm`) và
  `adapter/db.py` đều `class X(Base)` để dồn vào chung `Base.metadata`. Vì vậy
  một vài bảng dùng `__table_args__ = {"extend_existing": True}` để tránh xung
  đột khi metadata được import nhiều lần.
- `expire_on_commit=False`: tiện cho việc đọc object sau commit, nhưng nghĩa là
  object KHÔNG tự refresh — nếu cần giá trị mới nhất từ DB phải `await
  session.refresh(obj)`.
- `engine` là singleton process-wide tạo lúc import; với multi-worker
  (`UVICORN_WORKERS > 1`) mỗi worker có pool riêng — đừng giả định một connection
  pool dùng chung giữa các process.
- Tên file header còn ghi `src/pam/...` (di sản đổi tên dự án PAM → AgentRag);
  package thật là `src.agentrag.database`. Chỉ là comment lạc hậu, không ảnh
  hưởng runtime.
- `Conversation`/`Document`/`EventLog` có `user_id` nullable — code đọc phải chịu
  được NULL (bản ghi tạo trước khi auth được bật).
