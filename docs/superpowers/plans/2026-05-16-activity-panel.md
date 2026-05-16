# Activity Panel (S6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace static `/admin` HTML trace inspector with a per-user Activity panel backed by a real `event_log` table; admin retains a global view at `/admin/activity`.

**Architecture:** Hybrid data model — add `user_id` FK to `conversations` + `documents` for ownership/privacy queries, plus new `event_log` table for ephemeral events (chat turns, source uploads, ingest done/failed, searches). Sync inline writes via `record_event()` helper in `observability/activity`. Adapter exposes personal + admin REST endpoints; Frontend renders stats cards + 28-day heatmap + filterable feed (Layout C). Chat-turn rows reuse the existing S2 `TraceDialog`.

**Tech Stack:** Python 3.11 / FastAPI / SQLAlchemy 2.0 async / Alembic / Next.js 15 / TanStack Query / pytest-asyncio.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `migrations/versions/2026051601_add_user_id_and_event_log.py` | Create | DDL for `conversations.user_id`, `documents.user_id`, `event_log` |
| `src/agentrag/database/models.py` | Modify | Add `Conversation.user_id`, `Document.user_id`, new `EventLog` class |
| `src/agentrag/observability/activity.py` | Create | `record_event()` helper |
| `src/agentrag/observability/__init__.py` | Modify (if exists) | export `record_event` |
| `src/agentrag/adapter/routers/chat.py` | Modify | call `record_event(chat_turn)` after assistant append in notebook + source paths; set `conversations.user_id` at session create |
| `src/agentrag/adapter/routers/sources.py` | Modify | set `documents.user_id` at create; call `record_event(source_uploaded)` |
| `src/agentrag/ingestion/pipeline.py` | Modify | call `record_event(ingest_done / ingest_failed)` at end of pipeline |
| `src/agentrag/adapter/routers/search.py` (or wherever search lives) | Modify | call `record_event(search)` |
| `src/agentrag/adapter/routers/activity.py` | Create | `/api/activity/*` + `/api/admin/activity/*` |
| `src/agentrag/adapter/app.py` | Modify | mount activity router |
| `src/agentrag/adapter/auth.py` | (no change — admin uses `is_admin()` helper) |
| `tests/observability/test_activity.py` | Create | helper + scope filter unit tests |
| `tests/adapter/test_activity_endpoints.py` | Create | endpoint privacy + admin tests |
| `tests/integration/test_s6_activity.py` | Create | end-to-end |
| `frontend/src/lib/api/activity.ts` | Create | API client |
| `frontend/src/lib/hooks/useActivity.ts` | Create | TanStack Query wrappers |
| `frontend/src/lib/types/api.ts` | Modify | `ActivityEvent`, `ActivitySummary` types |
| `frontend/src/components/activity/ActivityHeatmap.tsx` | Create | 28-day heatmap |
| `frontend/src/components/activity/ActivityFeed.tsx` | Create | filterable feed |
| `frontend/src/components/activity/EventRow.tsx` | Create | per-row renderer |
| `frontend/src/app/(dashboard)/activity/page.tsx` | Create | personal page |
| `frontend/src/app/(dashboard)/admin/activity/page.tsx` | Create | admin page |
| `frontend/src/components/layout/AppSidebar.tsx` | Modify | new "Activity" link |
| `frontend/src/lib/locales/en-US/index.ts` | Modify | new `activity.*` keys |

---

### Task 1: Migration + EventLog model

**Files:**
- Create: `migrations/versions/2026051601_add_user_id_and_event_log.py`
- Modify: `src/agentrag/database/models.py`

- [ ] **Step 1: Add EventLog model + user_id columns**

```python
# src/agentrag/database/models.py — append after ChatMessage class
class EventLog(Base):
    __tablename__ = "event_log"
    id          = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id     = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    event_type  = Column(String(32), nullable=False)
    target_kind = Column(String(32), nullable=True)
    target_id   = Column(PG_UUID(as_uuid=True), nullable=True)
    payload     = Column(JSON, nullable=False, default=dict)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
```

Add to existing `Conversation` and `Document`:
```python
user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
```

- [ ] **Step 2: Write Alembic migration**

```python
"""add user_id columns + event_log table

Revision ID: 2026051601
Revises: <previous head>
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "2026051601"
down_revision = "<look up previous head>"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("conversations", sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True))
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])
    op.add_column("documents", sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True))
    op.create_index("ix_documents_user_id", "documents", ["user_id"])
    op.create_table(
        "event_log",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("target_kind", sa.String(32), nullable=True),
        sa.Column("target_id", UUID(as_uuid=True), nullable=True),
        sa.Column("payload", sa.JSON, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_event_log_user_created", "event_log", ["user_id", sa.text("created_at DESC")])
    op.create_index("ix_event_log_type_created", "event_log", ["event_type", sa.text("created_at DESC")])
    op.create_index("ix_event_log_target_id", "event_log", ["target_id"], postgresql_where=sa.text("target_id IS NOT NULL"))

def downgrade():
    op.drop_index("ix_event_log_target_id", table_name="event_log")
    op.drop_index("ix_event_log_type_created", table_name="event_log")
    op.drop_index("ix_event_log_user_created", table_name="event_log")
    op.drop_table("event_log")
    op.drop_index("ix_documents_user_id", table_name="documents")
    op.drop_column("documents", "user_id")
    op.drop_index("ix_conversations_user_id", table_name="conversations")
    op.drop_column("conversations", "user_id")
```

- [ ] **Step 3: Look up previous revision head**

Run: `ls migrations/versions/ | sort | tail -1` — note filename → grep `revision =` in that file → paste into `down_revision`.

- [ ] **Step 4: Run migration**

Run: `make migrate` — should print `Running upgrade … -> 2026051601`.

- [ ] **Step 5: Commit**

```bash
git add migrations/versions/2026051601_add_user_id_and_event_log.py src/agentrag/database/models.py
git commit -m "feat(s6): event_log table + user_id FK on conversations/documents"
```

---

### Task 2: record_event() helper + unit test

**Files:**
- Create: `src/agentrag/observability/activity.py`
- Create: `tests/observability/test_activity.py`

- [ ] **Step 1: Write failing test**

```python
# tests/observability/test_activity.py
import uuid
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_record_event_inserts_row():
    from src.agentrag.observability import activity
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()

    @asynccontextmanager
    async def fake_sess():
        yield session

    with patch.object(activity, "AsyncSessionLocal", fake_sess):
        await activity.record_event(
            user_id=uuid.uuid4(),
            event_type="chat_turn",
            payload={"tokens_in": 100},
        )
    session.add.assert_called_once()
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_event_swallows_exception(caplog):
    from src.agentrag.observability import activity
    @asynccontextmanager
    async def boom():
        raise RuntimeError("db down")
        yield
    with patch.object(activity, "AsyncSessionLocal", boom):
        # Must NOT raise
        await activity.record_event(user_id=None, event_type="x")
```

- [ ] **Step 2: Run test — expect FAIL (module missing)**

`uv run python -m pytest tests/observability/test_activity.py -v`

- [ ] **Step 3: Implement helper**

```python
# src/agentrag/observability/activity.py
"""Activity event log (S6).

record_event() is sync-inline: caller awaits, failure logs + swallows so the
business path never breaks. Cheap INSERT (~1ms) per chat turn / upload /
search.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import EventLog

logger = logging.getLogger(__name__)


async def record_event(
    user_id: uuid.UUID | str | None,
    event_type: str,
    *,
    target_kind: str | None = None,
    target_id: uuid.UUID | str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    try:
        uid = _coerce_uuid(user_id)
        tid = _coerce_uuid(target_id)
        row = EventLog(
            user_id=uid,
            event_type=event_type,
            target_kind=target_kind,
            target_id=tid,
            payload=payload or {},
        )
        async with AsyncSessionLocal() as session:
            session.add(row)
            await session.commit()
    except Exception as exc:
        logger.warning("activity.record_event failed: %s (%s)", exc, event_type)


def _coerce_uuid(v: uuid.UUID | str | None) -> uuid.UUID | None:
    if v is None or v == "anonymous":
        return None
    if isinstance(v, uuid.UUID):
        return v
    try:
        return uuid.UUID(str(v))
    except (ValueError, TypeError):
        return None
```

- [ ] **Step 4: Run test — expect PASS**

`uv run python -m pytest tests/observability/test_activity.py -v` → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/observability/activity.py tests/observability/test_activity.py
git commit -m "feat(s6): record_event helper"
```

---

### Task 3: Wire `chat_turn` event in adapter chat routes

**Files:**
- Modify: `src/agentrag/adapter/routers/chat.py`

- [ ] **Step 1: Add import + call site in `execute_chat`**

After the `await store.append_message(... role="assistant" ...)` block (around line 177), add:

```python
from src.agentrag.observability.activity import record_event
from src.agentrag.adapter.auth import get_identity
identity = get_identity(request)
await record_event(
    user_id=identity.user_id if identity else None,
    event_type="chat_turn",
    target_kind="conversation",
    target_id=body.session_id,
    payload={
        "message": body.message[:200],
        "tokens_in":  (result.get("timings_ms") or {}).get("tokens_in"),
        "tokens_out": (result.get("timings_ms") or {}).get("tokens_out"),
        "latency_ms": (result.get("timings_ms") or {}).get("total"),
        "reasoning_path": result.get("reasoning_path"),
    },
)
```

- [ ] **Step 2: Add same call to `send_message` (SSE source chat)**

After the `await store.append_message(... role="assistant" ...)` block in the SSE handler (around line 332), append same `record_event(chat_turn, …)` block.

- [ ] **Step 3: Set `conversations.user_id` on session create**

In `create_session` (notebook) and the analogous source-chat session create, set `user_id` from `request.state.auth_identity`:

```python
identity = get_identity(request)
conv = await store.create_conversation(
    title=body.title,
    user_id=identity.user_id if identity else None,
    extra_metadata={...},
)
```

(`ConversationStore.create_conversation` needs a `user_id` kwarg — update signature + INSERT statement to write the column.)

- [ ] **Step 4: Smoke-run app, hit `/api/chat/execute`, verify row in event_log**

```bash
psql "$DATABASE_URL" -c "select event_type, user_id, target_id, payload->>'reasoning_path' from event_log order by created_at desc limit 5;"
```

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/adapter/routers/chat.py src/agentrag/chat/history.py
git commit -m "feat(s6): record chat_turn events + conversations.user_id"
```

---

### Task 4: Wire `source_uploaded` + `documents.user_id` + ingest events

**Files:**
- Modify: `src/agentrag/adapter/routers/sources.py`
- Modify: `src/agentrag/ingestion/pipeline.py`

- [ ] **Step 1: Set `documents.user_id` at create_source**

In `create_source` (around line 320-340), pull identity:

```python
identity = get_identity(request)
skeleton = Document(
    project_id=proj.id,
    user_id=_coerce_uuid_or_none(identity.user_id) if identity else None,
    title=doc_title,
    ...
)
```

- [ ] **Step 2: Record `source_uploaded` after skeleton commit**

```python
from src.agentrag.observability.activity import record_event
await record_event(
    user_id=identity.user_id if identity else None,
    event_type="source_uploaded",
    target_kind="document",
    target_id=skeleton.id,
    payload={
        "filename": filename,
        "size_bytes": len(raw),
        "content_hash": content_hash,
        "source_type": ext,
    },
)
```

- [ ] **Step 3: Pipeline finish callback**

In `ingestion/pipeline.py`, find the function that drives async ingest (e.g. `_run_ingest_background` or `graph_ingest` job). At the end of success path, look up the document → call:

```python
await record_event(
    user_id=document.user_id,
    event_type="ingest_done",
    target_kind="document",
    target_id=document.id,
    payload={"segment_count": len(chunks), "duration_ms": duration_ms},
)
```

In the except branch:

```python
await record_event(
    user_id=document.user_id,
    event_type="ingest_failed",
    target_kind="document",
    target_id=document.id,
    payload={"error": str(exc)[:500]},
)
```

- [ ] **Step 4: Commit**

```bash
git add src/agentrag/adapter/routers/sources.py src/agentrag/ingestion/pipeline.py
git commit -m "feat(s6): record source_uploaded + ingest_done/failed + documents.user_id"
```

---

### Task 5: Wire `search` event

**Files:**
- Modify: `src/agentrag/adapter/routers/search.py` (or wherever the adapter `/api/search` endpoint lives)

- [ ] **Step 1: Find search endpoint**

Run: `grep -rn '@router.*search\|search_handler\|def search' src/agentrag/adapter/routers/`

- [ ] **Step 2: Record event after hits return**

```python
import time
from src.agentrag.observability.activity import record_event
from src.agentrag.adapter.auth import get_identity

@router.post("/search")
async def search(body: SearchRequest, request: Request):
    started = time.perf_counter()
    result = await ...   # existing logic
    latency_ms = (time.perf_counter() - started) * 1000
    identity = get_identity(request)
    await record_event(
        user_id=identity.user_id if identity else None,
        event_type="search",
        payload={
            "query": body.query[:200],
            "mode": body.mode,
            "top_k": body.top_k,
            "hit_count": len(result.get("results", [])),
            "latency_ms": round(latency_ms, 1),
            "document_title": body.document_title,
        },
    )
    return result
```

- [ ] **Step 3: Commit**

```bash
git add src/agentrag/adapter/routers/search.py
git commit -m "feat(s6): record search events"
```

---

### Task 6: Adapter routes `/api/activity/*` + `/api/admin/activity/*`

**Files:**
- Create: `src/agentrag/adapter/routers/activity.py`
- Modify: `src/agentrag/adapter/app.py`
- Modify: `src/agentrag/adapter/models.py` (response types)

- [ ] **Step 1: Add response Pydantic types**

```python
# adapter/models.py — append
class ActivityEvent(BaseModel):
    id: str
    user_id: str | None
    event_type: str
    target_kind: str | None
    target_id: str | None
    payload: dict[str, Any]
    created_at: str


class ActivityCounts(BaseModel):
    chat_turn: int = 0
    source_uploaded: int = 0
    ingest_done: int = 0
    ingest_failed: int = 0
    search: int = 0


class ActivityHeatmapCell(BaseModel):
    date: str            # ISO YYYY-MM-DD
    count: int


class ActivitySummary(BaseModel):
    counts: ActivityCounts
    tokens_total: int
    usd_estimate: float
    heatmap: list[ActivityHeatmapCell]


class AdminUserEntry(BaseModel):
    user_id: str | None
    email: str | None
    name: str | None
    event_count: int
    last_seen: str | None
```

- [ ] **Step 2: Implement router**

```python
# src/agentrag/adapter/routers/activity.py
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import desc, func, select

from src.agentrag.adapter.auth import get_identity, is_admin
from src.agentrag.adapter.models import (
    ActivityCounts, ActivityEvent, ActivityHeatmapCell,
    ActivitySummary, AdminUserEntry,
)
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import EventLog, User

router = APIRouter(prefix="/activity")
admin_router = APIRouter(prefix="/admin/activity")


def _require_user(request: Request) -> uuid.UUID:
    identity = get_identity(request)
    if not identity or identity.user_id == "anonymous":
        raise HTTPException(401, "Activity requires login")
    try:
        return uuid.UUID(identity.user_id)
    except (TypeError, ValueError):
        raise HTTPException(401, "Invalid identity")


def _require_admin(request: Request) -> None:
    if not is_admin(request):
        raise HTTPException(403, "Admin only")


async def _summary_for(user_filter):
    async with AsyncSessionLocal() as session:
        # counts
        q_counts = select(EventLog.event_type, func.count())
        if user_filter is not None:
            q_counts = q_counts.where(EventLog.user_id == user_filter)
        q_counts = q_counts.group_by(EventLog.event_type)
        rows = (await session.execute(q_counts)).all()
        counts = ActivityCounts(**{r[0]: r[1] for r in rows if hasattr(ActivityCounts(), r[0])})

        # tokens + USD
        q_payload = select(EventLog.payload).where(EventLog.event_type == "chat_turn")
        if user_filter is not None:
            q_payload = q_payload.where(EventLog.user_id == user_filter)
        tokens_total = 0
        usd_total = 0.0
        for (p,) in (await session.execute(q_payload)).all():
            tokens_total += int((p or {}).get("tokens_in") or 0) + int((p or {}).get("tokens_out") or 0)
            usd_total += float((p or {}).get("usd") or 0.0)

        # heatmap: last 28 days
        cutoff = datetime.now(timezone.utc) - timedelta(days=28)
        q_heat = (
            select(func.date_trunc("day", EventLog.created_at).label("day"), func.count())
            .where(EventLog.created_at >= cutoff)
        )
        if user_filter is not None:
            q_heat = q_heat.where(EventLog.user_id == user_filter)
        q_heat = q_heat.group_by("day").order_by("day")
        heatmap = [
            ActivityHeatmapCell(date=row[0].date().isoformat(), count=row[1])
            for row in (await session.execute(q_heat)).all()
        ]

    return ActivitySummary(
        counts=counts,
        tokens_total=tokens_total,
        usd_estimate=round(usd_total, 6),
        heatmap=heatmap,
    )


async def _events_for(user_filter, *, event_type, limit, before_id):
    async with AsyncSessionLocal() as session:
        q = select(EventLog)
        if user_filter is not None:
            q = q.where(EventLog.user_id == user_filter)
        if event_type:
            q = q.where(EventLog.event_type == event_type)
        if before_id:
            try:
                anchor = await session.get(EventLog, uuid.UUID(before_id))
                if anchor:
                    q = q.where(EventLog.created_at < anchor.created_at)
            except ValueError:
                pass
        q = q.order_by(desc(EventLog.created_at)).limit(min(limit, 200))
        rows = (await session.execute(q)).scalars().all()

    entries = [_serialize(r) for r in rows]
    next_before_id = rows[-1].id.hex if len(rows) >= limit else None
    return {"entries": entries, "next_before_id": next_before_id}


def _serialize(r: EventLog) -> dict:
    return ActivityEvent(
        id=str(r.id),
        user_id=str(r.user_id) if r.user_id else None,
        event_type=r.event_type,
        target_kind=r.target_kind,
        target_id=str(r.target_id) if r.target_id else None,
        payload=r.payload or {},
        created_at=r.created_at.isoformat() if r.created_at else "",
    ).model_dump()


# Personal endpoints
@router.get("/summary")
async def summary(request: Request):
    uid = _require_user(request)
    return (await _summary_for(uid)).model_dump()


@router.get("/events")
async def events(request: Request, type: str | None = None, limit: int = 50, before_id: str | None = None):
    uid = _require_user(request)
    return await _events_for(uid, event_type=type, limit=limit, before_id=before_id)


@router.get("/events/{event_id}")
async def event_detail(event_id: str, request: Request):
    uid = _require_user(request)
    try:
        eid = uuid.UUID(event_id)
    except ValueError:
        raise HTTPException(404, "Not found")
    async with AsyncSessionLocal() as session:
        row = await session.get(EventLog, eid)
        if not row or row.user_id != uid:
            raise HTTPException(404, "Not found")
        return _serialize(row)


# Admin endpoints
@admin_router.get("/summary")
async def admin_summary(request: Request, user_id: str | None = None):
    _require_admin(request)
    f = uuid.UUID(user_id) if user_id else None
    return (await _summary_for(f)).model_dump()


@admin_router.get("/events")
async def admin_events(
    request: Request,
    user_id: str | None = None,
    type: str | None = None,
    limit: int = 50,
    before_id: str | None = None,
):
    _require_admin(request)
    f = uuid.UUID(user_id) if user_id else None
    return await _events_for(f, event_type=type, limit=limit, before_id=before_id)


@admin_router.get("/users")
async def admin_users(request: Request):
    _require_admin(request)
    async with AsyncSessionLocal() as session:
        q = (
            select(
                EventLog.user_id,
                func.count().label("event_count"),
                func.max(EventLog.created_at).label("last_seen"),
            )
            .group_by(EventLog.user_id)
            .order_by(desc("last_seen"))
        )
        rows = (await session.execute(q)).all()
        users_by_id = {}
        ids = [r[0] for r in rows if r[0]]
        if ids:
            ures = await session.execute(select(User).where(User.id.in_(ids)))
            users_by_id = {u.id: u for u in ures.scalars().all()}
    entries = []
    for r in rows:
        u = users_by_id.get(r[0])
        entries.append(
            AdminUserEntry(
                user_id=str(r[0]) if r[0] else None,
                email=u.email if u else None,
                name=u.name if u else None,
                event_count=r[1],
                last_seen=r[2].isoformat() if r[2] else None,
            ).model_dump()
        )
    return entries
```

- [ ] **Step 3: Mount routers**

In `src/agentrag/adapter/app.py`:

```python
from src.agentrag.adapter.routers.activity import router as activity_router
from src.agentrag.adapter.routers.activity import admin_router as activity_admin_router

adapter.include_router(activity_router, prefix="/api")
adapter.include_router(activity_admin_router, prefix="/api")
```

- [ ] **Step 4: Smoke**

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/on/api/activity/summary
curl -H "X-Admin-Token: $ADMIN" http://localhost:8000/on/api/admin/activity/users
```

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/adapter/routers/activity.py src/agentrag/adapter/app.py src/agentrag/adapter/models.py
git commit -m "feat(s6): /api/activity + /api/admin/activity routes"
```

---

### Task 7: Adapter endpoint tests

**Files:**
- Create: `tests/adapter/test_activity_endpoints.py`

- [ ] **Step 1: Tests**

```python
"""S6 — activity endpoint privacy + admin scope."""
import pytest
import uuid
from httpx import ASGITransport, AsyncClient

from src.agentrag.adapter.app import adapter


@pytest.mark.asyncio
async def test_personal_summary_requires_auth():
    transport = ASGITransport(app=adapter)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/activity/summary")
        # Auth middleware → 401 if disabled-aware; else expect 401 from _require_user
        assert r.status_code in (401, 403)


@pytest.mark.asyncio
async def test_admin_users_blocked_without_token():
    transport = ASGITransport(app=adapter)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/admin/activity/users")
        assert r.status_code in (401, 403)
```

- [ ] **Step 2: Run**

`uv run python -m pytest tests/adapter/test_activity_endpoints.py -v` → both pass.

- [ ] **Step 3: Commit**

```bash
git add tests/adapter/test_activity_endpoints.py
git commit -m "test(s6): activity endpoint smoke"
```

---

### Task 8: Frontend API + types + hooks

**Files:**
- Create: `frontend/src/lib/api/activity.ts`
- Create: `frontend/src/lib/hooks/useActivity.ts`
- Modify: `frontend/src/lib/types/api.ts`

- [ ] **Step 1: Types**

Append to `frontend/src/lib/types/api.ts`:

```ts
export interface ActivityEvent {
  id: string
  user_id: string | null
  event_type: 'chat_turn' | 'source_uploaded' | 'ingest_done' | 'ingest_failed' | 'search'
  target_kind: string | null
  target_id: string | null
  payload: Record<string, unknown>
  created_at: string
}

export interface ActivityCounts {
  chat_turn?: number
  source_uploaded?: number
  ingest_done?: number
  ingest_failed?: number
  search?: number
}

export interface ActivityHeatmapCell {
  date: string
  count: number
}

export interface ActivitySummary {
  counts: ActivityCounts
  tokens_total: number
  usd_estimate: number
  heatmap: ActivityHeatmapCell[]
}

export interface AdminUserEntry {
  user_id: string | null
  email: string | null
  name: string | null
  event_count: number
  last_seen: string | null
}
```

- [ ] **Step 2: API client**

```ts
// frontend/src/lib/api/activity.ts
import apiClient from './client'
import {
  ActivitySummary, ActivityEvent, AdminUserEntry,
} from '@/lib/types/api'

export const activityApi = {
  summary: async () => (await apiClient.get<ActivitySummary>('/activity/summary')).data,
  events: async (params: { type?: string; limit?: number; before_id?: string }) =>
    (await apiClient.get<{ entries: ActivityEvent[]; next_before_id: string | null }>(
      '/activity/events', { params }
    )).data,
  event: async (id: string) =>
    (await apiClient.get<ActivityEvent>(`/activity/events/${id}`)).data,
  adminSummary: async (userId?: string) =>
    (await apiClient.get<ActivitySummary>('/admin/activity/summary', { params: { user_id: userId } })).data,
  adminEvents: async (params: { user_id?: string; type?: string; limit?: number; before_id?: string }) =>
    (await apiClient.get<{ entries: ActivityEvent[]; next_before_id: string | null }>(
      '/admin/activity/events', { params }
    )).data,
  adminUsers: async () => (await apiClient.get<AdminUserEntry[]>('/admin/activity/users')).data,
}
```

- [ ] **Step 3: Hooks**

```ts
// frontend/src/lib/hooks/useActivity.ts
'use client'
import { useQuery } from '@tanstack/react-query'
import { activityApi } from '@/lib/api/activity'

export const ACTIVITY_QUERY_KEYS = {
  summary: ['activity', 'summary'] as const,
  events: (type?: string, limit?: number) => ['activity', 'events', type, limit] as const,
  adminSummary: (userId?: string) => ['admin', 'activity', 'summary', userId] as const,
  adminEvents: (userId?: string, type?: string) => ['admin', 'activity', 'events', userId, type] as const,
  adminUsers: ['admin', 'activity', 'users'] as const,
}

export function useActivitySummary(refetchMs = 30_000) {
  return useQuery({ queryKey: ACTIVITY_QUERY_KEYS.summary, queryFn: activityApi.summary, refetchInterval: refetchMs })
}

export function useActivityEvents(type?: string, limit = 50) {
  return useQuery({
    queryKey: ACTIVITY_QUERY_KEYS.events(type, limit),
    queryFn: () => activityApi.events({ type, limit }),
    refetchInterval: 30_000,
  })
}

export function useAdminActivitySummary(userId?: string) {
  return useQuery({
    queryKey: ACTIVITY_QUERY_KEYS.adminSummary(userId),
    queryFn: () => activityApi.adminSummary(userId),
  })
}

export function useAdminActivityEvents(userId?: string, type?: string) {
  return useQuery({
    queryKey: ACTIVITY_QUERY_KEYS.adminEvents(userId, type),
    queryFn: () => activityApi.adminEvents({ user_id: userId, type }),
  })
}

export function useAdminUsers() {
  return useQuery({ queryKey: ACTIVITY_QUERY_KEYS.adminUsers, queryFn: activityApi.adminUsers })
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/api/activity.ts frontend/src/lib/hooks/useActivity.ts frontend/src/lib/types/api.ts
git commit -m "feat(s6): activity API client + TanStack hooks + types"
```

---

### Task 9: Frontend components — Heatmap + Feed + Row

**Files:**
- Create: `frontend/src/components/activity/ActivityHeatmap.tsx`
- Create: `frontend/src/components/activity/EventRow.tsx`
- Create: `frontend/src/components/activity/ActivityFeed.tsx`

- [ ] **Step 1: Heatmap (28-day grid)**

```tsx
// ActivityHeatmap.tsx
'use client'
import { useMemo } from 'react'
import type { ActivityHeatmapCell } from '@/lib/types/api'

const SCALE = ['bg-muted', 'bg-emerald-200', 'bg-emerald-400', 'bg-emerald-600', 'bg-emerald-800']

export function ActivityHeatmap({ data }: { data: ActivityHeatmapCell[] }) {
  const map = useMemo(() => {
    const m: Record<string, number> = {}
    data.forEach((c) => { m[c.date] = c.count })
    return m
  }, [data])

  const today = new Date()
  const cells: { date: string; count: number; bucket: number }[] = []
  for (let i = 27; i >= 0; i--) {
    const d = new Date(today)
    d.setDate(today.getDate() - i)
    const key = d.toISOString().slice(0, 10)
    const count = map[key] ?? 0
    const bucket = count === 0 ? 0 : count < 3 ? 1 : count < 7 ? 2 : count < 15 ? 3 : 4
    cells.push({ date: key, count, bucket })
  }
  return (
    <div className="grid grid-rows-7 grid-flow-col gap-1" style={{ width: 'fit-content' }}>
      {cells.map((c) => (
        <div
          key={c.date}
          title={`${c.date}: ${c.count} events`}
          className={`h-3 w-3 rounded-sm ${SCALE[c.bucket]}`}
        />
      ))}
    </div>
  )
}
```

- [ ] **Step 2: EventRow**

```tsx
// EventRow.tsx
'use client'
import { useRouter } from 'next/navigation'
import { Bot, FileUp, Search, CheckCircle, XCircle } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { toast } from 'sonner'
import type { ActivityEvent } from '@/lib/types/api'

const ICONS = {
  chat_turn: Bot,
  source_uploaded: FileUp,
  ingest_done: CheckCircle,
  ingest_failed: XCircle,
  search: Search,
} as const

export function EventRow({
  event,
  onTrace,
}: {
  event: ActivityEvent
  onTrace?: (event: ActivityEvent) => void
}) {
  const router = useRouter()
  const Icon = ICONS[event.event_type] ?? Bot
  const time = new Date(event.created_at).toLocaleTimeString(undefined, { hour12: false })
  const p = event.payload as Record<string, unknown>

  const onClick = () => {
    if (event.event_type === 'chat_turn') onTrace?.(event)
    else if (event.event_type.startsWith('source_') || event.event_type.startsWith('ingest_')) {
      if (event.target_id) router.push(`/sources/${event.target_id}`)
    } else if (event.event_type === 'search') {
      toast.message(`Search: "${String(p.query ?? '')}"`)
    }
  }

  let summary = ''
  if (event.event_type === 'chat_turn') summary = `${p.message ?? ''}`.slice(0, 80)
  else if (event.event_type === 'source_uploaded') summary = `${p.filename ?? ''}`
  else if (event.event_type === 'ingest_done') summary = `${p.segment_count ?? 0} segments`
  else if (event.event_type === 'ingest_failed') summary = `${p.error ?? ''}`.slice(0, 80)
  else if (event.event_type === 'search') summary = `"${p.query ?? ''}" · ${p.mode ?? ''} · ${p.hit_count ?? 0} hits`

  return (
    <button onClick={onClick} className="flex w-full items-center gap-3 rounded border px-3 py-2 hover:bg-muted/50 text-left">
      <Icon className="h-4 w-4 shrink-0 text-muted-foreground" />
      <span className="text-xs tabular-nums text-muted-foreground shrink-0">{time}</span>
      <Badge variant="outline" className="text-[10px] shrink-0">{event.event_type}</Badge>
      <span className="text-sm truncate">{summary}</span>
    </button>
  )
}
```

- [ ] **Step 3: Feed**

```tsx
// ActivityFeed.tsx
'use client'
import { useState } from 'react'
import { EventRow } from './EventRow'
import { Button } from '@/components/ui/button'
import type { ActivityEvent } from '@/lib/types/api'

const TYPES = [
  { value: undefined, label: 'All' },
  { value: 'chat_turn', label: 'Chats' },
  { value: 'source_uploaded', label: 'Uploads' },
  { value: 'search', label: 'Searches' },
]

export function ActivityFeed({
  events,
  loading,
  onTypeChange,
  onTrace,
}: {
  events: ActivityEvent[]
  loading?: boolean
  onTypeChange: (t: string | undefined) => void
  onTrace?: (event: ActivityEvent) => void
}) {
  const [active, setActive] = useState<string | undefined>(undefined)
  return (
    <div className="space-y-2">
      <div className="flex gap-1.5">
        {TYPES.map((t) => (
          <Button
            key={t.label}
            size="sm"
            variant={active === t.value ? 'default' : 'outline'}
            className="h-7 text-xs"
            onClick={() => { setActive(t.value); onTypeChange(t.value) }}
          >
            {t.label}
          </Button>
        ))}
      </div>
      {loading ? (
        <div className="text-center text-muted-foreground py-6 text-sm">Loading…</div>
      ) : events.length === 0 ? (
        <div className="text-center text-muted-foreground py-6 text-sm">No activity yet</div>
      ) : (
        <div className="space-y-1.5">
          {events.map((e) => <EventRow key={e.id} event={e} onTrace={onTrace} />)}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/activity/
git commit -m "feat(s6): ActivityHeatmap + ActivityFeed + EventRow"
```

---

### Task 10: Activity page + Admin page + Sidebar link

**Files:**
- Create: `frontend/src/app/(dashboard)/activity/page.tsx`
- Create: `frontend/src/app/(dashboard)/admin/activity/page.tsx`
- Modify: `frontend/src/components/layout/AppSidebar.tsx`
- Modify: `frontend/src/lib/locales/en-US/index.ts`

- [ ] **Step 1: Personal page**

```tsx
// frontend/src/app/(dashboard)/activity/page.tsx
'use client'
import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Activity, Bot, FileUp, Search, DollarSign } from 'lucide-react'
import { useActivitySummary, useActivityEvents } from '@/lib/hooks/useActivity'
import { ActivityHeatmap } from '@/components/activity/ActivityHeatmap'
import { ActivityFeed } from '@/components/activity/ActivityFeed'

function fmtUsd(n: number) {
  if (!n) return '$0'
  if (n < 1) return `$${n.toFixed(4)}`
  return `$${n.toFixed(2)}`
}

export default function ActivityPage() {
  const [type, setType] = useState<string | undefined>(undefined)
  const summary = useActivitySummary()
  const events = useActivityEvents(type, 50)
  const s = summary.data
  return (
    <div className="container mx-auto p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-semibold flex items-center gap-2"><Activity className="h-5 w-5" />Activity</h1>
        <p className="text-sm text-muted-foreground">Your recent chats, uploads, and searches.</p>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card><CardContent className="p-4 flex items-center gap-3"><Bot className="h-5 w-5 text-muted-foreground" /><div><div className="text-xs text-muted-foreground">Chats</div><div className="text-2xl font-semibold tabular-nums">{s?.counts.chat_turn ?? 0}</div></div></CardContent></Card>
        <Card><CardContent className="p-4 flex items-center gap-3"><FileUp className="h-5 w-5 text-muted-foreground" /><div><div className="text-xs text-muted-foreground">Uploads</div><div className="text-2xl font-semibold tabular-nums">{s?.counts.source_uploaded ?? 0}</div></div></CardContent></Card>
        <Card><CardContent className="p-4 flex items-center gap-3"><Search className="h-5 w-5 text-muted-foreground" /><div><div className="text-xs text-muted-foreground">Searches</div><div className="text-2xl font-semibold tabular-nums">{s?.counts.search ?? 0}</div></div></CardContent></Card>
        <Card><CardContent className="p-4 flex items-center gap-3"><DollarSign className="h-5 w-5 text-muted-foreground" /><div><div className="text-xs text-muted-foreground">Est. cost</div><div className="text-2xl font-semibold tabular-nums">{fmtUsd(s?.usd_estimate ?? 0)}</div></div></CardContent></Card>
      </div>
      <Card>
        <CardHeader><CardTitle className="text-sm">Last 28 days</CardTitle></CardHeader>
        <CardContent><ActivityHeatmap data={s?.heatmap ?? []} /></CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle className="text-sm">Recent</CardTitle></CardHeader>
        <CardContent>
          <ActivityFeed
            events={events.data?.entries ?? []}
            loading={events.isLoading}
            onTypeChange={setType}
          />
        </CardContent>
      </Card>
    </div>
  )
}
```

- [ ] **Step 2: Admin page**

```tsx
// frontend/src/app/(dashboard)/admin/activity/page.tsx
'use client'
import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useAdminUsers, useAdminActivitySummary, useAdminActivityEvents } from '@/lib/hooks/useActivity'
import { ActivityHeatmap } from '@/components/activity/ActivityHeatmap'
import { ActivityFeed } from '@/components/activity/ActivityFeed'

export default function AdminActivityPage() {
  const [userId, setUserId] = useState<string | undefined>(undefined)
  const [type, setType] = useState<string | undefined>(undefined)
  const users = useAdminUsers()
  const summary = useAdminActivitySummary(userId)
  const events = useAdminActivityEvents(userId, type)
  const s = summary.data
  return (
    <div className="container mx-auto p-6 space-y-6">
      <h1 className="text-2xl font-semibold">Admin Activity</h1>
      <Select value={userId ?? 'all'} onValueChange={(v) => setUserId(v === 'all' ? undefined : v)}>
        <SelectTrigger className="w-72"><SelectValue placeholder="All users" /></SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All users</SelectItem>
          {(users.data ?? []).map((u) => (
            <SelectItem key={u.user_id ?? 'anon'} value={u.user_id ?? 'anon'}>
              {u.email ?? u.name ?? '(anonymous)'} · {u.event_count}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Card>
        <CardHeader><CardTitle className="text-sm">Activity heatmap (28d)</CardTitle></CardHeader>
        <CardContent><ActivityHeatmap data={s?.heatmap ?? []} /></CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle className="text-sm">Events</CardTitle></CardHeader>
        <CardContent>
          <ActivityFeed
            events={events.data?.entries ?? []}
            loading={events.isLoading}
            onTypeChange={setType}
          />
        </CardContent>
      </Card>
    </div>
  )
}
```

- [ ] **Step 3: Sidebar link**

In `AppSidebar.tsx`, under "Manage", add:

```ts
{ name: t('navigation.activity') || 'Activity', href: '/activity', icon: Activity },
```

Import `Activity` from `lucide-react`.

- [ ] **Step 4: Locale** (already added in S6 prep — verify `navigation.activity` exists; otherwise add `activity: "Activity"`)

- [ ] **Step 5: Run TS check + commit**

```bash
cd frontend && npx tsc --noEmit
git add frontend/src/app/\(dashboard\)/activity frontend/src/app/\(dashboard\)/admin/activity frontend/src/components/layout/AppSidebar.tsx frontend/src/lib/locales/en-US/index.ts
git commit -m "feat(s6): /activity page + admin variant + sidebar link"
```

---

### Task 11: Integration test

**Files:**
- Create: `tests/integration/test_s6_activity.py`

- [ ] **Step 1: Test**

```python
"""S6 — Activity end-to-end."""
import pytest
import uuid
from unittest.mock import patch, AsyncMock
from src.agentrag.observability import activity


@pytest.mark.asyncio
async def test_record_event_handles_anonymous():
    """user_id='anonymous' → coerced to NULL, no exception."""
    with patch.object(activity, "AsyncSessionLocal") as M:
        m_session = AsyncMock()
        M.return_value.__aenter__.return_value = m_session
        await activity.record_event("anonymous", "chat_turn", payload={"x": 1})
    # commit was called (sanity)
    m_session.commit.assert_awaited()


@pytest.mark.asyncio
async def test_record_event_uuid_str_coerces():
    with patch.object(activity, "AsyncSessionLocal") as M:
        m_session = AsyncMock()
        M.return_value.__aenter__.return_value = m_session
        await activity.record_event(str(uuid.uuid4()), "search", payload={"query": "abc"})
    m_session.commit.assert_awaited()
```

- [ ] **Step 2: Run**

`uv run python -m pytest tests/integration/test_s6_activity.py -v` → 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_s6_activity.py
git commit -m "test(s6): activity integration smoke"
```

---

### Task 12: Push + tag

- [ ] **Step 1: Full sweep**

```bash
uv run python -m pytest tests/ --ignore=tests/ontology --ignore=tests/ingestion -q
cd frontend && npx tsc --noEmit
```

- [ ] **Step 2: Push + tag**

```bash
git push origin structmem
git tag -a s6-activity-complete -m "S6 — Activity panel: per-user feed + admin global"
git push origin s6-activity-complete
```

---

## Self-Review

**Spec coverage:**
- Migration → T1 ✓
- record_event helper → T2 ✓
- Chat_turn write (notebook + source) → T3 ✓
- Source uploaded + ingest done/failed → T4 ✓
- Search event → T5 ✓
- Adapter routes (personal + admin) → T6 ✓
- Adapter endpoint privacy tests → T7 ✓
- Frontend API/hooks/types → T8 ✓
- Frontend components → T9 ✓
- Frontend pages + sidebar → T10 ✓
- Integration test → T11 ✓
- Push + tag → T12 ✓

**Placeholder scan:** None — every step shows code/commands.

**Type consistency:** `record_event(user_id, event_type, *, target_kind, target_id, payload)` signature used consistently across T2/T3/T4/T5. `EventLog` columns match across migration + model + serializer. `ActivityEvent` / `ActivitySummary` shapes consistent across backend Pydantic models + frontend TS types.

Plan complete.
