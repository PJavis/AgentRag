# Activity Panel — Personal + Admin Feed (S6)

> Status: spec — approved 2026-05-16. Replaces the static `/admin` HTML
> trace inspector with a per-user activity timeline backed by a real
> event log. Admin retains a global view at `/admin/activity`.

## Goals

1. Each user sees own activity (chat turns, source uploads + ingest
   results, search executions) at `/activity`.
2. Admin (User.is_admin OR X-Admin-Token) sees a global feed at
   `/admin/activity` with per-user drill-down.
3. Reuse the existing S2 `TraceDialog` for chat-turn drill-down — zero
   new graph rendering code.
4. Privacy enforced at SQL layer. Anonymous events bucketed separately
   and only visible to admin.

## Non-goals

- Notebook / note CRUD / login events (excluded during scoping).
- Async event recording (sync inline INSERT chosen for simplicity).
- Retention / archival policy.
- Dedicated full-canvas graph view for events (TraceDialog covers it).

## Data model

**Migration `2026051601_add_user_id_and_event_log`:**

```sql
ALTER TABLE conversations ADD COLUMN user_id UUID NULL REFERENCES users(id);
CREATE INDEX ix_conversations_user_id ON conversations(user_id);

ALTER TABLE documents ADD COLUMN user_id UUID NULL REFERENCES users(id);
CREATE INDEX ix_documents_user_id ON documents(user_id);

CREATE TABLE event_log (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NULL REFERENCES users(id),   -- NULL = anonymous bucket
  event_type  VARCHAR(32) NOT NULL,             -- chat_turn|source_uploaded|ingest_done|ingest_failed|search
  target_kind VARCHAR(32) NULL,                 -- conversation|document|chat_message|null
  target_id   UUID NULL,
  payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_event_log_user_created    ON event_log(user_id, created_at DESC);
CREATE INDEX ix_event_log_type_created    ON event_log(event_type, created_at DESC);
CREATE INDEX ix_event_log_target_id       ON event_log(target_id) WHERE target_id IS NOT NULL;
```

Backfill: leave existing `conversations.user_id` / `documents.user_id`
NULL. They surface only to admin.

**Python model** in `src/agentrag/database/models.py`:

```python
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

## Write path

Module `src/agentrag/observability/activity.py`:

```python
async def record_event(
    user_id: uuid.UUID | None,
    event_type: str,
    *,
    target_kind: str | None = None,
    target_id: uuid.UUID | None = None,
    payload: dict | None = None,
) -> None:
    """Sync INSERT into event_log. Caller awaits; failure logs + raises None."""
```

| Call site | event_type | payload |
|---|---|---|
| `adapter/routers/chat.py::execute_chat` | `chat_turn` | `{message_id, tokens_in, tokens_out, latency_ms, reasoning_path, model}` |
| `adapter/routers/chat.py::send_message` (SSE) | `chat_turn` | same |
| `adapter/routers/sources.py::create_source` | `source_uploaded` | `{filename, size_bytes, content_hash, source_type}` |
| ingestion pipeline finish callback | `ingest_done` / `ingest_failed` | `{segment_count, duration_ms, error?}` |
| `search.py::search_handler` (and adapter equivalent) | `search` | `{query, mode, top_k, hit_count, latency_ms, document_title?}` |

`record_event` runs inline (await) inside each handler. Adds 1 INSERT
per chat turn / upload / search. Failure swallowed via try/except so
business path never breaks; logged via `logger.warning` instead.

Also retroactive (not in scope): `conversations.user_id` written when
`create_conversation` runs — pull from `request.state.auth_identity`.
`documents.user_id` set in `create_source`.

## Read path

Adapter routes (`src/agentrag/adapter/routers/activity.py` — NEW):

```
GET /api/activity/summary
    {
      "counts": {"chat_turn": 42, "source_uploaded": 12, "search": 87},
      "tokens_total": 124_500,
      "usd_estimate": 0.42,
      "heatmap": [{"date": "2026-05-16", "count": 12}, …]    # last 28 days
    }

GET /api/activity/events?type=&limit=20&before_id=
    {"entries": [Event…], "next_before_id": "…"}

GET /api/activity/events/{event_id}
    {Event…}  # 404 if not owner

GET /api/admin/activity/summary?user_id=
GET /api/admin/activity/events?user_id=&type=&limit=&before_id=
GET /api/admin/activity/users
    [{user_id, email, name, event_count, last_seen}, …]
```

Auth filter pattern (helper `_scope_to_identity(query, identity)`):

```python
if not identity or identity.user_id == "anonymous":
    return None              # 401 — Activity requires login
return query.where(EventLog.user_id == uuid.UUID(identity.user_id))
```

Admin routes require `is_admin(request)` (existing helper). Mount under
`/api/admin/activity` (NOT in `_PUBLIC_PREFIXES`).

USD estimate computed from `payload.tokens_in/out` against the same
price table used by `observability/cost.py`. Surfaced for completeness;
admin sees global total.

## UI (frontend)

**Route:** `frontend/src/app/(dashboard)/activity/page.tsx`

**Layout** (Layout C from brainstorm):

```
┌────────────────────────────────────────────────────────────────────┐
│ Activity                                          [Reset] [Refresh]│
│ Your recent chats, uploads, and searches.                          │
├────────────────────────────────────────────────────────────────────┤
│ [42 chats] [12 uploads] [87 searches] [$0.42 spent]                │
├────────────────────────────────────────────────────────────────────┤
│  Mon Tue Wed Thu Fri Sat Sun   ← 28-day heatmap                    │
│  ▢ ▣ ▣ ▣ ▢ ▢ ▢ …                                                  │
├────────────────────────────────────────────────────────────────────┤
│ Recent  [All] [Chat] [Upload] [Search]               20/50/100      │
│ ──────────────────────────────────────────────────────────────────  │
│ 14:02  chat   medical-qa session · 12 turns · $0.0012     [Trace]  │
│ 13:40  upload giai_phau.pdf · 2.1 MB · ingest done                  │
│ 13:35  search "van hai lá" · hybrid · 8 hits · 240ms                │
│ …                                                                   │
└────────────────────────────────────────────────────────────────────┘
```

**Components**

- `frontend/src/lib/api/activity.ts` — `summary()`, `events({type, limit, before_id})`, `event(id)`, admin variants.
- `frontend/src/lib/hooks/useActivity.ts` — TanStack Query wrappers; 30s `refetchInterval` on summary.
- `frontend/src/components/activity/ActivityHeatmap.tsx` — 7-row × 4-col cell grid, color scale by count.
- `frontend/src/components/activity/ActivityFeed.tsx` — virtualized list of `EventRow`. Per-type icon + click handler.
- `frontend/src/components/activity/EventRow.tsx` — one row, type-specific renderer + click → action.
- `frontend/src/app/(dashboard)/admin/activity/page.tsx` — same components, fed by admin endpoints + user-picker dropdown.

**Click behaviour**

| Event type | Click action |
|---|---|
| `chat_turn` | Fetch assistant message_id from `payload.message_id` → open existing `TraceDialog` (same component used in ChatPanel). Need helper to load message by id; reuse `chatApi.getMessage`. |
| `source_uploaded` | Navigate to `/sources/{target_id}`. |
| `ingest_done` / `ingest_failed` | Navigate to `/sources/{target_id}`. |
| `search` | Toast showing query string + mode. |

**Sidebar link**

`AppSidebar.tsx` — add under "Manage":
```
{ name: t('navigation.activity'), href: '/activity', icon: Activity }
```

Admin route hidden by default; surface via header dropdown when
`identity.is_admin === true`.

## Privacy / Security

- Personal routes: SQL `WHERE user_id = identity.user_id`. No user_id query param accepted from client.
- Anonymous identity (`user_id == "anonymous"`) → 401. No anonymous activity feed.
- Admin routes: middleware `is_admin(request)`. Returns 403 otherwise.
- `payload` contents may include user text (search queries, chat content
  references) — admin viewing global feed sees this. Acceptable per
  brainstorm (admin = high-trust).

## Testing

- `tests/observability/test_activity.py` — `record_event()` writes row;
  failure logs but does not raise.
- `tests/adapter/test_activity_endpoints.py` — summary / events / admin
  routes. Privacy: user A cannot read user B events. Anonymous → 401.
  Admin (header token) bypasses scope.
- `tests/integration/test_s6_activity.py` — end-to-end:
  - send a notebook chat → 1 `chat_turn` row written
  - upload a source → 1 `source_uploaded` row, then ingest finish → 1
    `ingest_done` row
  - `GET /api/activity/summary` returns expected counts

## Migration plan

1. Alembic migration (model + columns + indexes).
2. `observability/activity.py` helper.
3. Wire `record_event()` into chat / source / search / ingest finish.
4. New adapter router + mount on `/api` and `/api/admin`.
5. Frontend page + hooks + components + sidebar link.
6. Test sweep.
7. Commit + tag `s6-activity-complete`. Drop static `/admin` HTML page
   from sidebar nav (route stays as legacy until S7).

## Acceptance

1. Send a notebook chat as user A → `event_log` row appears with
   `event_type=chat_turn`, `user_id=A.id`. `/activity` page shows it.
2. User B does not see A's event in `/activity`.
3. Admin token holder hits `/api/admin/activity/events` → both rows
   returned with their `user_id`.
4. Clicking the chat row opens TraceDialog with the persisted trace.
5. Anonymous user (no auth) → 401 from `/api/activity/*`.

## Out of scope (S7+)

- ARQ async event recording (only revisit if INSERTs add measurable p95
  latency).
- Notebook / note CRUD events.
- Login / signout audit log.
- Retention / pruning policy for event_log.
- Export to CSV / JSON.
- WebSocket live tail.
