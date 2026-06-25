# adapter_chat_feedback migration — design

**Date:** 2026-06-25 · **Objective:** make the feedback-capture table authoritative
in alembic so the (already-built) feedback → finetune pipeline isn't silently
starved on migration-only deploys.

## Context

The thumbs up/down feedback pipeline is built end-to-end:
- UI `frontend/src/components/source/FeedbackButtons.tsx` → `POST /chat/feedback`
- endpoint `adapter/routers/chat.py::submit_chat_feedback` upserts into the
  `AdapterChatFeedback` ORM (`adapter/db.py:98`)
- miners read it: `scripts/mine_finetune_pairs.py` (embedding/reranker triplets from
  `rating=1` + `chat_messages.tool_trace`), `scripts/mine_sft.py` (SFT excluding
  `rating=-1`).

**Audit (2026-06-25):** of all ORM tables, `adapter_chat_feedback` is the **only** one
with no alembic migration. It exists solely via the `create_all(checkfirst=True)`
safety-net in `adapter/db.py:118` (`create_adapter_tables()`). The project rule is
"Alembic là nguồn sự thật của schema, không phải `create_all`"
(`database/README.md:110`). On a migration-only deploy (`alembic upgrade head`, no
create_all) the table is missing → `/chat/feedback` fails → zero training data.

## Scope

In: one new alembic migration creating `adapter_chat_feedback`, matching the ORM
exactly, with up + down. Out: DPO/ORPO preference-pair miner (defer until real
feedback accumulates); backfilling retrieved-context into feedback rows.

## Design

**File:** `migrations/versions/2026062501_add_adapter_chat_feedback.py`
- `revision = "2026062501"`, `down_revision = "2026060501"` (current linear head).

**upgrade()** — create `adapter_chat_feedback` matching `AdapterChatFeedback` exactly:

| column | type | notes |
|---|---|---|
| id | `UUID` | primary key (server can default; ORM defaults `uuid4`) |
| user_id | `String(128)` | NOT NULL, indexed |
| conversation_id | `String(128)` | nullable, indexed |
| turn_id | `String(128)` | NOT NULL, indexed |
| rating | `Integer` | NOT NULL (+1/−1) |
| comment | `Text` | nullable |
| question | `Text` | nullable |
| answer | `Text` | nullable |
| reasoning_path | `String(32)` | nullable |
| extra_metadata | `JSONB` | nullable |
| created_at | `DateTime(timezone=True)` | `server_default=now()` |
| updated_at | `DateTime(timezone=True)` | `server_default=now()` |

Indexes: `ix_adapter_chat_feedback_user_id`, `…_conversation_id`, `…_turn_id`
(matching `index=True` on those columns). Guard against the create_all net having
already made the table: check `inspector.has_table(...)` and no-op if present (so
`upgrade` is safe whether or not the safety-net ran first).

**downgrade()** — drop the three indexes + `drop_table("adapter_chat_feedback")`.

## Data flow

Unchanged. Endpoint and miners already target the table; the migration only makes
the schema versioned + reproducible on clean deploys.

## Testing

Docker/Postgres is **not available on this host**, so:
- **Offline (runs here):** `uv run alembic history` must show the new revision
  chaining off `2026060501` with no branches. A unit test
  (`tests/migrations/test_feedback_migration_parity.py`) imports the
  `AdapterChatFeedback` ORM and the migration module, and asserts the migration's
  `create_table` column names == the ORM's column names (drift guard, no DB needed).
- **Live (CI):** `ci.yml` has a postgres service and runs `alembic upgrade head` on
  PR — real up-migration validation. Down-migration is exercised by the parity test's
  structural check (a full up/down round-trip is a CI follow-up if desired).

## Error handling

`has_table` guard makes `upgrade` idempotent vs the create_all net. `downgrade` drops
indexes before the table. No data migration (new table).
