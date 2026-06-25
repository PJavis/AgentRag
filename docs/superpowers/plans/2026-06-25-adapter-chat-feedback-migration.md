# adapter_chat_feedback Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the missing alembic migration for `adapter_chat_feedback` so the
already-built feedback→finetune pipeline persists on migration-only deploys.

**Architecture:** One new alembic revision chaining off the current head
(`2026060501`), creating `adapter_chat_feedback` to match the `AdapterChatFeedback`
ORM exactly, with a `has_table` idempotency guard (the `create_all` safety-net may
have already created it in dev). Two offline parity tests guard against drift.

**Tech Stack:** Alembic, SQLAlchemy (pydantic-free), Postgres (JSONB/UUID), pytest.

## Global Constraints

- Migration must match the ORM `AdapterChatFeedback` (`src/agentrag/adapter/db.py:98`)
  column-for-column: `id`(UUID pk), `user_id`(String(128) NOT NULL), `conversation_id`
  (String(128) null), `turn_id`(String(128) NOT NULL), `rating`(Integer NOT NULL),
  `comment`/`question`/`answer`(Text null), `reasoning_path`(String(32) null),
  `extra_metadata`(JSONB null), `created_at`/`updated_at`(DateTime tz, server_default now()).
- Indexes on `user_id`, `conversation_id`, `turn_id` named `ix_adapter_chat_feedback_<col>`.
- `revision = "2026062501"`, `down_revision = "2026060501"` (current single head).
- Docker/Postgres is NOT available on this host → validate offline (pytest + `alembic history`); live up-migration runs in CI (`ci.yml`).
- `upgrade` must be idempotent vs the `create_all(checkfirst=True)` net (`adapter/db.py:118`).

---

## File Structure

| Path | Responsibility |
|---|---|
| `migrations/versions/2026062501_add_adapter_chat_feedback.py` (create) | the migration (up/down) |
| `tests/migrations/test_feedback_migration_parity.py` (create) | offline drift guards: ORM↔migration column parity + single-head chain |
| `tests/migrations/__init__.py` (create if absent) | package marker |

---

### Task 1: adapter_chat_feedback migration + parity tests

**Files:**
- Create: `migrations/versions/2026062501_add_adapter_chat_feedback.py`
- Create: `tests/migrations/test_feedback_migration_parity.py`
- Create (if absent): `tests/migrations/__init__.py`

**Interfaces:**
- Consumes: `src.agentrag.adapter.db.AdapterChatFeedback` (ORM, `__table__.columns`).
- Produces: alembic revision `2026062501` (new single head).

- [ ] **Step 1: Write the failing parity tests.**

```python
# tests/migrations/test_feedback_migration_parity.py
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

from src.agentrag.adapter.db import AdapterChatFeedback

_MIGRATION = Path("migrations/versions/2026062501_add_adapter_chat_feedback.py")


def test_migration_covers_every_orm_column():
    """Every AdapterChatFeedback ORM column must appear in the migration (drift guard)."""
    src = _MIGRATION.read_text(encoding="utf-8")
    for col in AdapterChatFeedback.__table__.columns:
        assert f'"{col.name}"' in src, f"migration missing column {col.name}"


def test_single_head_is_new_revision():
    """The revision graph stays linear and the new revision is the sole head."""
    script = ScriptDirectory.from_config(Config("alembic.ini"))
    assert script.get_heads() == ("2026062501",)
    rev = script.get_revision("2026062501")
    assert rev.down_revision == "2026060501"
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `uv run pytest tests/migrations/test_feedback_migration_parity.py -v`
Expected: FAIL — `test_migration_covers_every_orm_column` errors on
`FileNotFoundError` (migration absent), `test_single_head_is_new_revision` fails on
`get_heads() == ("2026060501",)`. (Create `tests/migrations/__init__.py` first if
collection complains about the package.)

- [ ] **Step 3: Write the migration.**

```python
# migrations/versions/2026062501_add_adapter_chat_feedback.py
"""add adapter_chat_feedback

Revision ID: 2026062501
Revises: 2026060501
Create Date: 2026-06-25
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

revision: str = "2026062501"
down_revision: Union[str, Sequence[str], None] = "2026060501"
branch_labels = None
depends_on = None

TABLE = "adapter_chat_feedback"


def upgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table(TABLE):
        return  # create_all(checkfirst=True) safety-net already created it
    op.create_table(
        TABLE,
        sa.Column("id", PG_UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("conversation_id", sa.String(128), nullable=True),
        sa.Column("turn_id", sa.String(128), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("reasoning_path", sa.String(32), nullable=True),
        sa.Column("extra_metadata", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_adapter_chat_feedback_user_id", TABLE, ["user_id"])
    op.create_index("ix_adapter_chat_feedback_conversation_id", TABLE, ["conversation_id"])
    op.create_index("ix_adapter_chat_feedback_turn_id", TABLE, ["turn_id"])


def downgrade() -> None:
    op.drop_index("ix_adapter_chat_feedback_turn_id", table_name=TABLE)
    op.drop_index("ix_adapter_chat_feedback_conversation_id", table_name=TABLE)
    op.drop_index("ix_adapter_chat_feedback_user_id", table_name=TABLE)
    op.drop_table(TABLE)
```

- [ ] **Step 4: Run the tests to verify they pass.**

Run: `uv run pytest tests/migrations/test_feedback_migration_parity.py -v`
Expected: PASS (2 passed). Both run offline — no Postgres needed.

- [ ] **Step 5: Validate the revision graph end-to-end (offline).**

Run: `uv run alembic history | head -3`
Expected: top line shows `2026060501 -> 2026062501 (head)`.
Run: `uv run alembic heads`
Expected: `2026062501 (head)` — single head, no branches.

- [ ] **Step 6: Confirm no broader regression.**

Run: `uv run pytest -q --ignore=tests/ontology --ignore=tests/ingestion`
Expected: PASS (the prior green count + 2 new = green; no collection errors from the new `tests/migrations` package).

- [ ] **Step 7: Commit.**

```bash
git add migrations/versions/2026062501_add_adapter_chat_feedback.py \
        tests/migrations/test_feedback_migration_parity.py tests/migrations/__init__.py
git commit -m "feat(db): alembic migration for adapter_chat_feedback (was create_all-only)

Audit found adapter_chat_feedback is the only ORM table with no migration — it
existed solely via the create_all safety-net, so migration-only deploys dropped the
feedback table and starved the finetune miners. Migration matches the ORM exactly,
idempotent vs create_all via has_table guard. Offline parity tests guard drift; CI
runs the live up-migration."
```

---

## Self-Review

**Spec coverage:** spec's single deliverable (the migration matching the ORM + offline
tests + idempotency guard) → Task 1 Steps 3 (migration), 1 (parity test), 3 (`has_table`).
CI live-migration is the existing `ci.yml` (no new task). All spec sections covered.

**Placeholder scan:** none — full migration + test code inline, exact revision ids,
exact column types from the ORM, exact commands with expected output.

**Type consistency:** `revision`/`down_revision` strings (`2026062501`/`2026060501`)
match across the migration, the spec, and `test_single_head_is_new_revision`. Index
names `ix_adapter_chat_feedback_<col>` consistent between `upgrade`/`downgrade`. Column
names sourced from `AdapterChatFeedback.__table__.columns` so the parity test can't
drift from the ORM.
