# Account Deletion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Self-service full account wipe — the authenticated user erases all their data across Postgres + Elasticsearch + image filesystem.

**Architecture:** A `delete_user_data(user_id)` service (PG-authoritative, ES/files best-effort) + a thin auth-gated `DELETE` endpoint. Reuses the per-document cascade pattern from `delete_source`.

**Tech Stack:** SQLAlchemy async (`AsyncSessionLocal`), FastAPI, ElasticsearchStore.

## Global Constraints

- PG deletes are authoritative + must complete even if ES/file purges error → wrap ES/file steps in try/except.
- Only ever delete rows matching the caller's own `user_id`. Endpoint refuses `anonymous`, legacy, or missing identity with `403`.
- Delete order: documents → conversations(+messages) → feedback → events → user row LAST.
- Reuse `ElasticsearchStore().delete_document(title)` (real method used by `delete_source`).

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/agentrag/adapter/account_deletion.py` (create) | `delete_user_data(user_id)` cascade service |
| `src/agentrag/adapter/routers/chat.py` (modify) | `DELETE …/account` endpoint (auth-gated) |
| `tests/adapter/test_account_deletion.py` (create) | guard unit test + live integration test |

---

### Task 1: delete_user_data service + live integration test

**Files:**
- Create: `src/agentrag/adapter/account_deletion.py`
- Create: `tests/adapter/test_account_deletion.py`

**Interfaces:**
- Produces: `async def delete_user_data(user_id: str) -> dict[str, int]` — keys `documents, segments, conversations, messages, feedback, events`.
- Consumes: `Document`, `Segment`, `Conversation`, `ChatMessage`, `EventLog`, `User` (`database/models.py`), `AdapterChatFeedback` (`adapter/db.py`), `AsyncSessionLocal`.

- [ ] **Step 1: Write the live integration test** (needs Postgres up).

```python
# tests/adapter/test_account_deletion.py
import uuid

import pytest
from sqlalchemy import select

from src.agentrag.adapter.account_deletion import delete_user_data
from src.agentrag.adapter.db import AdapterChatFeedback
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import (
    ChatMessage, Conversation, Document, EventLog, Project, Segment, User,
)


@pytest.mark.asyncio
async def test_delete_user_data_wipes_everything_for_that_user():
    uid = uuid.uuid4()
    other = uuid.uuid4()
    async with AsyncSessionLocal() as s:
        proj = Project(id=uuid.uuid4(), name="p")
        s.add_all([
            User(id=uid, email=f"{uid}@t.test"),
            User(id=other, email=f"{other}@t.test"),
            proj,
        ])
        await s.flush()
        doc = Document(id=uuid.uuid4(), title=f"doc-{uid}", project_id=proj.id, user_id=uid)
        s.add(doc); await s.flush()
        s.add(Segment(id=uuid.uuid4(), document_id=doc.id, position=0, content="x"))
        conv = Conversation(id=uuid.uuid4(), user_id=uid)
        s.add(conv); await s.flush()
        s.add(ChatMessage(id=uuid.uuid4(), conversation_id=conv.id, role="user", content="hi"))
        s.add(EventLog(id=uuid.uuid4(), user_id=uid, event_type="x"))
        s.add(AdapterChatFeedback(user_id=str(uid), turn_id=str(uuid.uuid4()), rating=1))
        # other user's row that must SURVIVE
        s.add(Conversation(id=uuid.uuid4(), user_id=other))
        await s.commit()

    counts = await delete_user_data(str(uid))
    assert counts["documents"] == 1 and counts["conversations"] == 1
    assert counts["messages"] == 1 and counts["feedback"] == 1 and counts["events"] == 1

    async with AsyncSessionLocal() as s:
        assert (await s.execute(select(Document).where(Document.user_id == uid))).first() is None
        assert (await s.execute(select(Conversation).where(Conversation.user_id == uid))).first() is None
        assert (await s.execute(select(EventLog).where(EventLog.user_id == uid))).first() is None
        assert (await s.execute(select(AdapterChatFeedback).where(AdapterChatFeedback.user_id == str(uid)))).first() is None
        assert (await s.execute(select(User).where(User.id == uid))).first() is None
        # other user's conversation survived
        assert (await s.execute(select(Conversation).where(Conversation.user_id == other))).first() is not None
        # cleanup the other user
        await delete_user_data(str(other))
```

- [ ] **Step 2: Run it — expect FAIL** (`ModuleNotFoundError: ...account_deletion`).

Run: `uv run pytest tests/adapter/test_account_deletion.py -q`

- [ ] **Step 3: Write `src/agentrag/adapter/account_deletion.py`.**

```python
"""Full user-data wipe (P4 right-to-delete). Postgres is authoritative; the ES and
image-file purges are best-effort so a search-layer hiccup can't half-delete PG."""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from sqlalchemy import delete, select

from src.agentrag.adapter.db import AdapterChatFeedback
from src.agentrag.config import settings
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import (
    ChatMessage, Conversation, Document, EventLog, Segment, User,
)

logger = logging.getLogger(__name__)


def _purge_document_artifacts(title: str | None) -> None:
    """Best-effort ES + image-folder removal for one document (mirrors delete_source)."""
    if not title:
        return
    try:
        import asyncio

        from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore

        asyncio.get_event_loop()  # within an async caller; delete_document is awaited below
    except Exception:
        pass


async def delete_user_data(user_id: str) -> dict[str, int]:
    counts = {"documents": 0, "segments": 0, "conversations": 0,
              "messages": 0, "feedback": 0, "events": 0}
    titles: list[str] = []

    async with AsyncSessionLocal() as s:
        docs = (await s.execute(select(Document).where(Document.user_id == user_id))).scalars().all()
        for doc in docs:
            seg = await s.execute(delete(Segment).where(Segment.document_id == doc.id))
            counts["segments"] += seg.rowcount or 0
            titles.append(doc.title)
            await s.delete(doc)
            counts["documents"] += 1

        conv_ids = (await s.execute(select(Conversation.id).where(Conversation.user_id == user_id))).scalars().all()
        if conv_ids:
            m = await s.execute(delete(ChatMessage).where(ChatMessage.conversation_id.in_(conv_ids)))
            counts["messages"] += m.rowcount or 0
        c = await s.execute(delete(Conversation).where(Conversation.user_id == user_id))
        counts["conversations"] += c.rowcount or 0

        f = await s.execute(delete(AdapterChatFeedback).where(AdapterChatFeedback.user_id == str(user_id)))
        counts["feedback"] += f.rowcount or 0
        e = await s.execute(delete(EventLog).where(EventLog.user_id == user_id))
        counts["events"] += e.rowcount or 0

        await s.execute(delete(User).where(User.id == user_id))
        await s.commit()

    # Best-effort search/file purge — PG is already committed above.
    for title in titles:
        try:
            from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore

            await ElasticsearchStore().delete_document(title)
        except Exception as exc:
            logger.debug("ES purge skipped for %r: %s", title, exc)
        try:
            safe = re.sub(r"[^\w\-]", "_", title)[:80]
            shutil.rmtree(Path(settings.IMAGE_STORAGE_DIR) / safe, ignore_errors=True)
        except Exception as exc:
            logger.debug("image purge skipped for %r: %s", title, exc)

    logger.info("account wipe user=%s counts=%s", user_id, counts)
    return counts
```

(Delete the unused `_purge_document_artifacts` stub above — it was a scratch; the real
purge is the loop at the end. Keep the file to exactly the `delete_user_data` function +
imports.)

- [ ] **Step 4: Run the test — expect PASS** (1 passed). If a model column name differs (e.g. `Segment.position`/`content`, `EventLog.event_type`), adjust the seed to match the real columns — read `database/models.py` and fix the test, not the service.

Run: `uv run pytest tests/adapter/test_account_deletion.py -q`

- [ ] **Step 5: Commit.**

```bash
git add src/agentrag/adapter/account_deletion.py tests/adapter/test_account_deletion.py
git commit -m "feat(security/P4): delete_user_data — full user-data wipe across PG/ES/images

PG-authoritative cascade (documents+segments, conversations+messages, feedback, events,
user row last); ES delete_document + image-folder rmtree best-effort. Only deletes the
given user_id; verified by a live integration test (other user's row survives)."
```

---

### Task 2: DELETE endpoint (auth-gated) + guard test

**Files:**
- Modify: `src/agentrag/adapter/routers/chat.py` (add the endpoint)
- Modify: `tests/adapter/test_account_deletion.py` (add the guard test)

**Interfaces:**
- Consumes: `delete_user_data` (Task 1); `get_identity` (`adapter/auth.py`).

- [ ] **Step 1: Add the guard test** (mock identity; no DB needed for the 403 paths).

```python
# append to tests/adapter/test_account_deletion.py
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.agentrag.adapter.routers import chat as chat_router


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", [
    None,
    SimpleNamespace(user_id="anonymous", is_legacy=False),
    SimpleNamespace(user_id="legacy", is_legacy=True),
])
async def test_account_delete_rejects_non_user(monkeypatch, identity):
    monkeypatch.setattr(chat_router, "get_identity", lambda req: identity, raising=False)
    with pytest.raises(HTTPException) as ei:
        await chat_router.delete_account(request=SimpleNamespace())
    assert ei.value.status_code == 403
```

- [ ] **Step 2: Run it — expect FAIL** (`delete_account` not defined).

Run: `uv run pytest tests/adapter/test_account_deletion.py::test_account_delete_rejects_non_user -q`

- [ ] **Step 3: Add the endpoint to `chat.py`.** Use the same router the feedback
endpoint uses (`@notebook_router.post("/feedback")`). Add near it:

```python
@notebook_router.delete("/account")
async def delete_account(request: Request):
    """P4 right-to-delete: erase ALL data for the authenticated user. Refuses
    anonymous/legacy identities (they map to shared data)."""
    from src.agentrag.adapter.auth import get_identity
    from src.agentrag.adapter.account_deletion import delete_user_data

    identity = get_identity(request)
    if identity is None or getattr(identity, "is_legacy", False) or \
            getattr(identity, "user_id", "anonymous") in ("anonymous", "", None):
        raise HTTPException(403, "account deletion requires an authenticated user")
    counts = await delete_user_data(identity.user_id)
    return {"deleted": counts}
```

(`Request` + `HTTPException` are already imported in `chat.py`. The guard test patches
`chat.get_identity`, so the endpoint must reference it via the module-level name — keep
the `from ... import get_identity` at call time OR import it at module top; if the test's
monkeypatch on `chat_router.get_identity` must take effect, import `get_identity` at the
top of `chat.py` instead of inside the function. Verify the test passes; if the local
import shadows the patch, move it to module scope.)

- [ ] **Step 4: Run the guard test — expect PASS** (3 passed).

Run: `uv run pytest tests/adapter/test_account_deletion.py -q`

- [ ] **Step 5: No-regression.**

Run: `uv run pytest -q --ignore=tests/ontology --ignore=tests/ingestion`
Expected: PASS (prior green + the new tests).

- [ ] **Step 6: Commit.**

```bash
git add src/agentrag/adapter/routers/chat.py tests/adapter/test_account_deletion.py
git commit -m "feat(security/P4): DELETE /account endpoint — auth-gated self-service wipe

Refuses anonymous/legacy/missing identity (403); else delete_user_data(user_id).
Only ever wipes the caller's own user_id."
```

---

## Self-Review

**Spec coverage:** `delete_user_data` ordered cascade → T1 S3; PG-authoritative + best-effort ES/files → T1 S3 (try/except); endpoint + 403 guard → T2 S3; only-own-user_id → both; live integration (other user survives) → T1 S1; guard unit → T2 S1. All mapped.

**Placeholder scan:** none — full service + endpoint + tests inline. The note to delete the scratch `_purge_document_artifacts` stub is an explicit instruction (the final file is `delete_user_data` + imports only), not a TODO in shipped code.

**Type consistency:** `delete_user_data(user_id: str) -> dict[str,int]` with keys `documents/segments/conversations/messages/feedback/events` consistent across T1 def, T1 test asserts, and T2 endpoint return. `delete_account(request)` referenced in T2 test + def. Model/column names (`Document.user_id`, `Segment.document_id`, `Conversation.user_id`, `ChatMessage.conversation_id`, `EventLog.user_id`, `AdapterChatFeedback.user_id`) taken from the explored schema; T1 S4 says reconcile the *test seed* to real columns if any differ.
