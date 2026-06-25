import uuid

import pytest
from sqlalchemy import select

from src.agentrag.adapter.account_deletion import delete_user_data
from src.agentrag.adapter.db import AdapterChatFeedback
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import (
    ChatMessage,
    Conversation,
    Document,
    EventLog,
    Project,
    Segment,
    User,
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
        doc = Document(id=uuid.uuid4(), title=f"doc-{uid}", source_type="markdown",
                       project_id=proj.id, user_id=uid)
        s.add(doc)
        await s.flush()
        s.add(Segment(id=uuid.uuid4(), document_id=doc.id, position=0, content="x"))
        conv = Conversation(id=uuid.uuid4(), user_id=uid)
        s.add(conv)
        await s.flush()
        s.add(ChatMessage(id=uuid.uuid4(), conversation_id=conv.id, role="user", content="hi"))
        s.add(EventLog(id=uuid.uuid4(), user_id=uid, event_type="x"))
        s.add(AdapterChatFeedback(user_id=str(uid), turn_id=str(uuid.uuid4()), rating=1))
        s.add(Conversation(id=uuid.uuid4(), user_id=other))  # must SURVIVE
        await s.commit()

    counts = await delete_user_data(str(uid))
    assert counts["documents"] == 1 and counts["conversations"] == 1
    assert counts["messages"] == 1 and counts["feedback"] == 1 and counts["events"] == 1

    async with AsyncSessionLocal() as s:
        assert (await s.execute(select(Document).where(Document.user_id == uid))).first() is None
        assert (await s.execute(select(Conversation).where(Conversation.user_id == uid))).first() is None
        assert (await s.execute(select(EventLog).where(EventLog.user_id == uid))).first() is None
        assert (await s.execute(
            select(AdapterChatFeedback).where(AdapterChatFeedback.user_id == str(uid))
        )).first() is None
        assert (await s.execute(select(User).where(User.id == uid))).first() is None
        # the OTHER user's conversation survived
        assert (await s.execute(select(Conversation).where(Conversation.user_id == other))).first() is not None

    await delete_user_data(str(other))  # cleanup


from types import SimpleNamespace

from fastapi import HTTPException

from src.agentrag.adapter.routers import chat as chat_router


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", [
    None,
    SimpleNamespace(user_id="anonymous", is_legacy=False),
    SimpleNamespace(user_id="legacy-user", is_legacy=True),
])
async def test_account_delete_rejects_non_user(monkeypatch, identity):
    monkeypatch.setattr(chat_router, "get_identity", lambda req: identity)
    with pytest.raises(HTTPException) as ei:
        await chat_router.delete_account(request=SimpleNamespace())
    assert ei.value.status_code == 403
