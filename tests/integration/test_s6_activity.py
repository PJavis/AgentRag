"""S6 — Activity end-to-end (mock DB layer)."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_record_event_handles_anonymous():
    """user_id='anonymous' coerces to NULL; commit still called."""
    from src.agentrag.observability import activity

    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()

    @asynccontextmanager
    async def fake_sess():
        yield session

    with patch.object(activity, "AsyncSessionLocal", fake_sess):
        await activity.record_event("anonymous", "chat_turn", payload={"x": 1})
    session.commit.assert_awaited_once()
    inserted = session.add.call_args.args[0]
    assert inserted.user_id is None


@pytest.mark.asyncio
async def test_record_event_uuid_str_coerces():
    from src.agentrag.observability import activity

    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()

    @asynccontextmanager
    async def fake_sess():
        yield session

    uid = uuid.uuid4()
    with patch.object(activity, "AsyncSessionLocal", fake_sess):
        await activity.record_event(str(uid), "search", payload={"query": "abc"})
    inserted = session.add.call_args.args[0]
    assert inserted.user_id == uid
    assert inserted.event_type == "search"


@pytest.mark.asyncio
async def test_record_event_invalid_uuid_returns_null():
    from src.agentrag.observability import activity

    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()

    @asynccontextmanager
    async def fake_sess():
        yield session

    with patch.object(activity, "AsyncSessionLocal", fake_sess):
        await activity.record_event("not-a-uuid", "chat_turn")
    inserted = session.add.call_args.args[0]
    assert inserted.user_id is None
