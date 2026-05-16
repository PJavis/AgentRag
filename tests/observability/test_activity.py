"""S6 — record_event helper."""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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
        yield  # pragma: no cover

    with patch.object(activity, "AsyncSessionLocal", boom), caplog.at_level(logging.WARNING):
        await activity.record_event(user_id=None, event_type="x")
    assert any("record_event failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_record_event_anonymous_coerced_to_null():
    from src.agentrag.observability import activity

    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()

    @asynccontextmanager
    async def fake_sess():
        yield session

    with patch.object(activity, "AsyncSessionLocal", fake_sess):
        await activity.record_event("anonymous", "search", payload={"q": "x"})
    inserted_row = session.add.call_args.args[0]
    assert inserted_row.user_id is None
