"""Source + notebook chat starters endpoints — graceful, best-effort."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _FakeResult:
    def __init__(self, values):
        self._values = values

    def scalars(self):
        m = MagicMock()
        m.all.return_value = self._values
        return m


@pytest.mark.asyncio
async def test_load_source_starter_inputs_uses_insights():
    from src.agentrag.adapter.routers import chat as chat_mod

    sid = uuid.uuid4()
    fake_doc = MagicMock()
    fake_doc.title = "Giải phẫu tim"

    session = MagicMock()
    session.get = AsyncMock(return_value=fake_doc)
    # insights query returns two items → summary path taken, segments skipped
    session.execute = AsyncMock(return_value=_FakeResult(["Insight A", "Insight B"]))

    @asynccontextmanager
    async def fake_session_local():
        yield session

    with patch.object(chat_mod, "AsyncSessionLocal", fake_session_local), \
         patch.object(chat_mod, "_parse_source_uuid", return_value=sid):
        title, summary = await chat_mod._load_source_starter_inputs("source:" + str(sid))

    assert title == "Giải phẫu tim"
    assert "Insight A" in summary and "Insight B" in summary


@pytest.mark.asyncio
async def test_source_starters_swallows_generator_failure():
    from src.agentrag.adapter.routers import chat as chat_mod
    with patch.object(chat_mod, "_load_source_starter_inputs",
                      new=AsyncMock(return_value=("T", "s"))), \
         patch("src.agentrag.agent.starters.generate_starters",
               new=AsyncMock(side_effect=RuntimeError("boom"))):
        resp = await chat_mod.get_source_starters("source:abc")
    assert resp == {"starters": []}
