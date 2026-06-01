"""Source + notebook chat starters endpoints — graceful, best-effort."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_build_summary_text_from_insights():
    from src.agentrag.adapter.routers import chat as chat_mod
    with patch.object(chat_mod, "_load_source_starter_inputs",
                      new=AsyncMock(return_value=("Giải phẫu tim", "summary text"))):
        title, summary = await chat_mod._load_source_starter_inputs("source:abc")
    assert title == "Giải phẫu tim"
    assert summary == "summary text"


@pytest.mark.asyncio
async def test_source_starters_swallows_generator_failure():
    from src.agentrag.adapter.routers import chat as chat_mod
    with patch.object(chat_mod, "_load_source_starter_inputs",
                      new=AsyncMock(return_value=("T", "s"))), \
         patch("src.agentrag.agent.starters.generate_starters",
               new=AsyncMock(side_effect=RuntimeError("boom"))):
        resp = await chat_mod.get_source_starters("source:abc")
    assert resp == {"starters": []}
