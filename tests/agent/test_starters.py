# tests/agent/test_starters.py
"""Starter-question generator for the empty chat state."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_generate_starters_parses_json():
    from src.agentrag.agent import starters
    starters._CACHE.clear()
    gateway = MagicMock()
    gateway.json_response = AsyncMock(return_value=(
        {"starters": ["Tóm tắt chương 1?", "Ai là tác giả?", "Kết luận chính?"]},
        9.0,
    ))
    out = await starters.generate_starters(
        kind="source",
        titles=["Giải phẫu tim"],
        summary_text="Tài liệu về giải phẫu tim mạch…",
        llm_gateway=gateway,
    )
    assert out == ["Tóm tắt chương 1?", "Ai là tác giả?", "Kết luận chính?"]
    gateway.json_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_starters_cache_hit():
    from src.agentrag.agent import starters
    starters._CACHE.clear()
    gateway = MagicMock()
    gateway.json_response = AsyncMock(return_value=({"starters": ["A?", "B?"]}, 5.0))
    out1 = await starters.generate_starters("source", ["T"], "s", gateway)
    out2 = await starters.generate_starters("source", ["T"], "s", gateway)
    assert out1 == out2 == ["A?", "B?"]
    gateway.json_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_starters_swallows_failure():
    from src.agentrag.agent import starters
    starters._CACHE.clear()
    gateway = MagicMock()
    gateway.json_response = AsyncMock(side_effect=RuntimeError("LLM down"))
    out = await starters.generate_starters("source", ["T"], "s", gateway)
    assert out == []


@pytest.mark.asyncio
async def test_generate_starters_empty_titles_returns_empty():
    from src.agentrag.agent import starters
    starters._CACHE.clear()
    gateway = MagicMock()
    gateway.json_response = AsyncMock()
    out = await starters.generate_starters("source", [], "", gateway)
    assert out == []
    gateway.json_response.assert_not_awaited()
