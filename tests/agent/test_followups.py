"""B2 — generate_followups."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_generate_followups_parses_json():
    from src.agentrag.agent import followups
    followups._CACHE.clear()
    gateway = MagicMock()
    gateway.json_response = AsyncMock(return_value=(
        {"follow_ups": ["Câu 1?", "Câu 2?", "Câu 3?"]},
        12.0,
    ))
    out = await followups.generate_followups(
        question="Van hai lá là gì?",
        answer="Van hai lá nằm…",
        citations=[],
        llm_gateway=gateway,
    )
    assert out == ["Câu 1?", "Câu 2?", "Câu 3?"]
    gateway.json_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_followups_cache_hit():
    from src.agentrag.agent import followups
    followups._CACHE.clear()
    gateway = MagicMock()
    gateway.json_response = AsyncMock(return_value=(
        {"follow_ups": ["A?", "B?"]}, 10.0,
    ))
    q, a = "x", "y"
    out1 = await followups.generate_followups(q, a, [], gateway)
    out2 = await followups.generate_followups(q, a, [], gateway)
    assert out1 == out2 == ["A?", "B?"]
    gateway.json_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_followups_swallows_failure():
    from src.agentrag.agent import followups
    followups._CACHE.clear()
    gateway = MagicMock()
    gateway.json_response = AsyncMock(side_effect=RuntimeError("LLM down"))
    out = await followups.generate_followups("q", "a", [], gateway)
    assert out == []
