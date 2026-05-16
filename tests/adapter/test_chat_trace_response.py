"""S2 — verify ChatMessage adapter model exposes trace fields."""
from __future__ import annotations

from src.agentrag.adapter.models import ChatMessage


def test_chat_message_optional_trace_fields():
    msg = ChatMessage(
        type="ai",
        role="assistant",
        content="hello",
    )
    assert msg.reasoning_path is None
    assert msg.timings_ms is None
    assert msg.plan_subqueries is None
    assert msg.sql_query is None


def test_chat_message_populated_trace():
    msg = ChatMessage(
        type="ai",
        role="assistant",
        content="hello",
        reasoning_path="semantic",
        timings_ms={"total": 1234.5, "plan": 12, "decide": 50, "tool": 800, "answer": 350},
        tool_trace=[{"tool_name": "search_hybrid_kg", "tool_input": {"query": "x"}, "tool_output": {}}],
        plan_subqueries=["sub-1", "sub-2"],
        sql_query=None,
    )
    d = msg.model_dump()
    assert d["reasoning_path"] == "semantic"
    assert d["timings_ms"]["total"] == 1234.5
    assert d["tool_trace"][0]["tool_name"] == "search_hybrid_kg"
    assert d["plan_subqueries"] == ["sub-1", "sub-2"]
