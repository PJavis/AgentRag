"""agent.chat must bound its TOTAL wall-clock (decide→tools→rerank→answer), not just
each LLM call — under a gemini 503 storm the loop ran >120s (42-min hang observed).
On exceeding AGENT_TOTAL_TIMEOUT_S the chat returns a graceful busy response, not a hang."""
import asyncio

import pytest

import src.agentrag.agent.graph_service as gs


@pytest.mark.asyncio
async def test_chat_returns_graceful_on_total_budget_exceeded(monkeypatch):
    monkeypatch.setattr(gs.settings, "AGENT_TOTAL_TIMEOUT_S", 0.05, raising=False)

    class _SlowGraph:
        async def ainvoke(self, initial, config=None):
            await asyncio.sleep(5)  # would hang far past the 0.05s budget
            return {"answer": "should never be returned"}

    monkeypatch.setattr(gs, "_GRAPH", _SlowGraph())
    svc = gs.GraphAgentService()

    out = await asyncio.wait_for(svc.chat(question="test?", conversation_id="t1"), timeout=2.0)

    assert out.get("timed_out") is True
    assert out["answer"] and "should never be returned" not in out["answer"]
    assert out["context"] == []
    assert out["reasoning_path"] == "timeout"


@pytest.mark.asyncio
async def test_chat_normal_path_unaffected(monkeypatch):
    monkeypatch.setattr(gs.settings, "AGENT_TOTAL_TIMEOUT_S", 5.0, raising=False)

    class _FastGraph:
        async def ainvoke(self, initial, config=None):
            return {"answer": "real answer", "packed_context": [{"content_hash": "X"}]}

    monkeypatch.setattr(gs, "_GRAPH", _FastGraph())
    svc = gs.GraphAgentService()

    out = await svc.chat(question="q?", conversation_id="t2")
    assert out["answer"] == "real answer"
    assert out.get("timed_out") is None
    assert out["context"] == [{"content_hash": "X"}]
