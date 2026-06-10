import asyncio
from src.agentrag.agent import graph_service as gs


def test_critique_node_records_positive_latency_when_enabled(monkeypatch):
    monkeypatch.setattr(gs.settings, "CRAG_ENABLED", True)
    # _INNER._critique is pure; stub it to a known decision
    monkeypatch.setattr(gs._INNER, "_critique",
                        lambda answer, citations, packed_context: {"grounded": True, "reason": "ok"})
    out = asyncio.run(gs.critique({"answer": "a", "citations": [{"source": 1}], "packed_context": [{"content": "x"}]}))
    assert out["critique_decision"]["grounded"] is True
    assert out["critique_latency_ms"] > 0


def test_critique_node_no_latency_when_disabled(monkeypatch):
    monkeypatch.setattr(gs.settings, "CRAG_ENABLED", False)
    out = asyncio.run(gs.critique({"answer": "a"}))
    assert "critique_latency_ms" not in out
    assert out["critique_decision"]["reason"] == "disabled"
