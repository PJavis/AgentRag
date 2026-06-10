from src.agentrag.agent.graph_service import _message_signals


def test_message_signals_reads_cache_hit_and_mode_from_first_trace():
    tool_trace = [
        {"tool_name": "search_hybrid_kg",
         "tool_output": {"semantic_cache_hit": True, "mode": "hybrid_kg",
                         "domain_route": "tim_mach"}},
        {"tool_name": "search_hybrid_kg", "tool_output": {"mode": "hybrid"}},
    ]
    sig = _message_signals(tool_trace)
    assert sig["semantic_cache_hit"] is True
    assert sig["retrieval_mode"] == "hybrid_kg"
    assert sig["domain_route"] == "tim_mach"


def test_message_signals_defaults_when_absent():
    sig = _message_signals([])
    assert sig["semantic_cache_hit"] is False
    assert sig["retrieval_mode"] is None
    assert sig["domain_route"] is None
