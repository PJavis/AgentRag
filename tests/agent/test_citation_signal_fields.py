from src.agentrag.agent.service import AgentService


def test_build_packed_citations_carries_node_level_and_context_text():
    svc = AgentService.__new__(AgentService)
    packed = [
        {"document_title": "Tim mạch", "content": "Nhồi máu cơ tim ...",
         "segment_type": "raptor_summary", "node_level": 1,
         "context_text": "From the cardiology chapter on MI."},
        {"document_title": "Tim mạch", "content": "Aspirin ...",
         "segment_type": "text", "node_level": 0},
    ]
    out = svc._build_packed_citations(packed)
    assert out[0]["node_level"] == 1
    assert out[0]["segment_type"] == "raptor_summary"
    assert out[0]["context_text"] == "From the cardiology chapter on MI."
    assert out[1]["node_level"] == 0
    assert out[1].get("context_text") in (None, "")
