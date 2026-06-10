from src.agentrag.agent.service import _has_uncertainty, AgentService


def test_uncertainty_phrases_detected():
    assert _has_uncertainty("Xin lỗi, tôi không tìm thấy thông tin về điều này.")
    assert _has_uncertainty("I don't have enough information to answer.")
    assert not _has_uncertainty("Nhồi máu cơ tim là tình trạng tắc nghẽn mạch vành.")


def test_critique_flags_when_no_citations():
    svc = AgentService.__new__(AgentService)
    decision = svc._critique(
        answer="Nhồi máu cơ tim là ...",
        citations=[],                      # nothing grounded
        packed_context=[{"content": "..."}],
    )
    assert decision["grounded"] is False
    assert decision["reason"] == "no_citations"


def test_critique_flags_on_too_few_hits():
    svc = AgentService.__new__(AgentService)
    decision = svc._critique(
        answer="Câu trả lời.",
        citations=[{"source": 1}],
        packed_context=[],                 # retrieval returned nothing
    )
    assert decision["grounded"] is False
    assert decision["reason"] == "insufficient_context"


def test_critique_passes_for_grounded_answer():
    svc = AgentService.__new__(AgentService)
    decision = svc._critique(
        answer="Nhồi máu cơ tim là tắc động mạch vành [1].",
        citations=[{"source": 1}],
        packed_context=[{"content": "..."}, {"content": "..."}],
    )
    assert decision["grounded"] is True
