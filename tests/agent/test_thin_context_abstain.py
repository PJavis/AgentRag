from src.agentrag.agent.service import (
    _is_thin_context, _should_drop_abstention_citations, _answer_system_prompt,
)
from src.agentrag.agent import service as svc


def test_is_thin_context_true_when_best_below_floor():
    packed = [{"rerank_score": 0.1}, {"rerank_score": 0.25}]
    assert _is_thin_context(packed, 0.3) is True


def test_is_thin_context_false_when_some_above_floor():
    packed = [{"rerank_score": 0.1}, {"rerank_score": 0.8}]
    assert _is_thin_context(packed, 0.3) is False


def test_is_thin_context_false_when_no_scores():
    assert _is_thin_context([{"document_title": "x"}], 0.3) is False
    assert _is_thin_context([], 0.3) is False
    assert _is_thin_context(None, 0.3) is False


def test_prompt_thin_override_instructs_clean_abstain(monkeypatch):
    monkeypatch.setattr(svc.settings, "ANSWER_ABSTAIN_ON_THIN_CONTEXT", True)
    monkeypatch.setattr(svc.settings, "RETRIEVAL_RELEVANCE_FLOOR", 0.3)
    p = _answer_system_prompt("Thuốc Zxylopraxin-9?", False, [{"rerank_score": 0.1}])
    assert "do not cite" in p.lower() or "không.*trích" in p.lower() or "cite any source" in p.lower()
    assert "background knowledge" in p.lower()


def test_prompt_normal_when_flag_off(monkeypatch):
    monkeypatch.setattr(svc.settings, "ANSWER_ABSTAIN_ON_THIN_CONTEXT", False)
    p = _answer_system_prompt("Triệu chứng NMCT?", False, [{"rerank_score": 0.1}])
    assert "INLINE CITATIONS" in p          # the normal full prompt, not the override


def test_should_drop_citations_only_on_thin_abstention(monkeypatch):
    monkeypatch.setattr(svc.settings, "ANSWER_ABSTAIN_ON_THIN_CONTEXT", True)
    monkeypatch.setattr(svc.settings, "RETRIEVAL_RELEVANCE_FLOOR", 0.3)
    thin = [{"rerank_score": 0.1}]
    assert _should_drop_abstention_citations("Tôi không tìm thấy thông tin.", thin, 0.3) is True
    # confident answer → keep citations
    assert _should_drop_abstention_citations("NMCT là tắc mạch vành.", thin, 0.3) is False
    # not thin → keep
    assert _should_drop_abstention_citations("Không tìm thấy.", [{"rerank_score": 0.9}], 0.3) is False
