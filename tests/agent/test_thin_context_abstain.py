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


def test_borderline_relevant_chunk_no_longer_abstains_at_new_floor():
    """Prod finding (2026-06-26): paraphrased-relevant VN chunks jittered under the
    floor → flaky false-abstention, so it was lowered (old double-sigmoid scale
    0.6 → 0.55; as a probability 0.405 → 0.2007). A relevant chunk that dips to
    p≈0.32 (old-scale 0.58) still answers; off-corpus p≈0 (old 0.50) abstains."""
    floor = svc.settings.RETRIEVAL_RELEVANCE_MIN_PROB
    assert floor <= 0.2007, f"floor regressed to {floor}; borderline-relevant chunks will flaky-abstain"
    assert _is_thin_context([{"rerank_score": 0.32}], floor) is False   # relevant → answer
    assert _is_thin_context([{"rerank_score": 0.0}], floor) is True     # off-corpus → abstain


def test_probability_floor_matches_old_double_sigmoid_decisions():
    """2026-10-04 scale fix must not change a single abstain decision: for any
    probability p, old rule sigmoid(p) < 0.55 ⇔ new rule p < 0.2007."""
    import math

    from src.agentrag.agent.service import _in_gray_band

    def sig(x):
        return 1 / (1 + math.exp(-x))

    for i in range(0, 1001):
        p = i / 1000
        if abs(p - 0.2007) < 1e-3 or abs(p - 0.7538) < 1e-3:
            continue  # rounding of the converted constants lives here
        old = [{"rerank_score": sig(p)}]
        new = [{"rerank_score": p}]
        assert _is_thin_context(old, 0.55) == _is_thin_context(new, 0.2007), p
        assert _in_gray_band(old, 0.55, 0.13) == _in_gray_band(new, 0.2007, 0.5531), p


def test_prompt_thin_override_instructs_clean_abstain(monkeypatch):
    monkeypatch.setattr(svc.settings, "ANSWER_ABSTAIN_ON_THIN_CONTEXT", True)
    monkeypatch.setattr(svc.settings, "RETRIEVAL_RELEVANCE_MIN_PROB", 0.3)
    p = _answer_system_prompt("Thuốc Zxylopraxin-9?", False, [{"rerank_score": 0.1}])
    assert "do not cite" in p.lower() or "không.*trích" in p.lower() or "cite any source" in p.lower()
    assert "background knowledge" in p.lower()


def test_prompt_normal_when_flag_off(monkeypatch):
    monkeypatch.setattr(svc.settings, "ANSWER_ABSTAIN_ON_THIN_CONTEXT", False)
    p = _answer_system_prompt("Triệu chứng NMCT?", False, [{"rerank_score": 0.1}])
    assert "INLINE CITATIONS" in p          # the normal full prompt, not the override


def test_should_drop_citations_only_on_thin_abstention(monkeypatch):
    monkeypatch.setattr(svc.settings, "ANSWER_ABSTAIN_ON_THIN_CONTEXT", True)
    monkeypatch.setattr(svc.settings, "RETRIEVAL_RELEVANCE_MIN_PROB", 0.3)
    thin = [{"rerank_score": 0.1}]
    assert _should_drop_abstention_citations("Tôi không tìm thấy thông tin.", thin, 0.3) is True
    # confident answer → keep citations
    assert _should_drop_abstention_citations("NMCT là tắc mạch vành.", thin, 0.3) is False
    # not thin → keep
    assert _should_drop_abstention_citations("Không tìm thấy.", [{"rerank_score": 0.9}], 0.3) is False
