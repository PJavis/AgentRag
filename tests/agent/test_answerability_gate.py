from src.agentrag.agent.service import _in_gray_band, _should_drop_abstention_citations
from src.agentrag.config import settings


def _ctx(*scores):
    return [{"rerank_score": s} for s in scores]


def test_gray_band_true_when_best_in_band():
    # floor 0.6, margin 0.13 → band [0.60, 0.73)
    assert _in_gray_band(_ctx(0.50, 0.64), floor=0.6, margin=0.13) is True


def test_gray_band_false_when_best_above_band():
    assert _in_gray_band(_ctx(0.74, 0.40), floor=0.6, margin=0.13) is False


def test_gray_band_false_when_best_below_floor():
    # below floor is already handled by _is_thin_context, not the gray band
    assert _in_gray_band(_ctx(0.40, 0.55), floor=0.6, margin=0.13) is False


def test_gray_band_false_when_no_scores():
    assert _in_gray_band([{"text": "x"}], floor=0.6, margin=0.13) is False


_REFUSAL = "Tôi không có thông tin về vấn đề này."  # contains an uncertainty marker


def test_drop_citations_on_gray_band_abstain_when_gate_on(monkeypatch):
    # gray-band (best 0.64 ∈ [0.6,0.73)) → thin is False, but gate ON must still
    # scrub distractor citations on the forced refusal.
    monkeypatch.setattr(settings, "ANSWER_ABSTAIN_ON_THIN_CONTEXT", True)
    monkeypatch.setattr(settings, "ANSWERABILITY_GATE_ENABLED", True)
    monkeypatch.setattr(settings, "ANSWERABILITY_GRAY_MARGIN", 0.13)
    assert _should_drop_abstention_citations(_REFUSAL, _ctx(0.50, 0.64), floor=0.6) is True


def test_no_drop_on_gray_band_when_gate_off(monkeypatch):
    # gate OFF: gray band must NOT trigger the scrub (thin is False at 0.64).
    monkeypatch.setattr(settings, "ANSWER_ABSTAIN_ON_THIN_CONTEXT", True)
    monkeypatch.setattr(settings, "ANSWERABILITY_GATE_ENABLED", False)
    assert _should_drop_abstention_citations(_REFUSAL, _ctx(0.50, 0.64), floor=0.6) is False


def test_drop_citations_on_thin_regardless_of_gate(monkeypatch):
    # below floor (0.55 < 0.6) is thin → scrub fires even with the gate OFF.
    monkeypatch.setattr(settings, "ANSWER_ABSTAIN_ON_THIN_CONTEXT", True)
    monkeypatch.setattr(settings, "ANSWERABILITY_GATE_ENABLED", False)
    assert _should_drop_abstention_citations(_REFUSAL, _ctx(0.40, 0.55), floor=0.6) is True
