from src.agentrag.agent.service import _in_gray_band


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
