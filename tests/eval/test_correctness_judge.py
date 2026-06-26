from src.agentrag.eval.correctness_judge import (
    clamp01, parse_nuggets, aggregate_nugget_labels, ensemble,
    NuggetScore, EnsembleScore,
)


def test_clamp01_bounds():
    assert clamp01(-0.3) == 0.0
    assert clamp01(1.7) == 1.0
    assert clamp01(0.42) == 0.42


def test_parse_nuggets_strips_and_drops_empty():
    raw = {"nuggets": ["  fact A ", "", "fact B", 5, None]}
    assert parse_nuggets(raw) == ["fact A", "fact B"]


def test_parse_nuggets_missing_key():
    assert parse_nuggets({}) == []


def test_aggregate_nugget_labels_recall_minus_contradiction():
    # 4 nuggets: 2 covered, 1 contradicted, 1 absent
    s = aggregate_nugget_labels(["covered", "covered", "contradicted", "absent"])
    assert s.n_total == 4
    assert s.n_covered == 2
    assert s.n_contradicted == 1
    assert s.recall == 0.5
    assert s.contradiction_penalty == 0.25
    assert s.score == 0.25  # max(0, 0.5 - 0.25)


def test_aggregate_nugget_labels_empty_is_zero():
    s = aggregate_nugget_labels([])
    assert s == NuggetScore(0, 0, 0, 0.0, 0.0, 0.0)


def test_aggregate_floor_at_zero():
    # contradictions exceed coverage → score floored, not negative
    s = aggregate_nugget_labels(["contradicted", "contradicted", "covered"])
    assert s.score == 0.0


def test_ensemble_mean_and_delta_flag():
    e = ensemble(0.6, 0.7)
    assert e.nugget == 0.6
    assert e.rubric == 0.7
    assert abs(e.mean - 0.65) < 1e-9
    assert abs(e.abs_delta - 0.1) < 1e-9
    assert e.low_confidence is False


def test_ensemble_flags_high_delta():
    e = ensemble(0.2, 0.9)
    assert e.low_confidence is True  # abs_delta 0.7 > 0.2
