"""Relevance-floor citation gate — drop low-relevance distractors before the
answer node sees them. Pure-function unit tests + sigmoid sanity."""
from __future__ import annotations

from src.agentrag.agent.context import apply_relevance_floor
from src.agentrag.retrieval.reranker import _sigmoid


def test_drops_below_floor_keeps_above():
    items = [{"id": "a", "rerank_score": 0.8}, {"id": "b", "rerank_score": 0.1},
             {"id": "c", "rerank_score": 0.35}]
    out = apply_relevance_floor(items, floor=0.3)
    ids = {i["id"] for i in out}
    assert ids == {"a", "c"}          # b (0.1) dropped


def test_all_below_floor_yields_empty():
    items = [{"id": "a", "rerank_score": 0.1}, {"id": "b", "rerank_score": 0.05}]
    assert apply_relevance_floor(items, floor=0.3) == []   # → empty context → clean abstain


def test_no_score_present_is_passthrough():
    items = [{"id": "a"}, {"id": "b"}]                      # reranker off / no score
    assert apply_relevance_floor(items, floor=0.3) == items


def test_mixed_missing_score_is_kept():
    items = [{"id": "a", "rerank_score": 0.9}, {"id": "b"}]  # b has no score → kept
    out = apply_relevance_floor(items, floor=0.3)
    assert {i["id"] for i in out} == {"a", "b"}


def test_sigmoid_midpoint():
    assert _sigmoid(0.0) == 0.5


def test_sigmoid_saturates():
    assert _sigmoid(1000.0) > 0.999
    assert _sigmoid(-1000.0) < 0.001


def test_sigmoid_no_overflow_on_extremes():
    # Numerically stable: large magnitudes must not raise OverflowError.
    assert 0.0 <= _sigmoid(1000.0) <= 1.0
    assert 0.0 <= _sigmoid(-1000.0) <= 1.0
