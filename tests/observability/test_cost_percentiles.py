"""S3 — cost summary surfaces p50/p95 latency per task/model."""
from __future__ import annotations

import pytest

from src.agentrag.config import settings
from src.agentrag.observability import cost as ledger
from src.agentrag.observability.cost import _percentile


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setattr(settings, "LLM_COST_TRACKING_ENABLED", True)
    ledger.reset_ledger()
    yield
    ledger.reset_ledger()


def test_percentile_empty_returns_zero():
    assert _percentile([], 50) == 0.0


def test_percentile_single_value_constant():
    assert _percentile([42.0], 95) == 42.0


def test_percentile_linear_interp():
    # p50 of [10, 20, 30, 40] → 25 (between idx 1 and idx 2)
    assert _percentile([10, 20, 30, 40], 50) == 25.0


def test_summary_includes_p50_p95_per_task():
    for ms in [10, 20, 30, 100, 200]:
        ledger.record_llm_call(task="answer", model="m", latency_ms=ms, in_text="x", out_text="y")
    s = ledger.cost_summary()
    a = s["per_task"]["answer"]
    assert a["p50_latency_ms"] > 0
    assert a["p95_latency_ms"] >= a["p50_latency_ms"]
    assert a["p95_latency_ms"] >= a["avg_latency_ms"]  # p95 ≥ avg for skewed dist
