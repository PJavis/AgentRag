"""Cost ledger smoke tests (S1)."""
from __future__ import annotations

import time

import pytest

from src.agentrag.config import settings
from src.agentrag.observability import cost as ledger


@pytest.fixture(autouse=True)
def _reset_ledger(monkeypatch):
    monkeypatch.setattr(settings, "LLM_COST_TRACKING_ENABLED", True)
    ledger.reset_ledger()
    yield
    ledger.reset_ledger()


def test_record_then_summary_aggregates():
    ledger.record_llm_call(task="answer", model="gemini-2.5-flash", latency_ms=120, in_text="x" * 40, out_text="y" * 80)
    ledger.record_llm_call(task="answer", model="gemini-2.5-flash", latency_ms=80, in_text="x" * 20, out_text="y" * 40)
    ledger.record_llm_call(task="decide", model="gemini-2.5-flash-lite", latency_ms=50, in_text="x" * 10, out_text="y" * 5)

    s = ledger.cost_summary()
    assert s["total_calls"] == 3
    assert s["per_task"]["answer"]["calls"] == 2
    assert s["per_task"]["decide"]["calls"] == 1
    assert "gemini-2.5-flash" in s["per_model"]


def test_recent_calls_newest_first_with_timestamp_and_id():
    ledger.record_llm_call(task="a", model="m", latency_ms=10, in_text="x", out_text="y")
    ledger.record_llm_call(task="b", model="m", latency_ms=10, in_text="x", out_text="y")
    r = ledger.recent_calls(limit=10)
    assert len(r) == 2
    assert r[0]["task"] == "b"  # newest first
    assert r[1]["task"] == "a"
    assert r[0]["id"] > r[1]["id"]
    for e in r:
        assert "timestamp" in e
        assert e["timestamp"] > 0


def test_recent_calls_respects_limit():
    for i in range(20):
        ledger.record_llm_call(task=f"t{i}", model="m", latency_ms=1, in_text="x", out_text="y")
    r = ledger.recent_calls(limit=5)
    assert len(r) == 5


def test_recent_calls_since_filter():
    ledger.record_llm_call(task="old", model="m", latency_ms=1, in_text="x", out_text="y")
    cut = time.time() + 0.01
    time.sleep(0.02)
    ledger.record_llm_call(task="new", model="m", latency_ms=1, in_text="x", out_text="y")
    r = ledger.recent_calls(limit=10, since=cut)
    assert [e["task"] for e in r] == ["new"]


def test_record_no_op_when_tracking_disabled(monkeypatch):
    monkeypatch.setattr(settings, "LLM_COST_TRACKING_ENABLED", False)
    ledger.record_llm_call(task="x", model="m", latency_ms=1, in_text="x", out_text="y")
    assert ledger.cost_summary()["total_calls"] == 0
