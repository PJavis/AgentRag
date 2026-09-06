"""Cache-hit tokens are billed differently by the provider; the ledger must see them.

Measured on DeepSeek: a repeated 6k-token prefix returned
prompt_cache_hit_tokens=6016 / prompt_cache_miss_tokens=27. Billing all prompt
tokens at the miss rate overstates spend and hides whether caching works at all.
"""
from types import SimpleNamespace

import pytest

from src.agentrag.config import settings
from src.agentrag.observability import cost


@pytest.fixture(autouse=True)
def _tracking_on(monkeypatch):
    monkeypatch.setattr(settings, "LLM_COST_TRACKING_ENABLED", True)
    cost.reset_ledger()


def test_deepseek_has_its_own_price_and_is_not_billed_as_gemini():
    assert cost._price_for("deepseek-chat") != cost._price_for("gemini-2.5-flash")


def test_cache_hit_tokens_are_recorded_from_provider_usage():
    usage = SimpleNamespace(
        prompt_tokens=6043, completion_tokens=1,
        prompt_cache_hit_tokens=6016, prompt_cache_miss_tokens=27,
    )
    cost.record_llm_call(task="answer", model="deepseek-chat", latency_ms=10.0, usage=usage)
    entry = cost.recent(1)[0]
    assert entry["cache_hit_tokens"] == 6016
    assert entry["cache_miss_tokens"] == 27


def test_a_cached_prompt_costs_less_than_an_uncached_one():
    hit = SimpleNamespace(prompt_tokens=6043, completion_tokens=1,
                          prompt_cache_hit_tokens=6016, prompt_cache_miss_tokens=27)
    miss = SimpleNamespace(prompt_tokens=6043, completion_tokens=1,
                           prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=6043)
    cost.record_llm_call(task="answer", model="deepseek-chat", latency_ms=1.0, usage=miss)
    cost.record_llm_call(task="answer", model="deepseek-chat", latency_ms=1.0, usage=hit)
    entries = cost.recent(2)
    assert entries[1]["usd"] < entries[0]["usd"]


def test_a_provider_without_cache_fields_still_records_zero_not_none():
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10)
    cost.record_llm_call(task="answer", model="gemini-2.5-flash", latency_ms=1.0, usage=usage)
    entry = cost.recent(1)[0]
    assert entry["cache_hit_tokens"] == 0
    assert entry["cache_miss_tokens"] == 100  # all prompt tokens were misses


def test_summary_reports_a_cache_hit_rate():
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=1,
                            prompt_cache_hit_tokens=900, prompt_cache_miss_tokens=100)
    cost.record_llm_call(task="answer", model="deepseek-chat", latency_ms=1.0, usage=usage)
    summary = cost.cost_summary()
    assert summary["cache_hit_tokens"] == 900
    assert summary["cache_miss_tokens"] == 100
    assert abs(summary["cache_hit_rate"] - 0.9) < 1e-9


def test_hit_rate_is_none_when_no_prompt_tokens_were_seen():
    assert cost.cost_summary()["cache_hit_rate"] is None
