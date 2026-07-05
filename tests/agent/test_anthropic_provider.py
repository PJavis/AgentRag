"""A `claude-*` model in LLM_TASK_MODEL_MAP must auto-route to the anthropic provider
(Anthropic's OpenAI-compatible endpoint) — primary use is an INDEPENDENT cross-provider
eval judge so the correctness score isn't self-graded by the answer model's provider."""
import pytest

from src.agentrag.agent import llm as llm_mod


def test_claude_model_routes_to_anthropic_endpoint(monkeypatch):
    monkeypatch.setattr(llm_mod.settings, "ANTHROPIC_API_KEY", "sk-ant-test")
    a = llm_mod.AgentLLM(model_override="claude-haiku-4-5")
    assert a.model == "claude-haiku-4-5"
    assert a.base_url == "https://api.anthropic.com/v1/"
    assert a.api_key == "sk-ant-test"


def test_anthropic_requires_key(monkeypatch):
    monkeypatch.setattr(llm_mod.settings, "ANTHROPIC_API_KEY", None)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        llm_mod.AgentLLM(model_override="claude-sonnet-4-6")


def test_non_claude_model_unaffected(monkeypatch):
    # a gemini model must still route to gemini, not anthropic
    monkeypatch.setattr(llm_mod.settings, "GEMINI_API_KEY", "g-test")
    a = llm_mod.AgentLLM(model_override="gemini-2.5-flash")
    assert "anthropic" not in (a.base_url or "")
    assert a.model == "gemini-2.5-flash"
