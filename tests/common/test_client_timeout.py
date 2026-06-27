"""make_async_openai must inject a default request timeout so a stalled provider
connection (e.g. gemini 503/high-demand) fails fast + retries instead of hanging
forever — a 42-min agent.chat hang was observed without it (2026-06-26)."""
from src.agentrag.common.langfuse_client import _with_timeout_default
from src.agentrag.config import settings


def test_injects_default_timeout_when_absent():
    out = _with_timeout_default({"api_key": "x"})
    assert out["timeout"] == settings.LLM_REQUEST_TIMEOUT_S
    assert out["api_key"] == "x"  # other kwargs preserved


def test_preserves_explicit_timeout():
    out = _with_timeout_default({"api_key": "x", "timeout": 5.0})
    assert out["timeout"] == 5.0
