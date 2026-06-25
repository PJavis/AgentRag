from src.agentrag.common.langfuse_client import observe_chat_turn, update_turn_trace
from src.agentrag.config import settings


def test_observe_is_passthrough_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", False)

    async def f(x):
        return x + 1

    wrapped = observe_chat_turn(f)
    assert wrapped is f  # no wrapping, zero overhead when off


def test_update_turn_trace_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", False)
    # Must not raise and must return None even with no Langfuse server.
    assert update_turn_trace(name="q", session_id="conv-1") is None


def test_current_trace_id_none_when_disabled(monkeypatch):
    from src.agentrag.common.langfuse_client import current_trace_id
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", False)
    assert current_trace_id() is None


def test_score_trace_noop_when_disabled(monkeypatch):
    from src.agentrag.common.langfuse_client import score_trace
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", False)
    assert score_trace("trace-1", name="user_feedback", value=1.0) is None


def test_score_trace_noop_when_no_trace_id(monkeypatch):
    from src.agentrag.common.langfuse_client import score_trace
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", True)
    # No trace_id → must no-op (and not construct a Langfuse client).
    assert score_trace(None, name="user_feedback", value=-1.0) is None
