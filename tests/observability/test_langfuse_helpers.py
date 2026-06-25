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
