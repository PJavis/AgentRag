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


def test_content_or_none_gates_on_capture_flag(monkeypatch):
    from src.agentrag.common.langfuse_client import _content_or_none
    monkeypatch.setattr(settings, "OBSERVABILITY_CAPTURE_CONTENT", False)
    assert _content_or_none("PHI question text") is None  # privacy-safe default
    monkeypatch.setattr(settings, "OBSERVABILITY_CAPTURE_CONTENT", True)
    assert _content_or_none("PHI question text") == "PHI question text"


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


def test_score_trace_swallows_when_langfuse_raises(monkeypatch):
    """Enabled but Langfuse unreachable → score_trace must NOT raise into the caller
    (the /feedback endpoint mirrors ratings via this; Langfuse downtime must not 500)."""
    import sys
    from src.agentrag.common.langfuse_client import score_trace
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", True)

    class _Boom:
        def __init__(self, *a, **k):
            pass
        def score(self, *a, **k):
            raise RuntimeError("langfuse down")

    # Mock langfuse module
    mock_langfuse = type(sys)("langfuse")
    mock_langfuse.Langfuse = _Boom
    monkeypatch.setitem(sys.modules, "langfuse", mock_langfuse)
    assert score_trace("trace-xyz", name="user_feedback", value=1.0) is None  # swallowed


def test_update_turn_trace_swallows_when_langfuse_raises(monkeypatch):
    """Enabled but the decorator context raises → update_turn_trace must no-op, not raise."""
    import sys
    from src.agentrag.common import langfuse_client
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", True)

    class _BoomCtx:
        @staticmethod
        def update_current_trace(*a, **k):
            raise RuntimeError("no active trace")

    # Mock langfuse.decorators module
    mock_decorators = type(sys)("langfuse.decorators")
    mock_decorators.langfuse_context = _BoomCtx
    monkeypatch.setitem(sys.modules, "langfuse.decorators", mock_decorators)
    assert langfuse_client.update_turn_trace(name="q", session_id="c1", metadata={"k": 1}) is None


def test_langfuse_flush_swallows_when_langfuse_raises(monkeypatch):
    """Enabled but flush paths raise → langfuse_flush must swallow so shutdown never breaks."""
    import sys
    from src.agentrag.common.langfuse_client import langfuse_flush
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", True)

    class _Boom:
        def __init__(self, *a, **k):
            pass
        def flush(self, *a, **k):
            raise RuntimeError("flush failed")

    # break both the openai-integration flush and the singleton flush
    mock_langfuse = type(sys)("langfuse")
    mock_langfuse.Langfuse = _Boom
    monkeypatch.setitem(sys.modules, "langfuse", mock_langfuse)
    monkeypatch.setitem(sys.modules, "langfuse.openai", type(sys)("langfuse.openai"))
    assert langfuse_flush() is None  # swallowed, no raise
