import pytest
from src.agentrag.services.llm_gateway import LLMGateway, VisionDisabledError
from src.agentrag.config import settings


def test_multimodal_routes_to_vision_answer_model(monkeypatch):
    """Answer-time multimodal must resolve the VISION_ANSWER_MODEL client, not the
    text `answer` model."""
    monkeypatch.setattr(settings, "VISION_ANSWER_MODEL", "gemini-2.5-flash")
    gw = LLMGateway()
    client = gw._client_for_model(settings.VISION_ANSWER_MODEL)
    assert client.model == "gemini-2.5-flash"  # gemini-prefixed → Gemini provider
    # and it is cached / reused
    assert gw._client_for_model("gemini-2.5-flash") is client


def test_multimodal_disabled_when_model_empty(monkeypatch):
    """Empty VISION_ANSWER_MODEL → json_response_multimodal raises VisionDisabledError
    (caller falls back to text-only)."""
    monkeypatch.setattr(settings, "VISION_ANSWER_MODEL", "")
    gw = LLMGateway()
    with pytest.raises(VisionDisabledError):
        import asyncio
        asyncio.run(gw.json_response_multimodal("sys", "user", ["http://x/img.png"], task="answer_vision"))


def test_text_answer_model_unchanged(monkeypatch):
    """Text path still resolves the `answer` task model (not the vision model)."""
    monkeypatch.setattr(settings, "LLM_ROUTING_ENABLED", True)
    monkeypatch.setattr(settings, "LLM_TASK_MODEL_MAP", '{"answer":"deepseek-v4-flash"}')
    gw = LLMGateway()
    client = gw._resolve_client("answer", content="hi")
    assert client.model == "deepseek-v4-flash"
