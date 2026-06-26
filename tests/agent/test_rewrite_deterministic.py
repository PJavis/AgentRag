"""Query-rewrite must be deterministic (temperature 0) so the same question doesn't
flip answer<->abstain run-to-run via rewrite-driven retrieval variance near the
relevance floor (2026-06-26 flaky-abstention root cause). Verifies the temperature
plumbs gateway -> client, and that QueryRewriter requests 0.0."""
import pytest

from src.agentrag.services.llm_gateway import LLMGateway
from src.agentrag.retrieval.query_rewriter import QueryRewriter


class _FakeClient:
    def __init__(self):
        self.temps = []

    async def json_response(self, system_prompt, user_prompt, task="general", temperature=None):
        self.temps.append(temperature)
        return {}


@pytest.mark.asyncio
async def test_gateway_forwards_temperature(monkeypatch):
    gw = LLMGateway()
    fake = _FakeClient()
    monkeypatch.setattr(gw, "_resolve_client", lambda task, content="": fake)
    await gw.json_response(system_prompt="s json", user_prompt="u", task="decide", temperature=0.0)
    assert fake.temps == [0.0]


@pytest.mark.asyncio
async def test_gateway_default_temperature_is_none(monkeypatch):
    gw = LLMGateway()
    fake = _FakeClient()
    monkeypatch.setattr(gw, "_resolve_client", lambda task, content="": fake)
    await gw.json_response(system_prompt="s json", user_prompt="u", task="decide")
    assert fake.temps == [None]


class _FakeGateway:
    def __init__(self):
        self.temps = []

    async def json_response(self, system_prompt, user_prompt, task="general", temperature=None):
        self.temps.append(temperature)
        return ({"queries": []}, 1.0)


@pytest.mark.asyncio
async def test_query_rewriter_requests_temperature_zero():
    gw = _FakeGateway()
    rw = QueryRewriter(gw)
    await rw.make_hyde_text("Bob Marley đã qua đời vì bệnh gì?")
    await rw.decompose("Bob Marley đã qua đời vì bệnh gì?")
    assert gw.temps, "QueryRewriter made no gateway json_response call"
    assert all(t == 0.0 for t in gw.temps), f"rewrite temps not all 0.0: {gw.temps}"
