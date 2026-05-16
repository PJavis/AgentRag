"""S5 acceptance tests — domain partition end-to-end (mocked infra).

Covers:
  1. FederatedRetriever — explicit override skips router.
  2. FederatedRetriever — no override consults router and passes filters.
  3. FederatedRetriever — DOMAIN_FILTER_ENABLED=false short-circuits.
  4. ContextVar — set_domain_filter is read by AgentTools._current_filters().
  5. ContextVar isolation across concurrent tasks (async safety).
  6. Adapter — /api/ontology/{systems,specialties} return non-empty taxonomy
     used by DomainRouter prompt + SectionTagger.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.agentrag.adapter.app import adapter
from src.agentrag.agent.tools import AgentTools
from src.agentrag.config import settings
from src.agentrag.orchestration.domain_router import DomainRoute
from src.agentrag.retrieval.context import get_domain_filter, set_domain_filter
from src.agentrag.retrieval.federated import FederatedRetriever


# -- Federated retriever -----------------------------------------------------

@pytest.mark.asyncio
async def test_explicit_override_bypasses_router_and_passes_filters():
    base = MagicMock()
    base.search = AsyncMock(return_value={"hits": []})
    router = MagicMock()
    router.classify = AsyncMock()  # must NOT be called

    fr = FederatedRetriever(base=base, router=router)
    out = await fr.search(
        query="đau ngực",
        system_override="tim_mach",
        specialty_override=["noi"],
        mode="hybrid",
    )

    router.classify.assert_not_called()
    assert "domain_route" not in out
    call = base.search.await_args
    assert call.kwargs["filters"] == {
        "systems": ["tim_mach"],
        "specialties": ["noi"],
    }


@pytest.mark.asyncio
async def test_no_override_consults_router_and_uses_picks():
    base = MagicMock()
    base.search = AsyncMock(return_value={"hits": []})
    router = MagicMock()
    router.classify = AsyncMock(
        return_value=DomainRoute(
            systems=["ho_hap"], specialties=["noi"], confidence=0.92, raw={}
        )
    )

    fr = FederatedRetriever(base=base, router=router)
    out = await fr.search(query="ho có đờm 3 tuần", mode="hybrid")

    router.classify.assert_awaited_once()
    assert base.search.await_args.kwargs["filters"] == {
        "systems": ["ho_hap"],
        "specialties": ["noi"],
    }
    assert out["domain_route"]["systems"] == ["ho_hap"]
    assert out["domain_route"]["confidence"] == 0.92


@pytest.mark.asyncio
async def test_domain_filter_disabled_short_circuits(monkeypatch):
    base = MagicMock()
    base.search = AsyncMock(return_value={"hits": []})
    router = MagicMock()
    router.classify = AsyncMock()

    monkeypatch.setattr(settings, "DOMAIN_FILTER_ENABLED", False)
    fr = FederatedRetriever(base=base, router=router)
    await fr.search(query="anything", mode="hybrid")

    router.classify.assert_not_called()
    # Disabled path doesn't forward filters kwarg
    assert "filters" not in base.search.await_args.kwargs


# -- ContextVar plumbing -----------------------------------------------------

@pytest.mark.asyncio
async def test_context_var_round_trip():
    token = set_domain_filter({"system": "than_kinh", "specialties": ["noi"]})
    try:
        assert get_domain_filter() == {"system": "than_kinh", "specialties": ["noi"]}
    finally:
        # cleanup so test order doesn't matter
        from src.agentrag.retrieval.context import _domain_filter
        _domain_filter.reset(token)


@pytest.mark.asyncio
async def test_agent_tools_reads_context_var():
    # Avoid live ES instantiation in AgentTools.__init__
    with patch("src.agentrag.agent.tools.ElasticsearchRetriever"), \
         patch("src.agentrag.agent.tools.ElasticsearchStore"):
        tools = AgentTools()
    token = set_domain_filter({"system": "tim_mach", "specialties": ["cap_cuu"]})
    try:
        assert tools._current_filters() == {
            "systems": ["tim_mach"],
            "specialties": ["cap_cuu"],
        }
    finally:
        from src.agentrag.retrieval.context import _domain_filter
        _domain_filter.reset(token)


@pytest.mark.asyncio
async def test_context_var_isolated_across_tasks():
    """Two concurrent turns with different domain_filters must not bleed."""
    seen: dict[str, Any] = {}

    async def turn(label: str, value: dict[str, Any]):
        set_domain_filter(value)
        # yield to scheduler to ensure interleaving
        await asyncio.sleep(0)
        seen[label] = get_domain_filter()

    await asyncio.gather(
        turn("a", {"system": "tim_mach"}),
        turn("b", {"system": "ho_hap"}),
    )
    assert seen["a"] == {"system": "tim_mach"}
    assert seen["b"] == {"system": "ho_hap"}


# -- Adapter taxonomy --------------------------------------------------------

@pytest.mark.asyncio
async def test_taxonomy_endpoints_cover_router_closed_set():
    transport = ASGITransport(app=adapter)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        systems = (await ac.get("/api/ontology/systems")).json()
        specialties = (await ac.get("/api/ontology/specialties")).json()

    sys_values = {s["value"] for s in systems}
    spec_values = {s["value"] for s in specialties}

    # Must include the closed-set values the router prompt + tagger rely on.
    for v in ["tim_mach", "ho_hap", "than_kinh", "nhi_khoa", "da_he"]:
        assert v in sys_values, f"system {v} missing from /ontology/systems"
    for v in ["noi", "ngoai", "cap_cuu", "general"]:
        assert v in spec_values, f"specialty {v} missing from /ontology/specialties"


# -- Chat adapter — request model accepts domain_filter ---------------------

@pytest.mark.asyncio
async def test_execute_chat_request_accepts_domain_filter():
    from src.agentrag.adapter.models import ExecuteChatRequest

    req = ExecuteChatRequest.model_validate(
        {
            "session_id": "abc",
            "message": "đau ngực",
            "context": {"sources": [], "notes": []},
            "domain_filter": {"system": "tim_mach", "specialties": ["noi"]},
        }
    )
    assert req.domain_filter == {"system": "tim_mach", "specialties": ["noi"]}


@pytest.mark.asyncio
async def test_execute_chat_request_domain_filter_optional():
    from src.agentrag.adapter.models import ExecuteChatRequest

    req = ExecuteChatRequest.model_validate(
        {
            "session_id": "abc",
            "message": "hello",
            "context": {"sources": [], "notes": []},
        }
    )
    assert req.domain_filter is None
