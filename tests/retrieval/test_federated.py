"""FederatedRetriever smoke tests.

S4: router is now opt-in. Default (no router injected) returns hits with
no `domain_route` key. Tests use mocked base + router so they don't hit
live ES or LLM.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agentrag.orchestration.domain_router import DomainRoute
from src.agentrag.retrieval.federated import FederatedRetriever


def _mock_base() -> MagicMock:
    base = MagicMock()
    base.search = AsyncMock(return_value={"hits": []})
    return base


@pytest.mark.asyncio
async def test_explicit_system_override_skips_router():
    router = MagicMock()
    router.classify = AsyncMock()
    fr = FederatedRetriever(base=_mock_base(), router=router)
    out = await fr.search(
        query="anything",
        top_k=3,
        system_override="tim_mach",
        mode="hybrid",
    )
    router.classify.assert_not_called()
    assert "domain_route" not in out


@pytest.mark.asyncio
async def test_router_opt_in_when_injected():
    router = MagicMock()
    router.classify = AsyncMock(
        return_value=DomainRoute(systems=["tim_mach"], specialties=[], confidence=0.9, raw={})
    )
    fr = FederatedRetriever(base=_mock_base(), router=router)
    out = await fr.search(query="đau ngực", top_k=3, mode="hybrid")
    router.classify.assert_awaited_once()
    assert out["domain_route"]["systems"] == ["tim_mach"]


@pytest.mark.asyncio
async def test_no_router_no_auto_routing():
    """Without an injected router, no override → no domain filter applied."""
    base = _mock_base()
    fr = FederatedRetriever(base=base, router=None)
    out = await fr.search(query="đau ngực", top_k=3, mode="hybrid")
    assert "domain_route" not in out
    assert base.search.await_args.kwargs.get("filters") is None
