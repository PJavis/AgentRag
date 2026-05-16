"""FederatedRetriever smoke tests."""
from __future__ import annotations

import pytest

from src.agentrag.retrieval.federated import FederatedRetriever


@pytest.mark.asyncio
async def test_explicit_system_override_skips_router():
    fr = FederatedRetriever()
    out = await fr.search(
        query="anything",
        top_k=3,
        system_override="tim_mach",
        mode="hybrid",
    )
    # When override given, no router consultation → no domain_route in payload
    assert "domain_route" not in out


@pytest.mark.asyncio
async def test_no_override_consults_router():
    fr = FederatedRetriever()
    out = await fr.search(query="đau ngực", top_k=3, mode="hybrid")
    assert "domain_route" in out
    assert isinstance(out["domain_route"].get("systems"), list)
