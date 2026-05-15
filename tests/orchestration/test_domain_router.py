"""DomainRouter smoke. Requires running LLM gateway."""
from __future__ import annotations

import pytest

from src.agentrag.orchestration.domain_router import DomainRoute, DomainRouter


@pytest.mark.asyncio
async def test_classify_returns_route():
    router = DomainRouter()
    r = await router.classify("Đau ngực kèm khó thở ở bệnh nhân 60 tuổi")
    assert isinstance(r, DomainRoute)
    # accept either tim_mach or ho_hap or both; just verify model returns something
    assert 0.0 <= r.confidence <= 1.0
