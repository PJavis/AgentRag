"""Static taxonomy endpoint smoke tests."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.agentrag.adapter.app import adapter


@pytest.mark.asyncio
async def test_get_systems_returns_taxonomy():
    transport = ASGITransport(app=adapter)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/ontology/systems")
        assert r.status_code == 200
        data = r.json()
        values = {item["value"] for item in data}
        assert "tim_mach" in values
        assert "ho_hap" in values


@pytest.mark.asyncio
async def test_get_specialties_returns_taxonomy():
    transport = ASGITransport(app=adapter)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/ontology/specialties")
        assert r.status_code == 200
        data = r.json()
        values = {item["value"] for item in data}
        assert "noi" in values
        assert "cap_cuu" in values
