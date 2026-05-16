"""S6 — activity endpoint privacy + admin scope (auth gate only — no DB hits)."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.agentrag.adapter.app import adapter


@pytest.mark.asyncio
async def test_personal_summary_requires_auth():
    transport = ASGITransport(app=adapter)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/activity/summary")
        assert r.status_code in (401, 403)


@pytest.mark.asyncio
async def test_personal_events_requires_auth():
    transport = ASGITransport(app=adapter)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/activity/events")
        assert r.status_code in (401, 403)


@pytest.mark.asyncio
async def test_admin_users_blocked_without_token():
    transport = ASGITransport(app=adapter)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/admin/activity/users")
        assert r.status_code in (401, 403)


@pytest.mark.asyncio
async def test_admin_events_blocked_without_token():
    transport = ASGITransport(app=adapter)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/admin/activity/events")
        assert r.status_code in (401, 403)
