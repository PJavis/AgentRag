"""Auth on the root app (main.py routes).

Before this, only the /on sub-app had auth; main.py's /chat, /ingest/folder,
DELETE /documents/{id}, /search, /mcp ... were open, and deploy/nginx.conf
proxies them publicly.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import main
from src.agentrag.adapter.auth import _is_public
from src.agentrag.config import settings

PASSWORD = "test-legacy-password"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "OPEN_NOTEBOOK_PASSWORD", PASSWORD)
    return TestClient(main.app)


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/chat"),
        ("POST", "/chat/stream"),
        ("POST", "/ingest/folder"),
        ("POST", "/search"),
        ("GET", "/documents"),
        ("DELETE", "/documents/some-id"),
        ("GET", "/conversations"),
        ("DELETE", "/conversations/some-id"),
        ("POST", "/generate/summary"),
        ("GET", "/metrics"),
        ("GET", "/config/validate"),
        ("GET", "/health/providers"),
        ("POST", "/mcp/"),
    ],
)
def test_root_routes_reject_missing_token(client, method, path):
    assert client.request(method, path, json={}).status_code == 401


def test_root_routes_reject_wrong_token(client):
    resp = client.post("/search", json={}, headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401


def test_root_routes_accept_legacy_password(client):
    resp = client.post("/search", json={}, headers={"Authorization": f"Bearer {PASSWORD}"})
    assert resp.status_code != 401


def test_openapi_stays_public(client):
    assert client.get("/openapi.json").status_code == 200


def test_adapter_public_route_still_public(client):
    assert client.get("/on/api/auth/status").status_code != 401


def test_adapter_private_route_still_protected(client):
    assert client.get("/on/api/notebooks").status_code == 401


def test_auth_disabled_leaves_root_open(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ENABLED", False)
    assert TestClient(main.app).post("/search", json={}).status_code != 401


@pytest.mark.parametrize(
    "path",
    ["/on/api/config", "/on/api/auth/login", "/on/api/images/a.png", "/on/health", "/docs", "/images/x.png", "/"],
)
def test_public_paths(path):
    assert _is_public(path)


@pytest.mark.parametrize(
    "path",
    [
        # Used to be public: matching was endswith / substring.
        "/on/api/sources/abc/health",
        "/on/api/notebooks/x/docs",
        "/on/api/sources/x/admin/y",
        "/documents/health",
        "/on/api/chat/execute",
        "/chat",
    ],
)
def test_private_paths(path):
    assert not _is_public(path)
