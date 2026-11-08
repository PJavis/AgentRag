"""Root routes signal bad input with HTTP status codes, not {"error": ...} + 200.

A 200 with an error body was counted as success by an eval run (19 rejected calls
reported as "19 answered, 0 failed").
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

import main
from src.agentrag.config import settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ENABLED", False)
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    "path,body,needle",
    [
        ("/chat", {}, "question"),
        ("/chat/stream", {}, "question"),
        ("/search", {}, "query"),
        ("/ingest/folder", {}, "folder_path"),
        ("/ingest/folder", {"folder_path": "/x", "graph_ingest_mode": "later"}, "graph_ingest_mode"),
    ],
)
def test_missing_or_bad_input_is_400(client, path, body, needle):
    resp = client.post(path, json=body)
    assert resp.status_code == 400
    assert needle in resp.json()["detail"]


@pytest.mark.parametrize(
    "method,path",
    [("DELETE", "/documents/not-a-uuid"), ("GET", "/documents/not-a-uuid/graph-status")],
)
def test_non_uuid_document_id_is_404_not_500(client, method, path):
    # Used to reach Postgres and fail with InvalidTextRepresentation → 500.
    assert client.request(method, path).status_code == 404


def test_upload_filename_cannot_escape_temp_dir(client, monkeypatch):
    seen = {}

    async def fake_ingest_folder(folder):
        import os

        seen["files"] = sorted(os.listdir(folder))
        return {"ok": True}

    monkeypatch.setattr(main, "ingest_folder", fake_ingest_folder)
    resp = client.post(
        "/ingest/upload",
        files={"file": ("../../escaped.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 200
    assert seen["files"] == ["escaped.txt"]
