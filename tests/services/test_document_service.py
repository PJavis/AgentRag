"""One delete path: a deleted document must stop being retrievable everywhere.

Root `DELETE /documents/{id}` used to delete Postgres rows only, so the
document's segments stayed in Elasticsearch and `/chat` kept citing them.
"""
from __future__ import annotations

import ast
import inspect
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agentrag.services import document_service

REPO = Path(__file__).resolve().parents[2]


def document_service_group(source_id: str) -> str:
    from src.agentrag.graph.structmem_service import StructMemService

    return StructMemService.normalize_group_id(source_id)


class FakeSession:
    def __init__(self, docs: dict):
        self.docs = docs
        self.executed: list = []
        self.deleted: list = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, _model, key):
        return self.docs.get(key)

    async def execute(self, stmt):
        self.executed.append(stmt)

    async def delete(self, obj):
        self.deleted.append(obj)

    async def commit(self):
        self.commits += 1


@pytest.fixture
def es_calls(monkeypatch):
    calls: list[dict] = []

    class FakeES:
        async def delete_document(self, title, document_id=None, *, group_id=None, title_shared=False):
            calls.append({"title": title, "document_id": document_id,
                          "group_id": group_id, "title_shared": title_shared})
            return {"segments": 3}

    import src.agentrag.ingestion.stores.elasticsearch_store as es_mod

    monkeypatch.setattr(es_mod, "ElasticsearchStore", FakeES)
    return calls


@pytest.fixture
def bumps(monkeypatch):
    seen: list[int] = []
    import src.agentrag.common.corpus_version as cv

    monkeypatch.setattr(cv, "bump_corpus_version", lambda: seen.append(1) or "v2")
    return seen


@pytest.fixture
def title_free(monkeypatch):
    async def _free(_titles):
        return set()

    monkeypatch.setattr(document_service, "_titles_still_used", _free)
    monkeypatch.setattr(document_service, "_source_ids_still_used", _free)


async def test_delete_purges_es_images_and_bumps_cache(monkeypatch, tmp_path, es_calls, bumps, title_free):
    doc_id = uuid.uuid4()
    doc = SimpleNamespace(id=doc_id, title="Lec 10", source_id=f"{doc_id}/Lec 10.pdf")
    session = FakeSession({doc_id: doc})
    monkeypatch.setattr(document_service, "AsyncSessionLocal", lambda: session)
    monkeypatch.setattr(document_service.settings, "IMAGE_STORAGE_DIR", str(tmp_path))
    img_dir = tmp_path / "Lec_10"
    img_dir.mkdir()

    result = await document_service.delete_document(doc_id)

    assert result == {"document_id": str(doc_id), "title": "Lec 10", "purged": {"segments": 3}}
    assert session.deleted == [doc] and session.commits == 1
    assert len(session.executed) == 1  # the Segment delete
    assert es_calls == [{
        "title": "Lec 10", "document_id": str(doc_id),
        "group_id": document_service_group(f"{doc_id}/Lec 10.pdf"), "title_shared": False,
    }]
    assert not img_dir.exists()
    assert bumps == [1]  # cached answers citing the doc are re-keyed


async def test_delete_missing_returns_none_and_touches_nothing(monkeypatch, es_calls, bumps, title_free):
    monkeypatch.setattr(document_service, "AsyncSessionLocal", lambda: FakeSession({}))
    assert await document_service.delete_document(uuid.uuid4()) is None
    assert es_calls == [] and bumps == []


async def test_shared_title_purges_by_id_and_keeps_title_keyed_data(monkeypatch, tmp_path, es_calls, bumps):
    # Another document carries the same title: its segments must survive. The
    # deleted one's segments go by document_id; title-keyed data (legacy
    # segments, entities, images) is kept.
    async def _used(titles):
        return set(titles)

    async def _free(_ids):
        return set()

    monkeypatch.setattr(document_service, "_titles_still_used", _used)
    monkeypatch.setattr(document_service, "_source_ids_still_used", _free)
    monkeypatch.setattr(document_service.settings, "IMAGE_STORAGE_DIR", str(tmp_path))
    (tmp_path / "Shared").mkdir()
    ref = document_service.PurgeRef("d1", "Shared", "d1/Shared.pdf")
    await document_service.purge_document_artifacts([ref])
    assert es_calls == [{"title": "Shared", "document_id": "d1",
                         "group_id": document_service_group("d1/Shared.pdf"), "title_shared": True}]
    assert (tmp_path / "Shared").exists()
    assert bumps == [1]


async def test_es_failure_is_reported_not_raised(monkeypatch, bumps, title_free):
    class BrokenES:
        async def delete_document(self, title, document_id=None, **kw):
            raise ConnectionError("es down")

    import src.agentrag.ingestion.stores.elasticsearch_store as es_mod

    monkeypatch.setattr(es_mod, "ElasticsearchStore", BrokenES)
    purged = await document_service.purge_document_artifacts([document_service.PurgeRef("d", "T")])
    assert purged == {"d": "error: ConnectionError"}


async def test_in_session_delete_returns_refs_without_purging(es_calls):
    docs = [SimpleNamespace(id=uuid.uuid4(), title="A", source_id="a.pdf"),
            SimpleNamespace(id=uuid.uuid4(), title=None, source_id=None)]
    session = FakeSession({})
    refs = await document_service.delete_documents_in_session(session, docs)
    assert refs == [document_service.PurgeRef(str(docs[0].id), "A", "a.pdf"),
                    document_service.PurgeRef(str(docs[1].id), None, None)]
    assert session.deleted == docs and len(session.executed) == 2
    assert es_calls == []  # caller purges only after its commit


def _calls_in(func) -> set[str]:
    tree = ast.parse(inspect.cleandoc("\n" + inspect.getsource(func)))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return names


@pytest.mark.parametrize(
    "module,func,expected",
    [
        ("main", "delete_document", "delete_document"),
        ("src.agentrag.adapter.routers.sources", "delete_source", "delete_document"),
        ("src.agentrag.adapter.routers.notebooks", "delete_notebook", "purge_document_artifacts"),
        ("src.agentrag.adapter.account_deletion", "delete_user_data", "purge_document_artifacts"),
    ],
)
def test_every_delete_route_uses_the_service(module, func, expected):
    import importlib

    fn = getattr(importlib.import_module(module), func)
    assert expected in _calls_in(fn)


async def test_shared_structmem_group_is_not_purged(monkeypatch, es_calls, bumps):
    # Legacy uploads used the bare filename as source_id: two users' "report.pdf"
    # share one StructMem group. Deleting one must not wipe the other's memory.
    async def _none(_t):
        return set()

    async def _shared(ids):
        return set(ids)

    monkeypatch.setattr(document_service, "_titles_still_used", _none)
    monkeypatch.setattr(document_service, "_source_ids_still_used", _shared)
    await document_service.purge_document_artifacts(
        [document_service.PurgeRef("d1", "report", "report.pdf")]
    )
    assert es_calls[0]["group_id"] is None and es_calls[0]["document_id"] == "d1"
