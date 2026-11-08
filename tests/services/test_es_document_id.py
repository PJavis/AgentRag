"""Segments carry the Postgres document id; delete matches it, not the title.

Titles are client filenames: two users' "report.pdf" share the title "report".
Re-ingesting one used to run delete_by_query on document_title, which wiped
the other user's segments too.
"""
from __future__ import annotations

import pytest

from src.agentrag.ingestion.stores import elasticsearch_store as es_mod


class FakeClient:
    def __init__(self):
        self.deletes: dict[str, dict] = {}
        self.bulk_body = None

    async def delete_by_query(self, index, body, **kw):
        self.deletes[index] = body["query"]
        return {"deleted": 1}

    async def bulk(self, body, refresh):
        self.bulk_body = body
        return {"errors": False}


@pytest.fixture
def store():
    s = es_mod.ElasticsearchStore.__new__(es_mod.ElasticsearchStore)
    s.client = FakeClient()
    s.index_name = "seg"
    s.entity_index_name = "ent"
    s.relationship_index_name = "rel"

    async def _ensure(_dims):
        return None

    s.ensure_index = _ensure
    return s


async def test_delete_by_id_also_takes_this_titles_legacy_segments(store, monkeypatch):
    monkeypatch.setattr(es_mod.settings, "STRUCTMEM_INDEX", "mem", raising=False)
    await store.delete_document("report", document_id="D1", group_id="G1")
    seg = store.client.deletes["seg"]["bool"]
    assert seg["minimum_should_match"] == 1
    assert {"term": {"document_id": "D1"}} in seg["should"]
    legacy = [c for c in seg["should"] if "bool" in c][0]["bool"]
    assert legacy["must_not"] == [{"exists": {"field": "document_id"}}]
    assert store.client.deletes["mem"] == {"term": {"group_id": "G1"}}
    assert store.client.deletes["ent"] == {"term": {"document_title.keyword": "report"}}


async def test_shared_title_deletes_only_by_id(store, monkeypatch):
    monkeypatch.setattr(es_mod.settings, "STRUCTMEM_INDEX", "mem", raising=False)
    out = await store.delete_document("report", document_id="D1", group_id="G1", title_shared=True)
    assert store.client.deletes["seg"] == {
        "bool": {"should": [{"term": {"document_id": "D1"}}], "minimum_should_match": 1}
    }
    assert store.client.deletes["mem"] == {"term": {"group_id": "G1"}}
    # Entities / relationships are keyed by title only: the other document's.
    assert "ent" not in store.client.deletes and "rel" not in store.client.deletes
    assert out["ent"].startswith("kept")


async def test_shared_title_without_id_deletes_nothing(store, monkeypatch):
    monkeypatch.setattr(es_mod.settings, "STRUCTMEM_INDEX", "mem", raising=False)
    await store.delete_document("report", title_shared=True)
    assert store.client.deletes == {}


async def test_index_segments_stamps_document_id(store):
    await store.index_segments(
        [{"content": "c", "embedding": [0.1]}], "report", document_id="D1"
    )
    doc = store.client.bulk_body[1]
    assert doc["document_title"] == "report" and doc["document_id"] == "D1"


def test_index_mapping_declares_document_id():
    import inspect

    src = inspect.getsource(es_mod.ElasticsearchStore.ensure_index)
    src += inspect.getsource(es_mod.ElasticsearchStore._ensure_segment_fields)
    assert src.count('"document_id": {"type": "keyword"}') == 2
