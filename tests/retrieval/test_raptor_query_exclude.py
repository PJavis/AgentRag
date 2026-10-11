"""RETRIEVAL_EXCLUDE_RAPTOR_SUMMARIES drops RAPTOR summary nodes from sparse and
dense search at query time, so RAPTOR can be A/B-tested without a re-ingest."""
from __future__ import annotations

import asyncio
import json

from src.agentrag.config import settings
from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore

_EXCLUDE = {"bool": {"must_not": [{"term": {"segment_type": "raptor_summary"}}]}}


def _store():
    store = ElasticsearchStore.__new__(ElasticsearchStore)  # no ES client
    store.index_name = "test_segments"
    sent: list[dict] = []

    class _FakeClient:
        async def search(self, **body):
            sent.append(body)
            return {"hits": {"hits": []}}

    store.client = _FakeClient()
    return store, sent


def _has_exclusion(body: dict) -> bool:
    return json.dumps(_EXCLUDE) in json.dumps(body)


def test_sparse_and_dense_exclude_raptor_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_EXCLUDE_RAPTOR_SUMMARIES", True)
    store, sent = _store()
    asyncio.run(store.sparse_search("sốt xuất huyết", top_k=5))
    asyncio.run(store.dense_search([0.1, 0.2], top_k=5))
    assert len(sent) == 2
    assert all(_has_exclusion(b) for b in sent)


def test_exclusion_combines_with_document_filter(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_EXCLUDE_RAPTOR_SUMMARIES", True)
    store, sent = _store()
    asyncio.run(store.dense_search([0.1, 0.2], top_k=5, document_title="Nhi khoa"))
    knn_filter = sent[0]["knn"]["filter"]
    assert _has_exclusion(knn_filter)
    assert "Nhi khoa" in json.dumps(knn_filter, ensure_ascii=False)


def test_raptor_included_by_default(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_EXCLUDE_RAPTOR_SUMMARIES", False)
    store, sent = _store()
    asyncio.run(store.sparse_search("sốt xuất huyết", top_k=5))
    asyncio.run(store.dense_search([0.1, 0.2], top_k=5))
    assert not any(_has_exclusion(b) for b in sent)
