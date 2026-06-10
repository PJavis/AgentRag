from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore


def test_index_segments_doc_body_includes_new_fields(monkeypatch):
    store = ElasticsearchStore.__new__(ElasticsearchStore)  # no ES client
    store.index_name = "test_segments"
    captured = {}

    async def fake_ensure_index(dims):  # noqa: ANN001
        return None

    async def fake_bulk(body, refresh):  # noqa: ANN001
        captured["body"] = body
        return {"errors": False}

    class _FakeClient:
        async def bulk(self, body, refresh):  # noqa: ANN001
            return await fake_bulk(body, refresh)

    store.client = _FakeClient()
    store.ensure_index = fake_ensure_index  # type: ignore[assignment]

    chunks = [{
        "content": "leaf text",
        "context_text": "This passage is from the cardiology chapter on MI.",
        "embedding": [0.1, 0.2],
        "node_level": 0,
        "child_ids": [],
        "content_hash": "abc",
    }]
    import asyncio
    asyncio.run(store.index_segments(chunks, "Doc A"))

    doc_body = captured["body"][1]  # [action, doc, action, doc, ...]
    assert doc_body["context_text"] == "This passage is from the cardiology chapter on MI."
    assert doc_body["node_level"] == 0
    assert doc_body["child_ids"] == []
