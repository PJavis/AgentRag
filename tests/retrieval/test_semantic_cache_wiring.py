import asyncio
from src.agentrag.retrieval import elasticsearch_retriever as er


def test_second_similar_query_served_from_semantic_cache(monkeypatch):
    monkeypatch.setattr(er.settings, "SEMANTIC_CACHE_ENABLED", True)
    monkeypatch.setattr(er.settings, "RETRIEVAL_RERANK_ENABLED", False)

    r = er.ElasticsearchRetriever.__new__(er.ElasticsearchRetriever)
    r._last_rerank_reason = "not_attempted"
    r._semantic_cache = er.SemanticCache(threshold=0.97, ttl_seconds=100, max_items=8, clock=lambda: 0.0)

    calls = {"n": 0}

    class _FakeEmbedder:
        async def embed(self, texts):  # noqa: ANN001
            return [[1.0, 0.0, 0.0]]

    async def fake_impl(**kwargs):  # noqa: ANN001
        calls["n"] += 1
        return {"results": [{"rank": 1, "content": "X"}], "mode": kwargs["mode"], "top_k": 10}

    r.embedder = _FakeEmbedder()
    r._search_uncached = fake_impl  # type: ignore[assignment]

    out1 = asyncio.run(r.search_cached("q1", mode="hybrid", top_k=10))
    out2 = asyncio.run(r.search_cached("q1 rephrased", mode="hybrid", top_k=10))
    assert out1["results"] == out2["results"]
    assert calls["n"] == 1  # second served from semantic cache
