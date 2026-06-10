from src.agentrag.retrieval.elasticsearch_retriever import _cache_key


def test_cache_key_distinguishes_dense_query():
    # Same base query, different HyDE-augmented dense_query → MUST differ,
    # else a HyDE result collides with a non-HyDE result.
    k_plain = _cache_key("đau ngực", "hybrid", 10, None, True, dense_query=None)
    k_hyde = _cache_key("đau ngực", "hybrid", 10, None, True, dense_query="đau ngực do nhồi máu cơ tim ...")
    assert k_plain != k_hyde
