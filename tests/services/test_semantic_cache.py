from src.agentrag.services.semantic_cache import SemanticCache


def test_returns_hit_for_near_identical_embedding():
    cache = SemanticCache(threshold=0.97, ttl_seconds=100, max_items=8, clock=lambda: 0.0)
    cache.put([1.0, 0.0, 0.0], {"results": ["A"]})
    # cosine([1,0,0],[0.999,0.001,0]) ≈ 1.0 ≥ 0.97 → hit
    hit = cache.get([0.999, 0.001, 0.0])
    assert hit == {"results": ["A"]}


def test_miss_for_dissimilar_embedding():
    cache = SemanticCache(threshold=0.97, ttl_seconds=100, max_items=8, clock=lambda: 0.0)
    cache.put([1.0, 0.0, 0.0], {"results": ["A"]})
    assert cache.get([0.0, 1.0, 0.0]) is None


def test_entry_expires_after_ttl():
    now = {"t": 0.0}
    cache = SemanticCache(threshold=0.97, ttl_seconds=10, max_items=8, clock=lambda: now["t"])
    cache.put([1.0, 0.0, 0.0], {"results": ["A"]})
    now["t"] = 11.0
    assert cache.get([1.0, 0.0, 0.0]) is None


def test_lru_eviction_beyond_max_items():
    cache = SemanticCache(threshold=0.99, ttl_seconds=100, max_items=2, clock=lambda: 0.0)
    cache.put([1.0, 0.0], {"results": ["A"]})
    cache.put([0.0, 1.0], {"results": ["B"]})
    cache.put([1.0, 1.0], {"results": ["C"]})  # evicts oldest ([1,0])
    assert cache.get([1.0, 0.0]) is None
    assert cache.get([0.0, 1.0]) == {"results": ["B"]}
