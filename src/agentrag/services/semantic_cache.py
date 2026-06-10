from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Any, Callable


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SemanticCache:
    """In-process retrieval cache keyed by query embedding similarity.

    Tier-2 cache: complements the exact-key TTLCache in ElasticsearchRetriever.
    A lookup embeds the query, scans recent entries, and returns the cached
    payload of the most-similar entry whose cosine >= threshold and not expired.
    LRU-bounded by max_items; TTL-bounded by ttl_seconds. Per-worker (not
    distributed) — conservative threshold keeps false hits negligible.
    """

    def __init__(
        self,
        threshold: float,
        ttl_seconds: int,
        max_items: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._max = max_items
        self._clock = clock
        # key insertion-order -> (embedding, payload, stored_at)
        self._store: "OrderedDict[int, tuple[list[float], Any, float]]" = OrderedDict()
        self._next_id = 0

    def get(self, embedding: list[float]) -> Any | None:
        now = self._clock()
        best_key: int | None = None
        best_sim = self._threshold
        for key, (emb, _payload, stored_at) in list(self._store.items()):
            if now - stored_at > self._ttl:
                del self._store[key]
                continue
            sim = _cosine(embedding, emb)
            if sim >= best_sim:
                best_sim = sim
                best_key = key
        if best_key is None:
            return None
        self._store.move_to_end(best_key)  # mark as recently used
        return self._store[best_key][1]

    def put(self, embedding: list[float], payload: Any) -> None:
        self._store[self._next_id] = (embedding, payload, self._clock())
        self._next_id += 1
        while len(self._store) > self._max:
            self._store.popitem(last=False)  # evict oldest
