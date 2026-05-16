"""EmbeddingService — Execution Plane facade for dense embedding (S4 + S3).

S4: wraps the existing `build_embedding_provider()` factory so Reasoning
code fetches one stable instance via ServiceContainer instead of
constructing embedders ad-hoc. Satisfies `EmbeddingProtocol`.

S3: TTL cache keyed by SHA-256 of input text. Hits return the cached
vector without provider call. Effective on hot query paths (HyDE
rewrites, repeated sub-queries, ES retriever short-circuit chains).
Large batches bypass the cache so ingestion isn't penalised.
"""
from __future__ import annotations

import hashlib
from typing import Any

from cachetools import TTLCache

from src.agentrag.config import Settings, settings as global_settings
from src.agentrag.ingestion.embedders.base import BaseEmbeddingProvider
from src.agentrag.ingestion.embedders.factory import build_embedding_provider


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class EmbeddingService:
    """Stateless embedding facade — one provider per process."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        cache_size: int = 2048,
        cache_ttl_s: int = 600,
        cache_max_batch: int = 8,
    ) -> None:
        self._provider: BaseEmbeddingProvider = build_embedding_provider(
            settings or global_settings
        )
        # Cache only small batches (query path). Large batches = ingestion,
        # bypass to avoid memory pressure + duplicate work.
        self._cache_max_batch = cache_max_batch
        self._cache: TTLCache[str, list[float]] = TTLCache(
            maxsize=cache_size, ttl=cache_ttl_s
        )
        self._stats: dict[str, int] = {"hits": 0, "misses": 0, "skips": 0}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        n = len(texts)
        if n > self._cache_max_batch:
            self._stats["skips"] += n
            return await self._provider.embed(texts)

        keys = [_hash(t) for t in texts]
        result: list[list[float] | None] = [None] * n
        missing_idx: list[int] = []
        missing_text: list[str] = []
        for i, key in enumerate(keys):
            cached = self._cache.get(key)
            if cached is not None:
                self._stats["hits"] += 1
                result[i] = cached
            else:
                self._stats["misses"] += 1
                missing_idx.append(i)
                missing_text.append(texts[i])

        if missing_text:
            fresh = await self._provider.embed(missing_text)
            for j, vec in zip(missing_idx, fresh, strict=True):
                self._cache[keys[j]] = vec
                result[j] = vec

        # By construction every slot is now filled.
        return [v for v in result if v is not None]

    @property
    def model(self) -> str:
        return getattr(self._provider, "model", "unknown")

    @property
    def cache_stats(self) -> dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total) if total else 0.0
        return {
            **self._stats,
            "size": len(self._cache),
            "maxsize": self._cache.maxsize,
            "hit_rate": round(hit_rate, 4),
        }

    def reset_cache(self) -> None:
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0, "skips": 0}
