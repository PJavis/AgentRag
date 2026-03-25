from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from cachetools import LRUCache


class BaseEmbeddingProvider(ABC):
    def __init__(self, model: str, batch_size: int = 32, cache_size: int = 10_000):
        self.model = model
        self.batch_size = batch_size
        self.cache: LRUCache[str, list[float]] = LRUCache(maxsize=cache_size)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        pending_indices: list[int] = []
        pending_texts: list[str] = []

        for index, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                results[index] = cached
                continue
            pending_indices.append(index)
            pending_texts.append(text)

        if pending_texts:
            offset = 0
            for batch in self._batch_iter(pending_texts):
                batch_embeddings = await self._embed_batch(batch)
                if len(batch_embeddings) != len(batch):
                    raise RuntimeError(
                        f"{self.__class__.__name__} returned {len(batch_embeddings)} "
                        f"embeddings for {len(batch)} texts"
                    )
                for relative_index, embedding in enumerate(batch_embeddings):
                    original_index = pending_indices[offset + relative_index]
                    text = texts[original_index]
                    self.cache[text] = embedding
                    results[original_index] = embedding
                offset += len(batch)

        missing = [index for index, embedding in enumerate(results) if embedding is None]
        if missing:
            raise RuntimeError(
                f"{self.__class__.__name__} did not return embeddings for indexes {missing}"
            )
        return [embedding for embedding in results if embedding is not None]

    def _batch_iter(self, texts: list[str]) -> Iterable[list[str]]:
        for start in range(0, len(texts), self.batch_size):
            yield texts[start : start + self.batch_size]

    @abstractmethod
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
