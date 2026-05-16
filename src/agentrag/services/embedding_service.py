"""EmbeddingService — Execution Plane facade for dense embedding (S4).

Wraps the existing `build_embedding_provider()` factory so Reasoning code
fetches one stable instance via ServiceContainer instead of constructing
embedders ad-hoc. Satisfies `EmbeddingProtocol`.
"""
from __future__ import annotations

from src.agentrag.config import Settings, settings as global_settings
from src.agentrag.ingestion.embedders.base import BaseEmbeddingProvider
from src.agentrag.ingestion.embedders.factory import build_embedding_provider


class EmbeddingService:
    """Stateless embedding facade — one provider per process."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._provider: BaseEmbeddingProvider = build_embedding_provider(
            settings or global_settings
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._provider.embed(texts)

    @property
    def model(self) -> str:
        return getattr(self._provider, "model", "unknown")
