from __future__ import annotations

import math

import openai
from openai import AsyncOpenAI

from .base import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        batch_size: int = 64,
    ):
        super().__init__(model=model, batch_size=batch_size)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self.client.embeddings.create(model=self.model, input=texts)
        except openai.InternalServerError as exc:
            message = str(exc)
            if "unsupported value: NaN" in message and self._is_ollama():
                raise RuntimeError(
                    "Ollama embedding model returned NaN values and the server failed "
                    "to encode the response. This is usually a model/runtime issue, "
                    "not a client bug. Change EMBEDDING_MODEL to a stable Ollama "
                    "embedding model such as 'nomic-embed-text', "
                    "'nomic-embed-text-v2-moe', or 'mxbai-embed-large', then retry."
                ) from exc
            raise

        embeddings = [item.embedding for item in response.data]
        if not self._all_finite(embeddings):
            if self._is_ollama():
                raise RuntimeError(
                    "Ollama embedding response contained non-finite values. "
                    "Change EMBEDDING_MODEL to a stable embedding model such as "
                    "'nomic-embed-text', 'nomic-embed-text-v2-moe', or "
                    "'mxbai-embed-large'."
                )
            raise RuntimeError(
                f"{self.model} returned non-finite embedding values"
            )
        return embeddings

    def _is_ollama(self) -> bool:
        return bool(self.base_url and "11434" in self.base_url)

    def _all_finite(self, embeddings: list[list[float]]) -> bool:
        for embedding in embeddings:
            for value in embedding:
                if not math.isfinite(value):
                    return False
        return True
