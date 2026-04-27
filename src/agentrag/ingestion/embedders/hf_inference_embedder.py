from __future__ import annotations

import asyncio

from huggingface_hub import InferenceClient

from .base import BaseEmbeddingProvider


class HFInferenceEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model: str, api_key: str, batch_size: int = 32):
        super().__init__(model=model, batch_size=batch_size)
        self.client = InferenceClient(model=model, token=api_key)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await asyncio.to_thread(
                self.client.feature_extraction,
                text=texts,
                normalize=True,
                truncate=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Hugging Face inference error: {exc}") from exc
        return response
