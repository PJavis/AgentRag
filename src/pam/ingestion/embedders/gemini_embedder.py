from __future__ import annotations

import asyncio

import aiohttp

from .base import BaseEmbeddingProvider


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model: str, api_key: str, batch_size: int = 16):
        super().__init__(model=model, batch_size=batch_size)
        self.api_key = api_key
        self._url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:embedContent?key={self.api_key}"
        )

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with aiohttp.ClientSession() as session:
            tasks = [self._embed_single(session, text) for text in texts]
            return await asyncio.gather(*tasks)

    # Gemini embedding-001 giới hạn ~2048 tokens (~8000 ký tự)
    _TRUNCATE_SAFE_CHARS = 8_000

    async def _embed_single(
        self, session: aiohttp.ClientSession, text: str
    ) -> list[float]:
        payload = {
            "content": {
                "parts": [{"text": text[:self._TRUNCATE_SAFE_CHARS]}],
            }
        }
        async with session.post(self._url, json=payload) as response:
            if response.status >= 400:
                body = await response.text()
                raise RuntimeError(
                    f"Gemini embedding request failed ({response.status}): {body}"
                )
            data = await response.json()
        embedding = data.get("embedding", {})
        values = embedding.get("values")
        if not isinstance(values, list):
            raise RuntimeError("Gemini embedding response did not contain values")
        return values
