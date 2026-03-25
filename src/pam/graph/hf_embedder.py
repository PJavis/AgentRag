from __future__ import annotations

import asyncio
from typing import Iterable

from graphiti_core.embedder.client import EmbedderClient
from huggingface_hub import InferenceClient


class HFGraphEmbedder(EmbedderClient):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = InferenceClient(model=model, token=api_key)

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        if isinstance(input_data, str):
            values = await self._feature_extraction([input_data])
            return values[0]
        if isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            values = await self._feature_extraction(input_data)
            return values[0]
        raise TypeError("HFGraphEmbedder expects string inputs")

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return await self._feature_extraction(input_data_list)

    async def _feature_extraction(self, texts: list[str]) -> list[list[float]]:
        response = await asyncio.to_thread(
            self.client.feature_extraction,
            text=texts,
            normalize=True,
            truncate=True,
        )
        if not isinstance(response, list):
            raise RuntimeError("HF graph embedding response was not a list")
        return response
