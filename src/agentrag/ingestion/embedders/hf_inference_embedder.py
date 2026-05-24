from __future__ import annotations

import asyncio
import logging

from huggingface_hub import InferenceClient

from .base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


def _is_transient_hf(exc: Exception) -> bool:
    """Match HF Inference transient errors worth retrying (504/503/429/502)."""
    s = str(exc)
    return any(code in s for code in ("504", "503", "429", "502", "Gateway", "timeout", "Timeout"))


class HFInferenceEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model: str, api_key: str, batch_size: int = 16):
        super().__init__(model=model, batch_size=batch_size)
        self.client = InferenceClient(model=model, token=api_key)
        self._max_retries = 4

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        attempt = 0
        last_exc: Exception | None = None
        while attempt < self._max_retries:
            try:
                response = await asyncio.to_thread(
                    self.client.feature_extraction,
                    text=texts,
                    normalize=True,
                    truncate=True,
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries - 1 and _is_transient_hf(exc):
                    backoff = min(2 ** attempt * 2, 30)
                    logger.warning(
                        "HF embed transient error (attempt %d/%d): %s — retry in %ds",
                        attempt + 1, self._max_retries, str(exc)[:120], backoff,
                    )
                    attempt += 1
                    await asyncio.sleep(backoff)
                    continue
                raise RuntimeError(f"Hugging Face inference error: {exc}") from exc
        else:
            raise RuntimeError(
                f"Hugging Face inference error after {self._max_retries} retries: {last_exc}"
            ) from last_exc

        # HF returns numpy ndarray (2D for batch, 1D for single text) — convert
        # to plain list[list[float]] so downstream `isinstance(v, list)` checks
        # in ES indexing pass.
        try:
            import numpy as np  # type: ignore
            if isinstance(response, np.ndarray):
                if response.ndim == 1:
                    return [response.tolist()]
                return response.tolist()
        except ImportError:
            pass
        if hasattr(response, "tolist"):
            arr = response.tolist()
            if arr and not isinstance(arr[0], list):
                return [arr]
            return arr
        return response
