from __future__ import annotations

import math

import openai
from openai import AsyncOpenAI

from .base import BaseEmbeddingProvider

# Thứ tự giảm dần khi model báo "input too long".
# mxbai-embed-large: 512-token context; ~2-3 chars/token tiếng Việt → ~1024-1536 chars.
# nomic-embed-text:  2048-token context → ~8192 chars.
# Dùng vòng lặp để tự tìm giới hạn phù hợp với từng model.
_TRUNCATE_STEPS = [6000, 3000, 1500, 800, 400]


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
        except openai.BadRequestError as exc:
            message = str(exc).lower()
            if "context length" in message or "input length" in message:
                # Một hoặc nhiều text trong batch quá dài cho model.
                # Retry từng text riêng lẻ sau khi truncate.
                return await self._embed_one_by_one(texts)
            raise
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

    async def _embed_one_by_one(self, texts: list[str]) -> list[list[float]]:
        """Fallback: embed từng text riêng, giảm dần độ dài cho đến khi model chấp nhận."""
        return [await self._embed_single_with_truncate(t) for t in texts]

    async def _embed_single_with_truncate(self, text: str) -> list[float]:
        for char_limit in _TRUNCATE_STEPS:
            try:
                response = await self.client.embeddings.create(
                    model=self.model, input=[text[:char_limit]]
                )
                return response.data[0].embedding
            except openai.BadRequestError as exc:
                msg = str(exc).lower()
                if "context length" not in msg and "input length" not in msg:
                    raise
                # vẫn quá dài → thử bước tiếp theo
                continue
        # Bước cuối cùng: 200 ký tự đầu — đủ để tạo embedding không rỗng
        response = await self.client.embeddings.create(
            model=self.model, input=[text[:200]]
        )
        return response.data[0].embedding

    def _is_ollama(self) -> bool:
        return bool(self.base_url and "11434" in self.base_url)

    def _all_finite(self, embeddings: list[list[float]]) -> bool:
        for embedding in embeddings:
            for value in embedding:
                if not math.isfinite(value):
                    return False
        return True
