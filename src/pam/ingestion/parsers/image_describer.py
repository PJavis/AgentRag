"""
ImageDescriber: gọi VLM (Vision Language Model) để mô tả ảnh tại ingest time.

Mô tả được chèn vào markdown content dưới dạng:
  <!-- image_N: <mô tả ảnh> -->

Các chunk chứa image description sẽ có segment_type="image_description"
(được set bởi pipeline khi detect dòng <!-- image_*: ... -->).

Providers hỗ trợ:
  gemini  — gemini-2.0-flash, gemini-1.5-pro (có vision API riêng)
  openai  — gpt-4o, gpt-4o-mini
  ollama  — llava, llava-phi3, bakllava (dùng /api/chat với image base64)
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Literal

import aiohttp

logger = logging.getLogger(__name__)

ProviderType = Literal["gemini", "openai", "ollama"]


class ImageDescriber:
    def __init__(
        self,
        provider: ProviderType,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or self._default_base_url()

    def _default_base_url(self) -> str:
        if self.provider == "gemini":
            return "https://generativelanguage.googleapis.com/v1beta"
        if self.provider == "openai":
            return "https://api.openai.com/v1"
        return "http://127.0.0.1:11434"  # ollama

    async def describe(self, image_bytes: bytes, context: str = "") -> str:
        """
        Mô tả nội dung ảnh bằng VLM.
        context: đoạn text xung quanh ảnh để VLM hiểu ngữ cảnh.
        Trả về chuỗi mô tả, hoặc "" nếu thất bại.
        """
        try:
            if self.provider == "gemini":
                return await self._describe_gemini(image_bytes, context)
            if self.provider == "openai":
                return await self._describe_openai(image_bytes, context)
            if self.provider == "ollama":
                return await self._describe_ollama(image_bytes, context)
        except Exception as exc:
            logger.warning("ImageDescriber failed: %s", exc)
        return ""

    async def describe_batch(
        self,
        images: list[bytes],
        contexts: list[str] | None = None,
    ) -> list[str]:
        """Mô tả nhiều ảnh song song."""
        contexts = contexts or [""] * len(images)
        return list(
            await asyncio.gather(
                *[self.describe(img, ctx) for img, ctx in zip(images, contexts)],
                return_exceptions=False,
            )
        )

    # ── Gemini ──────────────────────────────────────────────────────────────

    async def _describe_gemini(self, image_bytes: bytes, context: str) -> str:
        b64 = base64.b64encode(image_bytes).decode()
        # Detect mime type từ magic bytes
        mime = _detect_mime(image_bytes)
        prompt = _build_prompt(context)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": mime, "data": b64}},
                    ]
                }
            ]
        }
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"Gemini vision error {resp.status}: {await resp.text()}")
                data = await resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    # ── OpenAI ──────────────────────────────────────────────────────────────

    async def _describe_openai(self, image_bytes: bytes, context: str) -> str:
        b64 = base64.b64encode(image_bytes).decode()
        mime = _detect_mime(image_bytes)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _build_prompt(context)},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 400,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"OpenAI vision error {resp.status}: {await resp.text()}")
                data = await resp.json()
        return data["choices"][0]["message"]["content"].strip()

    # ── Ollama ───────────────────────────────────────────────────────────────

    async def _describe_ollama(self, image_bytes: bytes, context: str) -> str:
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": _build_prompt(context),
                    "images": [b64],
                }
            ],
            "stream": False,
        }
        url = f"{self.base_url}/api/chat"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"Ollama vision error {resp.status}: {await resp.text()}")
                data = await resp.json()
        return data["message"]["content"].strip()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_prompt(context: str) -> str:
    base = (
        "Describe this image concisely and accurately. "
        "Include all text, numbers, labels, chart types, data values, "
        "and visual relationships visible in the image. "
        "If it is a chart or graph, describe the data it shows. "
        "If it is a diagram, describe the components and their relationships. "
        "Be specific enough that someone who cannot see the image can understand the data."
    )
    if context.strip():
        return f"Context from the surrounding document:\n\"{context[:500]}\"\n\n{base}"
    return base


def _detect_mime(data: bytes) -> str:
    """Detect MIME type từ magic bytes."""
    if data[:4] == b"\x89PNG":
        return "image/png"
    if data[:3] == b"GIF":
        return "image/gif"
    if data[:4] in (b"RIFF", b"WEBP"):
        return "image/webp"
    return "image/jpeg"  # default


def build_image_describer_from_settings(settings) -> ImageDescriber | None:
    """Factory function dùng trong pipeline."""
    if not settings.VISION_ENABLED:
        return None
    api_key = None
    if settings.VISION_PROVIDER == "gemini":
        api_key = settings.GEMINI_API_KEY
    elif settings.VISION_PROVIDER == "openai":
        api_key = settings.OPENAI_API_KEY
    return ImageDescriber(
        provider=settings.VISION_PROVIDER,
        model=settings.VISION_MODEL,
        api_key=api_key,
        base_url=settings.VISION_BASE_URL,
    )
