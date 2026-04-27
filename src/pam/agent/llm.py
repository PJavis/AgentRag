from __future__ import annotations

import json
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from src.pam.config import settings


class AgentLLM:
    def __init__(self, model_override: str | None = None) -> None:
        self.model, self.base_url, self.api_key = self._resolve_backend()
        if model_override:
            self.model = model_override
        self.temperature = (
            settings.AGENT_TEMPERATURE
            if settings.AGENT_TEMPERATURE is not None
            else settings.EXTRACTION_TEMPERATURE
        )
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def json_response(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        result = json.loads(content)
        # Some providers return a JSON array instead of an object; unwrap if needed
        if isinstance(result, list):
            result = result[0] if result and isinstance(result[0], dict) else {}
        elif not isinstance(result, dict):
            result = {}
        return result

    async def stream_text(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        """Stream raw text tokens từ LLM (không ép JSON)."""
        stream = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    def _resolve_backend(self) -> tuple[str, str | None, str]:
        provider = settings.AGENT_PROVIDER or settings.EXTRACTION_PROVIDER
        model = settings.AGENT_MODEL or settings.EXTRACTION_MODEL
        base_override = settings.AGENT_BASE_URL

        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required when AGENT/EXTRACTION provider is openai")
            return model, base_override or settings.EXTRACTION_BASE_URL, settings.OPENAI_API_KEY
        if provider == "ollama":
            return (
                model,
                base_override or settings.EXTRACTION_BASE_URL or settings.OLLAMA_BASE_URL,
                settings.OLLAMA_API_KEY,
            )
        if provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is required when AGENT/EXTRACTION provider is gemini")
            return (
                model,
                base_override
                or settings.EXTRACTION_BASE_URL
                or "https://generativelanguage.googleapis.com/v1beta/openai/",
                settings.GEMINI_API_KEY,
            )
        if provider == "hf_inference":
            if not settings.HF_TOKEN:
                raise ValueError("HF_TOKEN is required when AGENT/EXTRACTION provider is hf_inference")
            return (
                model,
                base_override or settings.EXTRACTION_BASE_URL or settings.HF_OPENAI_BASE_URL,
                settings.HF_TOKEN,
            )
        raise ValueError(f"Unsupported agent provider: {provider}")
