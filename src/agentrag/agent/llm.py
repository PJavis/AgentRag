from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from openai import AsyncOpenAI, NotFoundError

from src.agentrag.config import settings
from src.agentrag.observability.cost import record_llm_call


def _is_model_missing(exc: Exception) -> bool:
    """Detect 'model not found' across Ollama / OpenAI / vLLM error shapes."""
    if isinstance(exc, NotFoundError):
        return True
    msg = str(exc).lower()
    return "not found" in msg or "model_not_found" in msg or "does not exist" in msg


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
        # Sticky fallback: once we miss the primary model we keep using the
        # fallback for the rest of the process. Avoids re-hitting the 404.
        self._fallback_engaged = False

    async def _create(self, **kwargs):
        """Wrap client.chat.completions.create with model-missing fallback."""
        try:
            return await self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not _is_model_missing(exc):
                raise
            fallback = (settings.LLM_FALLBACK_MODEL or "").strip()
            if not fallback or fallback == kwargs.get("model"):
                raise
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "AgentLLM: model %r not found, falling back to %r",
                kwargs.get("model"), fallback,
            )
            kwargs["model"] = fallback
            self.model = fallback  # sticky
            self._fallback_engaged = True
            return await self.client.chat.completions.create(**kwargs)

    async def json_response(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        response = await self._create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency_ms = (time.perf_counter() - started) * 1000
        raw = response.choices[0].message.content or "{}"
        record_llm_call(
            task="json", model=self.model, latency_ms=latency_ms,
            in_text=system_prompt + user_prompt, out_text=raw,
            usage=getattr(response, "usage", None),
        )
        # Strip <think>...</think> chain-of-thought from reasoning models
        # (DeepSeek-R1, QwQ, etc.) so the residual JSON parses cleanly.
        from src.agentrag.common.thinking import clean_thinking_content
        cleaned = clean_thinking_content(raw)
        # Fallback: if cleaning emptied the response (model returned only
        # <think>…</think> with no answer body), keep raw so json.loads at
        # least surfaces a useful error instead of silently returning {}.
        content = cleaned if cleaned.strip() else raw
        # Extract first balanced JSON object in case of leading prose.
        if content and not content.lstrip().startswith("{"):
            idx = content.find("{")
            if idx >= 0:
                content = content[idx:]
        try:
            result = json.loads(content or "{}")
        except json.JSONDecodeError as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "json_response parse failed (model=%s): %s | raw[:300]=%r",
                self.model, e, raw[:300],
            )
            result = {}
        # Some providers return a JSON array instead of an object; unwrap if needed
        if isinstance(result, list):
            result = result[0] if result and isinstance(result[0], dict) else {}
        elif not isinstance(result, dict):
            result = {}
        return result

    async def text_response(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Plain text response — no JSON enforcement. Used by transformations."""
        started = time.perf_counter()
        response = await self._create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency_ms = (time.perf_counter() - started) * 1000
        raw = response.choices[0].message.content or ""
        record_llm_call(
            task="text", model=self.model, latency_ms=latency_ms,
            in_text=system_prompt + user_prompt, out_text=raw,
            usage=getattr(response, "usage", None),
        )
        from src.agentrag.common.thinking import clean_thinking_content
        return clean_thinking_content(raw)

    async def stream_text(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        """Stream raw text tokens từ LLM (không ép JSON)."""
        started = time.perf_counter()
        stream = await self._create(
            model=self.model,
            temperature=self.temperature,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        buf: list[str] = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                buf.append(delta)
                yield delta
        latency_ms = (time.perf_counter() - started) * 1000
        record_llm_call(
            task="stream", model=self.model, latency_ms=latency_ms,
            in_text=system_prompt + user_prompt, out_text="".join(buf),
        )

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
