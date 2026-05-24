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
    def __init__(
        self,
        model_override: str | None = None,
        provider_override: str | None = None,
    ) -> None:
        # Mixed-provider routing: when model name suggests cloud (gemini-* /
        # gemma-* / gpt-* / claude-*) auto-derive provider so a per-task entry
        # in LLM_TASK_MODEL_MAP can target a different backend than the
        # global AGENT_PROVIDER.
        if provider_override is None and model_override:
            mlow = model_override.lower()
            if mlow.startswith("gemini-") or mlow.startswith("gemma-"):
                provider_override = "gemini"
            elif mlow.startswith("gpt-") or mlow.startswith("o1") or mlow.startswith("o3"):
                provider_override = "openai"
            # else: stay on default provider (Ollama / whatever AGENT_PROVIDER set)
        if provider_override:
            self.model, self.base_url, self.api_key = self._resolve_backend_for(
                provider_override, model_override
            )
        else:
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

    def _resolve_backend_for(self, provider: str, model: str | None) -> tuple[str, str | None, str]:
        """Resolve backend for an arbitrary provider (used by per-task routing
        to mix providers — e.g. local Ollama for orchestration + Gemini for
        the final answer)."""
        if provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY required for Gemini-routed task")
            return (
                model or settings.AGENT_MODEL or settings.EXTRACTION_MODEL,
                "https://generativelanguage.googleapis.com/v1beta/openai/",
                settings.GEMINI_API_KEY,
            )
        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY required for OpenAI-routed task")
            return (model or "gpt-4o-mini", None, settings.OPENAI_API_KEY)
        if provider == "ollama":
            return (
                model or settings.AGENT_MODEL or settings.EXTRACTION_MODEL,
                settings.OLLAMA_BASE_URL,
                settings.OLLAMA_API_KEY,
            )
        if provider == "hf_inference":
            if not settings.HF_TOKEN:
                raise ValueError("HF_TOKEN required for HF-routed task")
            return (
                model or settings.AGENT_MODEL or settings.EXTRACTION_MODEL,
                settings.HF_OPENAI_BASE_URL,
                settings.HF_TOKEN,
            )
        raise ValueError(f"Unknown provider: {provider}")

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

    async def json_response_multimodal(
        self,
        system_prompt: str,
        user_text: str,
        image_urls: list[str],
        task: str = "json",
    ) -> dict[str, Any]:
        """OpenAI-style multimodal JSON response. Sends text + image parts in
        a single user message so the LLM can ground the answer in the actual
        image bytes (not just text caption). Supported by Gemini 2.5 Flash and
        GPT-4o via the OpenAI-compat schema."""
        started = time.perf_counter()
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for url in image_urls:
            if not url:
                continue
            content.append({"type": "image_url", "image_url": {"url": url}})
        kwargs: dict[str, Any] = dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=settings.AGENT_MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        if self.base_url and "11434" in self.base_url:
            kwargs["extra_body"] = {"options": {"num_ctx": settings.LLM_OLLAMA_NUM_CTX}}
        response = await self._create(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000
        raw = response.choices[0].message.content or "{}"
        record_llm_call(
            task=task, model=self.model, latency_ms=latency_ms,
            in_text=system_prompt + user_text, out_text=raw,
            usage=getattr(response, "usage", None),
        )
        from src.agentrag.common.thinking import clean_thinking_content
        cleaned = clean_thinking_content(raw)
        content_str = cleaned if cleaned.strip() else raw
        if content_str and not content_str.lstrip().startswith("{"):
            idx = content_str.find("{")
            if idx >= 0:
                content_str = content_str[idx:]
        try:
            result = json.loads(content_str or "{}")
        except json.JSONDecodeError as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "json_response_multimodal parse failed (model=%s): %s | raw[:300]=%r",
                self.model, e, raw[:300],
            )
            result = {}
        if isinstance(result, list):
            result = result[0] if result and isinstance(result[0], dict) else {}
        elif not isinstance(result, dict):
            result = {}
        return result

    async def json_response(
        self,
        system_prompt: str,
        user_prompt: str,
        task: str = "json",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        kwargs: dict[str, Any] = dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=settings.AGENT_MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        # Ollama-only: force a larger context window. Gemini/OpenAI ignore or
        # reject extra options, so only pass when targeting an Ollama base URL.
        if self.base_url and "11434" in self.base_url:
            kwargs["extra_body"] = {"options": {"num_ctx": settings.LLM_OLLAMA_NUM_CTX}}
        response = await self._create(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000
        raw = response.choices[0].message.content or "{}"
        record_llm_call(
            task=task, model=self.model, latency_ms=latency_ms,
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
        task: str = "text",
    ) -> str:
        """Plain text response — no JSON enforcement. Used by transformations."""
        started = time.perf_counter()
        kwargs: dict[str, Any] = dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=settings.AGENT_MAX_OUTPUT_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if self.base_url and "11434" in self.base_url:
            kwargs["extra_body"] = {"options": {"num_ctx": settings.LLM_OLLAMA_NUM_CTX}}
        response = await self._create(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000
        raw = response.choices[0].message.content or ""
        record_llm_call(
            task=task, model=self.model, latency_ms=latency_ms,
            in_text=system_prompt + user_prompt, out_text=raw,
            usage=getattr(response, "usage", None),
        )
        from src.agentrag.common.thinking import clean_thinking_content
        return clean_thinking_content(raw)

    async def stream_text(
        self,
        system_prompt: str,
        user_prompt: str,
        task: str = "stream",
    ) -> AsyncIterator[str]:
        """Stream raw text tokens từ LLM (không ép JSON)."""
        started = time.perf_counter()
        kwargs: dict[str, Any] = dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=settings.AGENT_MAX_OUTPUT_TOKENS,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if self.base_url and "11434" in self.base_url:
            kwargs["extra_body"] = {"options": {"num_ctx": settings.LLM_OLLAMA_NUM_CTX}}
        stream = await self._create(**kwargs)
        buf: list[str] = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                buf.append(delta)
                yield delta
        latency_ms = (time.perf_counter() - started) * 1000
        record_llm_call(
            task=task, model=self.model, latency_ms=latency_ms,
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
