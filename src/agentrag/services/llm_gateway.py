from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI

from src.agentrag.agent.llm import AgentLLM
from src.agentrag.common.langfuse_client import make_async_openai
from src.agentrag.config import settings
from src.agentrag.observability import cost as _cost

log = logging.getLogger(__name__)

# Gemini API pricing as of 2026-Q2, USD per 1M tokens (input / output).
# Adjust when pricing changes. Unknown models default to flash pricing.
_PRICE_PER_1M = {
    "gemini-2.5-pro":          (1.25, 10.00),
    "gemini-2.5-flash":        (0.075, 0.30),
    "gemini-2.5-flash-lite":   (0.05, 0.20),
    "gemini-2.0-flash":        (0.10, 0.40),
    "gpt-4o":                  (2.50, 10.00),
    "gpt-4o-mini":             (0.15, 0.60),
}


def _estimate_tokens(text: str) -> int:
    """Rough char→token estimate. ~4 chars/token for English, ~2 for VN/CJK.
    Avoids importing tiktoken (which we already use elsewhere) into the hot path.
    """
    if not text:
        return 0
    # Bias toward CJK density: count non-ASCII chars as ~0.5 tokens each.
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ascii_chars = len(text) - non_ascii
    return int(ascii_chars / 4 + non_ascii * 0.55) + 1


def _price_for(model: str) -> tuple[float, float]:
    if not model:
        return _PRICE_PER_1M["gemini-2.5-flash"]
    if model in _PRICE_PER_1M:
        return _PRICE_PER_1M[model]
    # Prefix match for variants like gemini-2.5-pro-preview.
    for k, v in _PRICE_PER_1M.items():
        if model.startswith(k):
            return v
    return _PRICE_PER_1M["gemini-2.5-flash"]


class LLMGateway:
    """
    LLM routing + cost tracking (Phase C).

    - v1 (default): single runtime model, latency tracking only.
    - v2 (LLM_ROUTING_ENABLED=true): per-task model routing từ LLM_TASK_MODEL_MAP.
    - cost_summary(): aggregate stats có thể expose qua GET /metrics.
    """

    def __init__(self) -> None:
        self._default_client = AgentLLM()
        self._routed_clients: dict[str, AgentLLM] = {}   # task → AgentLLM instance
        self._vision_client: AsyncOpenAI | None = None
        self._vision_model: str | None = None

    async def json_response(
        self,
        system_prompt: str,
        user_prompt: str,
        task: str = "general",
    ) -> tuple[dict[str, Any], float]:
        client = self._resolve_client(task, content=system_prompt + user_prompt)
        started = time.perf_counter()
        payload = await client.json_response(system_prompt, user_prompt, task=task)
        latency_ms = (time.perf_counter() - started) * 1000
        return payload, latency_ms

    async def json_response_multimodal(
        self,
        system_prompt: str,
        user_text: str,
        image_urls: list[str],
        task: str = "general",
    ) -> tuple[dict[str, Any], float]:
        """Route a multimodal (text + images) JSON call through the per-task
        client. Use when packed_context has image segments and the chosen
        answer model supports vision (Gemini 2.5 Flash, GPT-4o, etc.)."""
        client = self._resolve_client(task, content=system_prompt + user_text)
        started = time.perf_counter()
        payload = await client.json_response_multimodal(
            system_prompt, user_text, image_urls, task=task
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return payload, latency_ms

    async def text_response(
        self,
        system_prompt: str,
        user_prompt: str,
        task: str = "general",
    ) -> str:
        client = self._resolve_client(task, content=system_prompt + user_prompt)
        return await client.text_response(system_prompt, user_prompt, task=task)

    def cost_summary(self) -> dict[str, Any]:
        return _cost.cost_summary()

    # ── Routing helpers ───────────────────────────────────────────────────────

    def _resolve_client(self, task: str, content: str = "") -> AgentLLM:
        """
        Resolution order:
          1. If LLM_LARGE_CONTEXT_MODEL set AND content tokens > threshold
             → large-context client (highest priority).
          2. If LLM_ROUTING_ENABLED AND task ∈ LLM_TASK_MODEL_MAP
             → task-specific client.
          3. Default client.
        """
        # 1. Large-context auto-route (open-notebook pattern).
        large_model = settings.LLM_LARGE_CONTEXT_MODEL
        if large_model and content:
            tokens = _estimate_tokens(content)
            if tokens > settings.LLM_LARGE_CONTEXT_THRESHOLD:
                if large_model not in self._routed_clients:
                    self._routed_clients[large_model] = AgentLLM(
                        model_override=large_model
                    )
                return self._routed_clients[large_model]

        # 2. Task-based routing.
        if not settings.LLM_ROUTING_ENABLED:
            return self._default_client

        task_map = self._load_task_map()
        override_model = task_map.get(task)
        if not override_model:
            return self._default_client

        if override_model not in self._routed_clients:
            self._routed_clients[override_model] = AgentLLM(model_override=override_model)
        return self._routed_clients[override_model]

    @staticmethod
    def _load_task_map() -> dict[str, str]:
        """Parse LLM_TASK_MODEL_MAP JSON string. Trả {} nếu invalid."""
        try:
            result = json.loads(settings.LLM_TASK_MODEL_MAP)
            if isinstance(result, dict):
                return {str(k): str(v) for k, v in result.items()}
        except (json.JSONDecodeError, TypeError):
            pass
        return {}

    async def vision_response_batch(
        self,
        system_prompt: str,
        text_prompt: str,
        images: list[tuple[bytes, str]],
        task: str = "vision",
    ) -> list[str]:
        """Describe N images in ONE multimodal LLM call. Returns list of N
        description strings. Saves RPM cost (free Gemini = 12 RPM).
        Caller responsible for fallback when parse fails."""
        if not images:
            return []
        import json as _json
        client, model = self._get_vision_client()
        content: list[dict[str, Any]] = [{"type": "text", "text": text_prompt}]
        for img_bytes, mime in images:
            data_url = f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        started = time.perf_counter()
        response = await client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=settings.AGENT_MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            messages=messages,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        raw = (response.choices[0].message.content or "").strip()
        total_img_bytes = sum(len(b) for b, _ in images)
        _cost.record_llm_call(
            task=task,
            model=model,
            latency_ms=latency_ms,
            in_text=system_prompt + text_prompt + ("X" * (total_img_bytes // 100)),
            out_text=raw,
            usage=getattr(response, "usage", None),
        )
        from src.agentrag.common.thinking import clean_thinking_content
        cleaned = clean_thinking_content(raw)
        s = cleaned if cleaned.strip() else raw
        if s and not s.lstrip().startswith("{"):
            idx = s.find("{")
            if idx >= 0:
                s = s[idx:]
        try:
            parsed = _json.loads(s or "{}")
        except _json.JSONDecodeError:
            return []
        descs = parsed.get("descriptions")
        if isinstance(descs, list):
            return [str(d) for d in descs]
        return []

    async def vision_response(
        self,
        system_prompt: str,
        text_prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        task: str = "vision",
    ) -> tuple[str, float]:
        """Call a vision-capable LLM with an image. Returns (description, latency_ms).

        Uses VISION_PROVIDER / VISION_MODEL settings. Falls back to EXTRACTION_*
        if VISION_PROVIDER is not set. Raises RuntimeError if no vision provider configured.
        """
        client, model = self._get_vision_client()
        data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        started = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=settings.AGENT_MAX_OUTPUT_TOKENS,
                messages=messages,
            )
        except Exception as exc:
            from src.agentrag.agent.llm import _is_model_missing
            fallback = (settings.LLM_FALLBACK_MODEL or "").strip()
            if _is_model_missing(exc) and fallback and fallback != model:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "vision_response: model %r not found, falling back to %r",
                    model, fallback,
                )
                response = await client.chat.completions.create(
                    model=fallback,
                    temperature=0.0,
                    max_tokens=settings.AGENT_MAX_OUTPUT_TOKENS,
                    messages=messages,
                )
                model = fallback
            else:
                raise
        latency_ms = (time.perf_counter() - started) * 1000
        text = (response.choices[0].message.content or "").strip()
        # image bytes also bill tokens; pad in_text proxy by image size.
        _cost.record_llm_call(
            task=task,
            model=model,
            latency_ms=latency_ms,
            in_text=system_prompt + text_prompt + ("X" * (len(image_bytes) // 100)),
            out_text=text,
            usage=getattr(response, "usage", None),
        )
        return text, latency_ms

    def _get_vision_client(self) -> tuple[AsyncOpenAI, str]:
        """Lazily resolve and cache the vision LLM client."""
        if self._vision_client is not None and self._vision_model is not None:
            return self._vision_client, self._vision_model

        provider = settings.VISION_PROVIDER or settings.EXTRACTION_PROVIDER
        model = settings.VISION_MODEL or settings.EXTRACTION_MODEL
        base_url = settings.VISION_BASE_URL
        # Vision models (esp. local llava cold-start) cần timeout dài
        timeout = float(settings.VISION_TIMEOUT_SECONDS)

        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY required for vision")
            client = make_async_openai(api_key=settings.OPENAI_API_KEY, base_url=base_url, timeout=timeout)
        elif provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY required for vision")
            client = make_async_openai(
                api_key=settings.GEMINI_API_KEY,
                base_url=base_url or "https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=timeout,
            )
        elif provider == "ollama":
            client = make_async_openai(
                api_key=settings.OLLAMA_API_KEY,
                base_url=base_url or settings.OLLAMA_BASE_URL,
                timeout=timeout,
            )
        else:
            raise RuntimeError(f"Unsupported vision provider: {provider}")

        self._vision_client = client
        self._vision_model = model
        return client, model

