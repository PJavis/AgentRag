from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI

from src.agentrag.agent.llm import AgentLLM
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


def _relabel_last(model: str, generic_task: str, target_task: str) -> None:
    """AgentLLM records calls with generic 'json'/'text'/'stream' task labels.
    LLMGateway knows the real semantic task ('answer','decide',...). After
    each call, rewrite the last matching ledger entry so dashboards roll up
    per-task instead of per-method."""
    from src.agentrag.observability.cost import _LEDGER, _LOCK
    with _LOCK:
        for entry in reversed(_LEDGER):
            if entry["model"] == model and entry["task"] == generic_task:
                entry["task"] = target_task
                return


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
        """AgentLLM records the call internally; we just override the task label."""
        client = self._resolve_client(task)
        started = time.perf_counter()
        payload = await client.json_response(system_prompt, user_prompt)
        latency_ms = (time.perf_counter() - started) * 1000
        # Relabel last entry from generic "json" → the task tag we know.
        _relabel_last(client.model, "json", task)
        return payload, latency_ms

    async def text_response(
        self,
        system_prompt: str,
        user_prompt: str,
        task: str = "general",
    ) -> str:
        """Plain-text completion (no JSON enforcement)."""
        client = self._resolve_client(task)
        text = await client.text_response(system_prompt, user_prompt)
        _relabel_last(client.model, "text", task)
        return text

    def cost_summary(self) -> dict[str, Any]:
        return _cost.cost_summary()

    # ── Routing helpers ───────────────────────────────────────────────────────

    def _resolve_client(self, task: str) -> AgentLLM:
        """
        Nếu LLM_ROUTING_ENABLED và task có trong LLM_TASK_MODEL_MAP
        → trả cached AgentLLM với override model.
        Else → trả default client.
        """
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
        response = await client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=messages,
        )
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
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=base_url, timeout=timeout)
        elif provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY required for vision")
            client = AsyncOpenAI(
                api_key=settings.GEMINI_API_KEY,
                base_url=base_url or "https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=timeout,
            )
        elif provider == "ollama":
            client = AsyncOpenAI(
                api_key=settings.OLLAMA_API_KEY,
                base_url=base_url or settings.OLLAMA_BASE_URL,
                timeout=timeout,
            )
        else:
            raise RuntimeError(f"Unsupported vision provider: {provider}")

        self._vision_client = client
        self._vision_model = model
        return client, model

