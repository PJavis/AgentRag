from __future__ import annotations

import base64
import json
import time
from typing import Any

from openai import AsyncOpenAI

from src.agentrag.agent.llm import AgentLLM
from src.agentrag.config import settings


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
        self._cost_log: list[dict[str, Any]] = []        # in-memory, Phase C v1
        self._vision_client: AsyncOpenAI | None = None
        self._vision_model: str | None = None

    async def json_response(
        self,
        system_prompt: str,
        user_prompt: str,
        task: str = "general",
    ) -> tuple[dict[str, Any], float]:
        client = self._resolve_client(task)
        started = time.perf_counter()
        payload = await client.json_response(system_prompt, user_prompt)
        latency_ms = (time.perf_counter() - started) * 1000
        if settings.LLM_COST_TRACKING_ENABLED:
            self._record_cost(task, latency_ms)
        return payload, latency_ms

    async def text_response(
        self,
        system_prompt: str,
        user_prompt: str,
        task: str = "general",
    ) -> str:
        """Plain-text completion (no JSON enforcement)."""
        client = self._resolve_client(task)
        started = time.perf_counter()
        text = await client.text_response(system_prompt, user_prompt)
        latency_ms = (time.perf_counter() - started) * 1000
        if settings.LLM_COST_TRACKING_ENABLED:
            self._record_cost(task, latency_ms)
        return text

    def cost_summary(self) -> dict[str, Any]:
        """Aggregate cost_log theo task. Expose qua GET /metrics."""
        summary: dict[str, dict[str, Any]] = {}
        for entry in self._cost_log:
            t = entry["task"]
            if t not in summary:
                summary[t] = {"calls": 0, "total_latency_ms": 0.0, "avg_latency_ms": 0.0}
            summary[t]["calls"] += 1
            summary[t]["total_latency_ms"] += entry["latency_ms"]
        for t, stats in summary.items():
            if stats["calls"] > 0:
                stats["avg_latency_ms"] = round(stats["total_latency_ms"] / stats["calls"], 2)
            stats["total_latency_ms"] = round(stats["total_latency_ms"], 2)
        return {"tasks": summary, "total_calls": len(self._cost_log)}

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
        if settings.LLM_COST_TRACKING_ENABLED:
            self._record_cost(task, latency_ms)
        return text, latency_ms

    def _get_vision_client(self) -> tuple[AsyncOpenAI, str]:
        """Lazily resolve and cache the vision LLM client."""
        if self._vision_client is not None and self._vision_model is not None:
            return self._vision_client, self._vision_model

        provider = settings.VISION_PROVIDER or settings.EXTRACTION_PROVIDER
        model = settings.VISION_MODEL or settings.EXTRACTION_MODEL
        base_url = settings.VISION_BASE_URL

        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY required for vision")
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=base_url)
        elif provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY required for vision")
            client = AsyncOpenAI(
                api_key=settings.GEMINI_API_KEY,
                base_url=base_url or "https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        elif provider == "ollama":
            client = AsyncOpenAI(
                api_key=settings.OLLAMA_API_KEY,
                base_url=base_url or settings.OLLAMA_BASE_URL,
            )
        else:
            raise RuntimeError(f"Unsupported vision provider: {provider}")

        self._vision_client = client
        self._vision_model = model
        return client, model

    def _record_cost(self, task: str, latency_ms: float) -> None:
        self._cost_log.append({
            "task": task,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time(),
        })
