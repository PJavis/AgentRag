from __future__ import annotations

import time
from typing import Any

from src.pam.agent.llm import AgentLLM


class LLMGateway:
    """
    LLM routing/cost-tracking placeholder.

    v1 keeps a single runtime model (configured in AgentLLM) and returns call latency
    so we can add model routing + cost accounting later without changing callers.
    """

    def __init__(self):
        self._client = AgentLLM()

    async def json_response(
        self,
        system_prompt: str,
        user_prompt: str,
        task: str = "general",
    ) -> tuple[dict[str, Any], float]:
        _ = task
        started = time.perf_counter()
        payload = await self._client.json_response(system_prompt, user_prompt)
        latency_ms = (time.perf_counter() - started) * 1000
        return payload, latency_ms
