from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.pam.agents.data_agent import DataResult
from src.pam.services.llm_gateway import LLMGateway


@dataclass
class InsightResult:
    question: str
    insight: str
    key_findings: list[str]
    citations: list[dict[str, Any]]


_INSIGHT_SYSTEM = """\
You are a business insight analyst. Given data results for a question, extract key insights.

Return JSON:
{
  "insight": "main insight paragraph",
  "key_findings": ["finding 1", "finding 2", ...]
}

Be specific, data-driven, and concise. Use the same language as the question.
"""


class InsightAgent:
    """
    Specialized worker: tạo business insights từ DataResult.
    """

    def __init__(self, llm_gateway: LLMGateway) -> None:
        self._llm = llm_gateway

    async def run(self, data: DataResult, context: str = "") -> InsightResult:
        # Build context từ source_chunks
        excerpts = [
            {"section": c.get("section_path", ""), "text": (c.get("content") or c.get("excerpt") or "")[:300]}
            for c in data.source_chunks[:5]
        ]

        user_prompt = json.dumps(
            {
                "question": data.question,
                "reasoning_path": data.reasoning_path,
                "sql_query": data.sql_query,
                "source_excerpts": excerpts,
                "additional_context": context,
            },
            ensure_ascii=False,
        )

        try:
            raw, _latency = await self._llm.json_response(
                system_prompt=_INSIGHT_SYSTEM,
                user_prompt=user_prompt,
                task="insight",
            )
            return InsightResult(
                question=data.question,
                insight=str(raw.get("insight", "")),
                key_findings=[str(f) for f in (raw.get("key_findings") or [])],
                citations=data.citations,
            )
        except Exception as exc:
            return InsightResult(
                question=data.question,
                insight=f"Could not generate insight: {exc}",
                key_findings=[],
                citations=data.citations,
            )
