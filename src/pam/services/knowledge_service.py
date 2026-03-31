from __future__ import annotations

import json
from typing import Any

from src.pam.agent.tools import AgentTools
from src.pam.config import settings


class KnowledgeService:
    """
    Retrieval + tool-execution facade.

    This centralizes tool access so Supervisor Agent can stay orchestration-only.
    """

    def __init__(self):
        self._tools = AgentTools()

    def describe_tools(self) -> list[dict[str, Any]]:
        return self._tools.describe()

    def has_tool(self, tool_name: str | None) -> bool:
        return self._tools.has_tool(tool_name)

    @staticmethod
    def fingerprint_call(tool_name: str, tool_input: dict[str, Any]) -> str:
        return json.dumps(
            {"tool_name": tool_name, "tool_input": tool_input},
            ensure_ascii=True,
            sort_keys=True,
        )

    async def bootstrap_search(
        self,
        query: str,
        document_title: str | None,
        top_k: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        tool_input = {
            "query": query,
            "top_k": top_k or settings.AGENT_TOOL_TOP_K,
            "document_title": document_title,
        }
        output = await self._tools.call("search_hybrid_kg", tool_input)
        return tool_input, output

    async def execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        question: str,
        document_title: str | None,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        chosen_name, chosen_input = self.normalize_tool_call(
            tool_name=tool_name,
            tool_input=tool_input,
            question=question,
            document_title=document_title,
        )
        output = await self._tools.call(chosen_name, chosen_input)
        return chosen_name, chosen_input, output

    def normalize_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        question: str,
        document_title: str | None,
    ) -> tuple[str, dict[str, Any]]:
        chosen_name = tool_name
        chosen_input = dict(tool_input or {})

        if document_title and "document_title" not in chosen_input:
            chosen_input["document_title"] = document_title

        if not self._tools.has_tool(chosen_name):
            chosen_name = "search_hybrid_kg"
            chosen_input = {
                "query": question,
                "top_k": settings.AGENT_TOOL_TOP_K,
                "document_title": document_title,
            }
        return chosen_name, chosen_input
