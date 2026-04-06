from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.pam.agent.tools import AgentTools
from src.pam.config import settings

if TYPE_CHECKING:
    from src.pam.structured.query_classifier import ClassifierOutput


class KnowledgeService:
    """
    Retrieval + tool-execution facade.

    Phase B additions:
    - intent-aware retrieval mode selection
    - rule-based query expansion
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
        intent: ClassifierOutput | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        mode = self._select_retrieval_mode(intent)
        expanded_query = self.expand_query(query, intent)
        # Aggregation queries need wider context to avoid truncating enumeration tables
        effective_top_k = top_k or settings.AGENT_TOOL_TOP_K
        if intent is not None and intent.query_type == "aggregation":
            effective_top_k = max(effective_top_k, 15)
        tool_input = {
            "query": expanded_query,
            "mode": mode,
            "top_k": effective_top_k,
            "document_title": document_title,
        }
        # map mode → tool name
        tool_name = self._mode_to_tool(mode)
        output = await self._tools.call(tool_name, tool_input)
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

    # ── Intent-aware helpers ──────────────────────────────────────────────────

    def _select_retrieval_mode(self, intent: ClassifierOutput | None) -> str:
        """
        Chọn retrieval mode dựa trên intent:
        - None / semantic → "hybrid_kg" (default, giữ nguyên behavior cũ)
        - multi_hop       → "hybrid_kg" (graph giúp ích cho multi-hop)
        - các loại còn lại → "hybrid" (structured pipeline tự lý luận, không cần graph)
        """
        if intent is None or intent.intent == "semantic":
            return "hybrid_kg"
        mapping: dict[str, str] = {
            "multi_hop": "hybrid_kg",
            "comparison": "hybrid",
            "aggregation": "hybrid",
            "ranking": "hybrid",
            "multi_filter": "hybrid",
        }
        return mapping.get(intent.query_type or "", "hybrid_kg")

    def expand_query(self, query: str, intent: ClassifierOutput | None) -> str:
        """
        Rule-based keyword expansion — không cần LLM call.
        Giúp retrieval tìm được nhiều chunk liên quan hơn.
        """
        if intent is None or intent.intent == "semantic":
            return query
        expansions: dict[str, str] = {
            "aggregation": "count total sum tổng số lượng",
            "comparison": "compare difference versus khác nhau so sánh",
            "ranking": "top best highest ranking xếp hạng tốt nhất",
            "multi_filter": "list all filter điều kiện",
            "multi_hop": "relationship connection chain quan hệ liên kết",
        }
        suffix = expansions.get(intent.query_type or "")
        if suffix:
            return f"{query} {suffix}"
        return query

    @staticmethod
    def _mode_to_tool(mode: str) -> str:
        """Map retrieval mode string → AgentTools tool name."""
        return {
            "hybrid_kg": "search_hybrid_kg",
            "hybrid": "search_hybrid",
            "dense": "search_dense",
            "sparse": "search_sparse",
        }.get(mode, "search_hybrid_kg")
