"""ReasoningKnowledge — pure decision-side helpers (S4).

These functions live in the Reasoning Plane: they decide WHAT query to
run and WHICH tool to invoke. No IO, no LLM calls, no retrieval.

Concrete IO lives in `services/retrieval_service.py` and `agent/tools.py`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.agentrag.config import settings

if TYPE_CHECKING:
    from src.agentrag.structured.query_classifier import ClassifierOutput


_INTENT_TO_MODE: dict[str, str] = {
    "multi_hop": "hybrid_kg",
    "comparison": "hybrid",
    "aggregation": "hybrid",
    "ranking": "hybrid",
    "multi_filter": "hybrid",
}

_INTENT_TO_EXPANSION: dict[str, str] = {
    "aggregation": "count total sum tổng số lượng",
    "comparison": "compare difference versus khác nhau so sánh",
    "ranking": "top best highest ranking xếp hạng tốt nhất",
    "multi_filter": "list all filter điều kiện",
    "multi_hop": "relationship connection chain quan hệ liên kết",
}

_MODE_TO_TOOL: dict[str, str] = {
    "hybrid_kg": "search_hybrid_kg",
    "hybrid": "search_hybrid",
    "dense": "search_dense",
    "sparse": "search_sparse",
}


def select_retrieval_mode(intent: "ClassifierOutput | None") -> str:
    """Map classified intent → retrieval mode."""
    if intent is None or intent.intent == "semantic":
        return "hybrid_kg"
    return _INTENT_TO_MODE.get(intent.query_type or "", "hybrid_kg")


def expand_query(query: str, intent: "ClassifierOutput | None") -> str:
    """Append rule-based keyword expansion (no LLM call)."""
    if intent is None or intent.intent == "semantic":
        return query
    suffix = _INTENT_TO_EXPANSION.get(intent.query_type or "")
    return f"{query} {suffix}" if suffix else query


def mode_to_tool(mode: str) -> str:
    """Map retrieval mode → AgentTools tool name."""
    return _MODE_TO_TOOL.get(mode, "search_hybrid_kg")


def normalize_tool_call(
    tool_name: str,
    tool_input: dict[str, Any],
    question: str,
    document_title: str | None,
    valid_tools: set[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Fix up an LLM-emitted tool call:

    - Default `document_title` from caller scope when missing.
    - If `tool_name` isn't in `valid_tools`, fall back to `search_hybrid_kg`
      using the raw question + AGENT_TOOL_TOP_K.
    """
    chosen_name = tool_name
    chosen_input = dict(tool_input or {})
    if document_title and "document_title" not in chosen_input:
        chosen_input["document_title"] = document_title
    if valid_tools is not None and chosen_name not in valid_tools:
        chosen_name = "search_hybrid_kg"
        chosen_input = {
            "query": question,
            "top_k": settings.AGENT_TOOL_TOP_K,
            "document_title": document_title,
        }
    return chosen_name, chosen_input
