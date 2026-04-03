from __future__ import annotations

"""
MCP Server — expose PAM service layer như MCP tool provider.

Tools:
  - "search":            wraps KnowledgeService.bootstrap_search
  - "structured_query":  wraps StructuredReasoningPipeline.run

SecurityService.filter_tool_results áp dụng cho tất cả tool responses.

Usage (future):
    from src.pam.mcp.server import PAMMCPServer
    server = PAMMCPServer()
    await server.handle_tool_call("search", {"query": "...", "document_title": "..."})
"""

from typing import Any

from src.pam.services.knowledge_service import KnowledgeService
from src.pam.services.llm_gateway import LLMGateway
from src.pam.services.security_service import SecurityService
from src.pam.structured.pipeline import StructuredReasoningPipeline
from src.pam.structured.query_classifier import QueryIntentClassifier


TOOL_DEFINITIONS = [
    {
        "name": "search",
        "description": "Search PAM knowledge base using hybrid retrieval (BM25 + dense + graph).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "document_title": {"type": "string", "description": "Optional: scope to a specific document"},
                "top_k": {"type": "integer", "description": "Number of results to return (default: 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "structured_query",
        "description": (
            "Answer structured questions (comparison, aggregation, ranking) using SQL reasoning. "
            "Returns an answer with SQL transparency and citations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The structured question to answer"},
                "document_title": {"type": "string", "description": "Optional: scope to a specific document"},
                "query_type": {
                    "type": "string",
                    "enum": ["comparison", "aggregation", "ranking", "multi_filter", "multi_hop"],
                    "description": "Type of structured query",
                },
            },
            "required": ["question"],
        },
    },
]


class PAMMCPServer:
    """
    Thin MCP adapter layer over PAM service layer.
    Designed to be wrapped by FastMCP or raw MCP SDK in future.
    """

    def __init__(self) -> None:
        self._llm_gateway = LLMGateway()
        self._knowledge = KnowledgeService()
        self._security = SecurityService()
        self._classifier = QueryIntentClassifier(llm_gateway=self._llm_gateway)
        self._structured_pipeline = StructuredReasoningPipeline(
            knowledge_service=self._knowledge,
            llm_gateway=self._llm_gateway,
            security_service=self._security,
        )

    def list_tools(self) -> list[dict[str, Any]]:
        return TOOL_DEFINITIONS

    async def handle_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> dict[str, Any]:
        if tool_name == "search":
            return await self._handle_search(tool_input)
        if tool_name == "structured_query":
            return await self._handle_structured_query(tool_input)
        return {"error": f"Unknown tool: {tool_name}"}

    async def _handle_search(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        query = str(tool_input.get("query", ""))
        document_title = tool_input.get("document_title")
        top_k = tool_input.get("top_k")

        _, tool_output = await self._knowledge.bootstrap_search(
            query=query,
            document_title=document_title,
            top_k=top_k,
        )
        filtered = self._security.filter_tool_results(tool_output, document_title)
        results = filtered.get("results") or []

        return {
            "tool": "search",
            "query": query,
            "results": [
                {
                    "content": r.get("content", ""),
                    "document_title": r.get("document_title"),
                    "section_path": r.get("section_path"),
                    "content_hash": r.get("content_hash"),
                    "score": r.get("score") or r.get("rrf_score"),
                }
                for r in results
            ],
        }

    async def _handle_structured_query(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        question = str(tool_input.get("question", ""))
        document_title = tool_input.get("document_title")
        query_type = tool_input.get("query_type", "comparison")

        result = await self._structured_pipeline.run(
            question=question,
            document_title=document_title,
            chat_history=None,
            query_type=query_type,
            classifier_confidence=0.95,
        )

        if result.get("_structured_fallback"):
            return {
                "tool": "structured_query",
                "error": "structured_reasoning_failed",
                "fallback_reason": result.get("_fallback_reason"),
            }

        return {
            "tool": "structured_query",
            "question": question,
            "answer": result.get("answer", ""),
            "sql_query": result.get("sql_query"),
            "citations": result.get("citations") or [],
            "reasoning_path": result.get("reasoning_path", "structured"),
        }
