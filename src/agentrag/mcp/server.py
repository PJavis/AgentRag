from __future__ import annotations

"""
MCP Server — expose AgentRag service layer như MCP tool provider.

Tools:
  - "search":            wraps KnowledgeService.bootstrap_search

SecurityService.filter_tool_results áp dụng cho tất cả tool responses.

Usage:
    from src.agentrag.mcp.server import MCPServer
    server = MCPServer()
    await server.handle_tool_call("search", {"query": "...", "document_title": "..."})
"""

from typing import Any

from src.agentrag.services.knowledge_service import KnowledgeService
from src.agentrag.services.llm_gateway import LLMGateway
from src.agentrag.services.security_service import SecurityService


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
]


class MCPServer:
    """Thin MCP adapter layer over AgentRag service layer."""

    def __init__(self) -> None:
        self._llm_gateway = LLMGateway()
        self._knowledge = KnowledgeService()
        self._security = SecurityService()

    def list_tools(self) -> list[dict[str, Any]]:
        return TOOL_DEFINITIONS

    async def handle_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> dict[str, Any]:
        if tool_name == "search":
            return await self._handle_search(tool_input)
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
