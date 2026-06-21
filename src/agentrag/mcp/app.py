from __future__ import annotations

"""
FastMCP app — expose AgentRag tools via MCP protocol.

Mounted at /mcp in the FastAPI app (streamable HTTP transport).
MCP clients (e.g. Claude Desktop) connect to: http://host/mcp/

Tools:
  - search:            hybrid retrieval (BM25 + dense + graph)
"""

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.agentrag.services.knowledge_service import KnowledgeService
from src.agentrag.services.llm_gateway import LLMGateway
from src.agentrag.services.security_service import SecurityService


mcp = FastMCP("AgentRag")

# Lazy-initialized singletons — avoids DB/ES connections at import time
_svc: dict[str, Any] = {}


def _services() -> tuple[KnowledgeService, SecurityService]:
    if not _svc:
        knowledge = KnowledgeService()
        security = SecurityService()
        _svc["knowledge"] = knowledge
        _svc["security"] = security
    return _svc["knowledge"], _svc["security"]


@mcp.tool()
async def search(
    query: str,
    document_title: str | None = None,
    top_k: int = 5,
) -> str:
    """Search the AgentRag knowledge base using hybrid retrieval (BM25 + dense + graph)."""
    knowledge, security = _services()
    _, tool_output = await knowledge.bootstrap_search(
        query=query,
        document_title=document_title,
        top_k=top_k,
    )
    filtered = security.filter_tool_results(tool_output, document_title)
    results = filtered.get("results") or []
    return json.dumps({
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
    }, ensure_ascii=False)
