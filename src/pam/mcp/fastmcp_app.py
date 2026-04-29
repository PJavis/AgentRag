from __future__ import annotations

"""
FastMCP integration for PAM.

Registers PAM tools with FastMCP and exposes an SSE Starlette app
that can be mounted to the main FastAPI application.

Claude Desktop config (~/.claude/claude_desktop_config.json):
  {
    "mcpServers": {
      "pam": {
        "url": "http://127.0.0.1:8000/mcp/sse"
      }
    }
  }
"""

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

_mcp = FastMCP(
    name="PAM",
    instructions=(
        "PAM is a knowledge-base assistant. "
        "Use 'search' for open-ended questions and 'structured_query' for "
        "comparisons, aggregations, and rankings."
    ),
)

# Lazy singleton — created once at first tool call so that heavy service
# constructors (DB pool, embedders) run inside the async event loop.
_pam_server: Any = None


def _get_server() -> Any:
    global _pam_server
    if _pam_server is None:
        from src.pam.mcp.server import PAMMCPServer
        _pam_server = PAMMCPServer()
    return _pam_server


@_mcp.tool(
    description=(
        "Search PAM knowledge base using hybrid retrieval (BM25 + dense + graph). "
        "Returns relevant passages with section paths and scores."
    )
)
async def search(
    query: str,
    document_title: str | None = None,
    top_k: int = 5,
) -> str:
    result = await _get_server().handle_tool_call(
        "search",
        {"query": query, "document_title": document_title, "top_k": top_k},
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@_mcp.tool(
    description=(
        "Answer structured questions (comparison, aggregation, ranking, multi_filter, multi_hop) "
        "using SQL reasoning over the knowledge base. Returns an answer with the SQL query used "
        "and source citations."
    )
)
async def structured_query(
    question: str,
    document_title: str | None = None,
    query_type: str = "comparison",
) -> str:
    result = await _get_server().handle_tool_call(
        "structured_query",
        {
            "question": question,
            "document_title": document_title,
            "query_type": query_type,
        },
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def get_sse_app():
    """Return the Starlette SSE app to mount at /mcp."""
    return _mcp.sse_app()
