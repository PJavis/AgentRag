from __future__ import annotations

"""
FastMCP app — expose AgentRag tools via MCP protocol.

Mounted at /mcp in the FastAPI app (streamable HTTP transport).
MCP clients (e.g. Claude Desktop) connect to: http://host/mcp/

Tools:
  - search:            hybrid retrieval (BM25 + dense + graph)
  - structured_query:  SQL-driven reasoning for comparison/aggregation/ranking
"""

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.agentrag.services.knowledge_service import KnowledgeService
from src.agentrag.services.llm_gateway import LLMGateway
from src.agentrag.services.security_service import SecurityService
from src.agentrag.structured.pipeline import StructuredReasoningPipeline
from src.agentrag.structured.query_classifier import QueryIntentClassifier


mcp = FastMCP("AgentRag")

# Lazy-initialized singletons — avoids DB/ES connections at import time
_svc: dict[str, Any] = {}


def _services() -> tuple[KnowledgeService, SecurityService, StructuredReasoningPipeline]:
    if not _svc:
        gateway = LLMGateway()
        knowledge = KnowledgeService()
        security = SecurityService()
        classifier = QueryIntentClassifier(llm_gateway=gateway)
        pipeline = StructuredReasoningPipeline(
            knowledge_service=knowledge,
            llm_gateway=gateway,
            security_service=security,
        )
        _svc["knowledge"] = knowledge
        _svc["security"] = security
        _svc["pipeline"] = pipeline
    return _svc["knowledge"], _svc["security"], _svc["pipeline"]


@mcp.tool()
async def search(
    query: str,
    document_title: str | None = None,
    top_k: int = 5,
) -> str:
    """Search the AgentRag knowledge base using hybrid retrieval (BM25 + dense + graph)."""
    knowledge, security, _ = _services()
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


@mcp.tool()
async def structured_query(
    question: str,
    document_title: str | None = None,
    query_type: str = "comparison",
) -> str:
    """Answer structured questions (comparison, aggregation, ranking) using SQL reasoning."""
    _, _, pipeline = _services()
    result = await pipeline.run(
        question=question,
        document_title=document_title,
        chat_history=None,
        query_type=query_type,
        classifier_confidence=0.95,
    )
    if result.get("_structured_fallback"):
        return json.dumps({
            "error": "structured_reasoning_failed",
            "reason": result.get("_fallback_reason"),
        })
    return json.dumps({
        "answer": result.get("answer", ""),
        "sql_query": result.get("sql_query"),
        "citations": result.get("citations") or [],
        "reasoning_path": result.get("reasoning_path", "structured"),
    }, ensure_ascii=False)
