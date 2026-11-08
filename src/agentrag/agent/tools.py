from __future__ import annotations

from typing import Any, Callable, Awaitable

from sqlalchemy import select

from src.agentrag.config import settings
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document, Segment
from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore
from src.agentrag.retrieval.elasticsearch_retriever import ElasticsearchRetriever


ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def _scope() -> list[str] | None:
    """The turn's document ids (None = unscoped)."""
    from src.agentrag.retrieval.context import get_document_scope

    return get_document_scope()


def _scoped(query):
    """Restrict a Document query to the turn's documents, by id. The planner
    names documents by title, and titles collide across users, so a title
    check let a scoped turn read another notebook's same-named document."""
    import uuid as _uuid

    scope = _scope()
    if scope is None:
        return query
    ids = []
    for d in scope:
        try:
            ids.append(_uuid.UUID(str(d)))
        except (TypeError, ValueError):
            continue
    return query.where(Document.id.in_(ids))


class AgentTools:
    def __init__(self):
        self.retriever = ElasticsearchRetriever()
        self.es_store = ElasticsearchStore()
        self.planner_tools = (
            "search_sparse",
            "search_dense",
            "search_hybrid",
            "search_hybrid_kg",
            "get_document_segments",
            "get_chunk_by_hash",
        )
        self.tools: dict[str, ToolHandler] = {
            "search_sparse": self.search_sparse,
            "search_dense": self.search_dense,
            "search_hybrid": self.search_hybrid,
            "search_hybrid_kg": self.search_hybrid_kg,
            "compare_retrieval_modes": self.compare_retrieval_modes,
            "list_documents": self.list_documents,
            "get_document_graph_status": self.get_document_graph_status,
            "get_document_segments": self.get_document_segments,
            "get_chunk_by_hash": self.get_chunk_by_hash,
        }

    def has_tool(self, tool_name: str | None) -> bool:
        return bool(tool_name and tool_name in self.tools)

    def describe(self) -> list[dict[str, Any]]:
        all_tools = [
            {
                "name": "search_sparse",
                "description": "BM25 lexical search over indexed chunks",
                "input_schema": {"query": "str", "document_title": "str|null", "top_k": "int|null"},
            },
            {
                "name": "search_dense",
                "description": "Dense vector search over indexed chunks",
                "input_schema": {"query": "str", "document_title": "str|null", "top_k": "int|null"},
            },
            {
                "name": "search_hybrid",
                "description": "Hybrid retrieval using BM25 + dense + RRF",
                "input_schema": {"query": "str", "document_title": "str|null", "top_k": "int|null"},
            },
            {
                "name": "search_hybrid_kg",
                "description": "Hybrid retrieval over chunks + StructMem knowledge entries with RRF",
                "input_schema": {"query": "str", "document_title": "str|null", "top_k": "int|null"},
            },
            {
                "name": "compare_retrieval_modes",
                "description": "Compare sparse, dense, hybrid, and hybrid_kg results for one query",
                "input_schema": {"query": "str", "document_title": "str|null", "top_k": "int|null"},
            },
            {
                "name": "list_documents",
                "description": "List ingested documents",
                "input_schema": {"limit": "int|null"},
            },
            {
                "name": "get_document_graph_status",
                "description": "Get async StructMem ingest status by document title",
                "input_schema": {"document_title": "str"},
            },
            {
                "name": "get_document_segments",
                "description": "Fetch stored segments by document title",
                "input_schema": {"document_title": "str", "limit": "int|null"},
            },
            {
                "name": "get_chunk_by_hash",
                "description": "Fetch one or more chunks by content hash",
                "input_schema": {"content_hashes": "list[str]"},
            },
        ]
        return [tool for tool in all_tools if tool["name"] in self.planner_tools]

    async def call(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        return await self.tools[tool_name](tool_input)

    def _current_filters(self) -> dict[str, Any] | None:
        """Read S5 domain_filter + document scope from per-turn ContextVars."""
        from src.agentrag.retrieval.context import current_retrieval_filters

        return current_retrieval_filters()

    async def search_sparse(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        return await self.retriever.search(
            query=tool_input["query"],
            mode="sparse",
            top_k=tool_input.get("top_k") or settings.AGENT_TOOL_TOP_K,
            document_title=tool_input.get("document_title"),
            dense_query=tool_input.get("dense_query"),
            filters=self._current_filters(),
        )

    async def search_dense(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        return await self.retriever.search(
            query=tool_input["query"],
            mode="dense",
            top_k=tool_input.get("top_k") or settings.AGENT_TOOL_TOP_K,
            document_title=tool_input.get("document_title"),
            dense_query=tool_input.get("dense_query"),
            filters=self._current_filters(),
        )

    async def search_hybrid(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        return await self.retriever.search(
            query=tool_input["query"],
            mode="hybrid",
            top_k=tool_input.get("top_k") or settings.AGENT_TOOL_TOP_K,
            document_title=tool_input.get("document_title"),
            dense_query=tool_input.get("dense_query"),
            filters=self._current_filters(),
        )

    async def search_hybrid_kg(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        return await self.retriever.search(
            query=tool_input["query"],
            mode="hybrid_kg",
            top_k=tool_input.get("top_k") or settings.AGENT_TOOL_TOP_K,
            document_title=tool_input.get("document_title"),
            dense_query=tool_input.get("dense_query"),
            filters=self._current_filters(),
        )

    async def compare_retrieval_modes(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        query = tool_input["query"]
        document_title = tool_input.get("document_title")
        top_k = tool_input.get("top_k") or 3
        return {
            "query": query,
            "document_title": document_title,
            **{
                mode: await self.retriever.search(
                    query, mode, top_k, document_title, filters=self._current_filters()
                )
                for mode in ("sparse", "dense", "hybrid", "hybrid_kg")
            },
        }

    async def list_documents(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        limit = tool_input.get("limit") or 20
        # A scoped turn must not even learn the names of other documents.
        query = _scoped(select(Document))
        async with AsyncSessionLocal() as session:
            result = await session.execute(query.limit(limit))
            documents = result.scalars().all()
        return {
            "documents": [
                {
                    "document_id": str(doc.id),
                    "title": doc.title,
                    "source_id": doc.source_id,
                    "graph_status": doc.graph_status,
                }
                for doc in documents
            ]
        }

    async def get_document_graph_status(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        document_title = tool_input["document_title"]
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                _scoped(select(Document).where(Document.title == document_title))
            )
            documents = result.scalars().all()
        return {
            "documents": [
                {
                    "document_id": str(doc.id),
                    "title": doc.title,
                    "graph_status": doc.graph_status,
                    "graph_total_chunks": doc.graph_total_chunks,
                    "graph_processed_chunks": doc.graph_processed_chunks,
                    "graph_failed_chunks": doc.graph_failed_chunks,
                    "graph_last_error": doc.graph_last_error,
                }
                for doc in documents
            ]
        }

    async def get_document_segments(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        document_title = tool_input["document_title"]
        limit = tool_input.get("limit") or settings.AGENT_TOOL_TOP_K
        # The planner picks the title; a title outside the turn's scope (from
        # history, or hallucinated) must not read another notebook's segments.
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                _scoped(
                    select(Segment, Document)
                    .join(Document, Segment.document_id == Document.id)
                    .where(Document.title == document_title)
                )
                .limit(limit)
            )
            rows = result.all()
        return {
            "segments": [
                {
                    "document_title": document.title,
                    "section_path": segment.section_path,
                    "position": segment.position,
                    "content_hash": segment.content_hash,
                    "content": segment.content,
                }
                for segment, document in rows
            ]
        }

    async def get_chunk_by_hash(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        hashes = tool_input.get("content_hashes") or []
        return {
            "results": await self.es_store.get_chunks_by_hashes(
                hashes, filters=self._current_filters()
            )
        }
