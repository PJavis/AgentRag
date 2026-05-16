"""Wraps ElasticsearchRetriever with domain-aware filtering.

Resolution order:
  1. UI override (system_override / specialty_override) → use directly.
  2. Else consult DomainRouter (SLM) → pass picks as filter clauses.
  3. If DOMAIN_FILTER_ENABLED=false, behave identical to base retriever.

Returns the base payload + `domain_route` key when the router was consulted.
"""
from __future__ import annotations

from typing import Any

from src.agentrag.config import settings
from src.agentrag.orchestration.domain_router import DomainRoute, DomainRouter
from src.agentrag.retrieval.elasticsearch_retriever import ElasticsearchRetriever


class FederatedRetriever:
    def __init__(
        self,
        base: ElasticsearchRetriever | None = None,
        router: DomainRouter | None = None,
    ) -> None:
        self._base = base or ElasticsearchRetriever()
        self._router = router or DomainRouter()

    async def search(
        self,
        query: str,
        *,
        document_title: str | None = None,
        system_override: str | None = None,
        specialty_override: list[str] | None = None,
        top_k: int | None = None,
        mode: str = "hybrid_kg",
        rerank: bool | None = None,
        dense_query: str | None = None,
    ) -> dict[str, Any]:
        if not settings.DOMAIN_FILTER_ENABLED:
            return await self._base.search(
                query=query,
                document_title=document_title,
                top_k=top_k,
                mode=mode,
                rerank=rerank,
                dense_query=dense_query,
            )

        route: DomainRoute | None = None
        filters: dict[str, list[str]] = {}
        if system_override:
            filters["systems"] = [system_override]
        if specialty_override:
            filters["specialties"] = list(specialty_override)
        if not filters:
            route = await self._router.classify(query)
            if route.systems:
                filters["systems"] = route.systems
            if route.specialties:
                filters["specialties"] = route.specialties

        out = await self._base.search(
            query=query,
            document_title=document_title,
            top_k=top_k,
            mode=mode,
            rerank=rerank,
            dense_query=dense_query,
            filters=filters or None,
        )
        if route is not None:
            out["domain_route"] = {
                "systems": route.systems,
                "specialties": route.specialties,
                "confidence": route.confidence,
            }
        return out
