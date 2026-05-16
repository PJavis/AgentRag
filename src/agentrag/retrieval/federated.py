"""Federated retriever — filter-only wrapper over ElasticsearchRetriever (S4).

This is Execution Plane: no decision logic. The Reasoning Plane is
responsible for invoking `DomainRouter` and translating its output into
`system_override` / `specialty_override` kwargs.

Resolution order:
  1. Explicit override (`system_override` / `specialty_override`) →
     forwarded as filter clauses.
  2. No override → no domain filter applied. Reasoning Plane is expected
     to call `DomainRouter.classify` upstream and pass the picks here.
  3. `DOMAIN_FILTER_ENABLED=false` → all filters dropped, base behavior.

S5 backward-compat: pass a `router=DomainRouter()` to opt back into the
old auto-routing path (used by tests + legacy entry points).
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
        # S4: router is now opt-in (None = no auto routing).
        self._base = base or ElasticsearchRetriever()
        self._router = router

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
        if not filters and self._router is not None:
            # Legacy auto-routing path — only when an explicit router was
            # injected (S5 callers). S4 reasoning code calls DomainRouter
            # before passing overrides here, so this branch stays cold.
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
