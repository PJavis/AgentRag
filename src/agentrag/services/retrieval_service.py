"""RetrievalService — Execution Plane facade for hybrid search (S4).

This is the ONLY way Reasoning Plane code should retrieve. It wraps
`FederatedRetriever` (which in turn wraps `ElasticsearchRetriever`).

S4 contract:
- No routing logic here. Reasoning code calls `DomainRouter.classify`
  separately and passes `system_override`/`specialty_override` to
  `.search()`.
- No prompt construction. This is pure IO + filters.
"""
from __future__ import annotations

from typing import Any

from src.agentrag.retrieval.elasticsearch_retriever import ElasticsearchRetriever
from src.agentrag.retrieval.federated import FederatedRetriever


class RetrievalService:
    def __init__(
        self,
        base: ElasticsearchRetriever | None = None,
    ) -> None:
        # No router injected — Reasoning Plane drives routing.
        self._fed = FederatedRetriever(base=base, router=None)

    async def search(
        self,
        query: str,
        *,
        document_title: str | None = None,
        top_k: int | None = None,
        mode: str = "hybrid_kg",
        rerank: bool | None = None,
        dense_query: str | None = None,
        filters: dict[str, Any] | None = None,
        system_override: str | None = None,
        specialty_override: list[str] | None = None,
    ) -> dict[str, Any]:
        # Caller can pass either a generic `filters` dict OR
        # system_override/specialty_override (S5 UI form). Merge:
        if filters:
            sys = filters.get("systems") or []
            specs = filters.get("specialties") or []
            if sys and not system_override:
                system_override = sys[0]
            if specs and not specialty_override:
                specialty_override = list(specs)
        return await self._fed.search(
            query=query,
            document_title=document_title,
            top_k=top_k,
            mode=mode,
            rerank=rerank,
            dense_query=dense_query,
            system_override=system_override,
            specialty_override=specialty_override,
        )
