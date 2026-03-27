from __future__ import annotations

from typing import Any

from sqlalchemy import select

from src.pam.config import settings
from src.pam.database import AsyncSessionLocal
from src.pam.database.models import Document
from src.pam.graph.graphiti_service import GraphitiService
from src.pam.ingestion.embedders.factory import build_embedding_provider
from src.pam.ingestion.stores.elasticsearch_store import ElasticsearchStore
from src.pam.retrieval.reranker import LLMReranker


class ElasticsearchRetriever:
    def __init__(self):
        self.store = ElasticsearchStore()
        self.embedder = build_embedding_provider(settings)
        self.reranker = LLMReranker()
        self._last_rerank_reason = "not_attempted"
        self.graph_service: GraphitiService | None = None

    async def search(
        self,
        query: str,
        mode: str = "hybrid_kg",
        top_k: int | None = None,
        document_title: str | None = None,
        rerank: bool | None = None,
    ) -> dict:
        if mode not in {"sparse", "dense", "hybrid", "hybrid_kg"}:
            raise ValueError("mode must be one of: sparse, dense, hybrid, hybrid_kg")

        size = top_k or settings.RETRIEVAL_TOP_K
        should_rerank = settings.RETRIEVAL_RERANK_ENABLED if rerank is None else rerank
        candidate_size = self.reranker.candidate_size(size, force=should_rerank)

        if mode == "sparse":
            hits = await self.store.sparse_search(
                query=query,
                top_k=candidate_size,
                document_title=document_title,
            )
            hits = self._dedupe_hits(hits)
            hits, reranked = await self._rerank_hits(
                query=query,
                hits=hits,
                top_k=size,
                should_rerank=should_rerank,
            )
            hits = self._apply_query_intent_ranking(query=query, hits=hits)
            hits = self._finalize_ranks(hits)
            rerank_reason = self._last_rerank_reason
            return {
                "mode": mode,
                "top_k": size,
                "document_title": document_title,
                "results": hits,
                "reranked": reranked,
                "rerank_requested": should_rerank,
                "rerank_reason": rerank_reason,
            }

        query_embedding = (await self.embedder.embed([query]))[0]

        if mode == "dense":
            hits = await self.store.dense_search(
                query_embedding=query_embedding,
                top_k=candidate_size,
                document_title=document_title,
            )
            hits = self._dedupe_hits(hits)
            hits, reranked = await self._rerank_hits(
                query=query,
                hits=hits,
                top_k=size,
                should_rerank=should_rerank,
            )
            hits = self._apply_query_intent_ranking(query=query, hits=hits)
            hits = self._finalize_ranks(hits)
            rerank_reason = self._last_rerank_reason
            return {
                "mode": mode,
                "top_k": size,
                "document_title": document_title,
                "results": hits,
                "reranked": reranked,
                "rerank_requested": should_rerank,
                "rerank_reason": rerank_reason,
            }

        hits = await self.store.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            top_k=candidate_size,
            document_title=document_title,
        )
        if mode == "hybrid_kg":
            graph_hits, graph_reason = await self._graph_search(
                query=query,
                top_k=candidate_size,
                document_title=document_title,
            )
            hits = self._rrf_fuse_multi_source(
                sources={
                    "chunk": hits,
                    "graph": graph_hits,
                },
                top_k=candidate_size,
                rrf_k=settings.RETRIEVAL_RRF_K,
            )
        else:
            graph_reason = "not_requested"
        hits = self._dedupe_hits(hits)
        hits, reranked = await self._rerank_hits(
            query=query,
            hits=hits,
            top_k=size,
            should_rerank=should_rerank,
        )
        hits = self._apply_query_intent_ranking(query=query, hits=hits)
        hits = self._finalize_ranks(hits)
        rerank_reason = self._last_rerank_reason
        return {
            "mode": mode,
            "top_k": size,
            "document_title": document_title,
            "results": hits,
            "reranked": reranked,
            "rerank_requested": should_rerank,
            "rerank_reason": rerank_reason,
            "graph_reason": graph_reason,
        }

    async def _rerank_hits(
        self,
        query: str,
        hits: list[dict],
        top_k: int,
        should_rerank: bool,
    ) -> tuple[list[dict], bool]:
        if not should_rerank:
            self._last_rerank_reason = "not_requested"
            return hits[:top_k], False
        reranked_hits, reranked, reason = await self.reranker.maybe_rerank(
            query=query,
            candidates=hits,
            top_k=top_k,
            force=should_rerank,
        )
        self._last_rerank_reason = reason
        return reranked_hits, reranked

    async def _get_graph_service(self) -> GraphitiService:
        if self.graph_service is None:
            self.graph_service = GraphitiService()
            await self.graph_service.build_indices()
        return self.graph_service

    async def _graph_search(
        self,
        query: str,
        top_k: int,
        document_title: str | None,
    ) -> tuple[list[dict[str, Any]], str]:
        group_ids: list[str] | None = None
        if document_title:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(Document.source_id).where(Document.title == document_title)
                )
                source_ids = [row[0] for row in result.all() if row[0]]
            if source_ids:
                group_ids = [
                    GraphitiService.normalize_group_id(source_id)
                    for source_id in source_ids
                ]

        try:
            graph = await self._get_graph_service()
            edges = await graph.graph.search(
                query=query,
                group_ids=group_ids,
                num_results=top_k,
            )
        except Exception as exc:
            return [], f"graph_lookup_exception:{type(exc).__name__}"

        hits: list[dict[str, Any]] = []
        for rank, edge in enumerate(edges, start=1):
            edge_uuid = str(getattr(edge, "uuid", f"graph-{rank}"))
            fact = getattr(edge, "fact", None)
            name = getattr(edge, "name", None)
            content = fact or name or "graph fact"
            hits.append(
                {
                    "id": f"graph:{edge_uuid}",
                    "score": 1.0 / rank,
                    "rank": rank,
                    "source": "graph",
                    "content": content,
                    "document_title": document_title,
                    "section_path": "graph_lookup",
                    "position": rank,
                    "content_hash": f"graph:{edge_uuid}",
                    "metadata": {
                        "group_id": getattr(edge, "group_id", None),
                        "source_node_uuid": str(getattr(edge, "source_node_uuid", "")),
                        "target_node_uuid": str(getattr(edge, "target_node_uuid", "")),
                        "episodes": [str(item) for item in (getattr(edge, "episodes", None) or [])],
                    },
                }
            )
        return hits, "ok" if hits else "empty"

    @staticmethod
    def _rrf_fuse_multi_source(
        sources: dict[str, list[dict[str, Any]]],
        top_k: int,
        rrf_k: int,
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}

        for label, hits in sources.items():
            for rank, hit in enumerate(hits, start=1):
                doc_id = str(hit.get("id"))
                if doc_id not in fused:
                    fused[doc_id] = {
                        **hit,
                        "rrf_score": 0.0,
                        "sources": [],
                    }
                fused[doc_id]["rrf_score"] += 1.0 / (rrf_k + rank)
                fused[doc_id]["sources"].append(label)

        ranked = sorted(
            fused.values(),
            key=lambda item: item.get("rrf_score", 0.0),
            reverse=True,
        )
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        return ranked[:top_k]

    @staticmethod
    def _dedupe_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for hit in hits:
            key = (
                hit.get("content_hash")
                or f"{hit.get('document_title')}|{hit.get('section_path')}|{ElasticsearchRetriever._content_fingerprint(hit.get('content') or '')}"
            )
            if key not in deduped:
                deduped[key] = hit
                continue
            current_score = float(hit.get("rrf_score") or hit.get("score") or 0.0)
            existing_score = float(deduped[key].get("rrf_score") or deduped[key].get("score") or 0.0)
            if current_score > existing_score:
                deduped[key] = hit
        return list(deduped.values())

    @staticmethod
    def _content_fingerprint(content: str) -> str:
        normalized = "".join(ch if ch.isalnum() else " " for ch in content.lower())
        tokens = [token for token in normalized.split() if token]
        return " ".join(tokens[:40])

    @staticmethod
    def _is_features_query(query: str) -> bool:
        q = query.lower()
        keywords = (
            "tính năng",
            "feature",
            "features",
            "capabilities",
            "chức năng",
        )
        return any(keyword in q for keyword in keywords)

    def _apply_query_intent_ranking(
        self,
        query: str,
        hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not hits:
            return hits

        if not self._is_features_query(query):
            return hits

        def section_score(hit: dict[str, Any]) -> float:
            section = (hit.get("section_path") or "").lower()
            content = (hit.get("content") or "").lower()[:500]
            score = 0.0
            if "feature" in section or "capabil" in section:
                score += 3.0
            if "- [x]" in content or "twitter bot" in content or "youtube shorts" in content:
                score += 1.5
            if any(token in section for token in ("version", "installation", "code_of_conduct", "acknowledgment")):
                score -= 1.5
            return score

        annotated = []
        for order, hit in enumerate(hits):
            annotated.append(
                (
                    section_score(hit),
                    float(hit.get("rrf_score") or hit.get("score") or 0.0),
                    -order,
                    hit,
                )
            )
        annotated.sort(reverse=True)
        return [item[3] for item in annotated]

    @staticmethod
    def _finalize_ranks(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for idx, hit in enumerate(hits, start=1):
            if "retrieval_rank" not in hit and "rank" in hit:
                hit["retrieval_rank"] = hit["rank"]
            hit["rank"] = idx
        return hits
