from __future__ import annotations

import hashlib
import json
from typing import Any

from cachetools import TTLCache
from sqlalchemy import select

from src.agentrag.config import settings
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document
from src.agentrag.graph.structmem_service import StructMemService
from src.agentrag.ingestion.embedders.factory import build_embedding_provider
from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore
from src.agentrag.retrieval.reranker import LLMReranker


# Module-level result cache shared across retriever instances. 60-second TTL
# is short enough to not stall content updates but big enough to make repeated
# user clicks (which often re-issue the same query) feel instant.
_RESULT_CACHE: TTLCache[str, dict] = TTLCache(maxsize=512, ttl=60)


def _cache_key(query: str, mode: str, top_k: int | None, document_title: str | None, rerank: bool | None, extra: dict | None = None) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([query, mode, top_k, document_title, rerank, extra], ensure_ascii=False, sort_keys=True).encode())
    return h.hexdigest()


class ElasticsearchRetriever:
    def __init__(self):
        self.store = ElasticsearchStore()
        self.embedder = build_embedding_provider(settings)
        self.reranker = LLMReranker()
        self._last_rerank_reason = "not_attempted"

    async def search(
        self,
        query: str,
        mode: str = "hybrid_kg",
        top_k: int | None = None,
        document_title: str | None = None,
        rerank: bool | None = None,
        dense_query: str | None = None,
        filters: dict | None = None,
    ) -> dict:
        """Public entrypoint. Runs `_search_impl` once with the supplied
        filters; if that returns zero hits, retries with only the SOFT domain
        filters (systems/specialties) relaxed — so we still ground the answer
        when domain tags haven't been backfilled — but NEVER drops the hard
        `document_titles` scope. Dropping it would leak other notebooks' /
        sources' documents into a scoped chat (e.g. an empty notebook returning
        a different notebook's doc)."""
        payload = await self._search_impl(
            query=query,
            mode=mode,
            top_k=top_k,
            document_title=document_title,
            rerank=rerank,
            dense_query=dense_query,
            filters=filters,
        )
        if filters and not payload.get("results"):
            # Keep the hard document scope; relax only soft domain filters.
            hard = {k: v for k, v in filters.items() if k == "document_titles"}
            # Nothing soft to relax → the scope itself is empty; do not fall back.
            if hard == dict(filters):
                return payload
            fallback = await self._search_impl(
                query=query,
                mode=mode,
                top_k=top_k,
                document_title=document_title,
                rerank=rerank,
                dense_query=dense_query,
                filters=hard or None,
            )
            fallback["domain_filter_fallback"] = True
            fallback["domain_filter_attempted"] = filters
            return fallback
        return payload

    async def _search_impl(
        self,
        query: str,
        mode: str = "hybrid_kg",
        top_k: int | None = None,
        document_title: str | None = None,
        rerank: bool | None = None,
        dense_query: str | None = None,
        filters: dict | None = None,
    ) -> dict:
        """`query` drives BM25 + rerank + intent ranking (kept clean for keyword match).
        `dense_query` is embedded for kNN — pass HyDE-augmented text here. Falls back to `query` when None.
        `filters` carries S5 domain filters: {"systems": [...], "specialties": [...]}.
        """
        if mode not in {"sparse", "dense", "hybrid", "hybrid_kg"}:
            raise ValueError("mode must be one of: sparse, dense, hybrid, hybrid_kg")

        ck = _cache_key(
            query, mode, top_k, document_title, rerank,
            extra=filters,
        )
        cached = _RESULT_CACHE.get(ck)
        if cached is not None:
            return cached

        size = top_k or settings.RETRIEVAL_TOP_K
        lexical_query = self._rewrite_query(query)
        embed_text = dense_query if dense_query else query
        should_rerank = settings.RETRIEVAL_RERANK_ENABLED if rerank is None else rerank
        candidate_size = self.reranker.candidate_size(size, force=should_rerank)

        if mode == "sparse":
            hits = await self.store.sparse_search(
                query=lexical_query,
                top_k=candidate_size,
                document_title=document_title,
                filters=filters,
            )
            hits = self._dedupe_hits(hits)
            hits, reranked = await self._rerank_hits(
                query=query,
                hits=hits,
                top_k=size,
                should_rerank=should_rerank,
            )
            hits = self._apply_query_intent_ranking(query=query, hits=hits)
            hits = self._balance_segment_types_for_query(query, hits, size)
            hits = self._finalize_ranks(hits)
            rerank_reason = self._last_rerank_reason
            payload = {
                "mode": mode,
                "top_k": size,
                "document_title": document_title,
                "results": hits,
                "reranked": reranked,
                "rerank_requested": should_rerank,
                "rerank_reason": rerank_reason,
            }
            _RESULT_CACHE[ck] = payload
            return payload

        query_embedding = (await self.embedder.embed([embed_text]))[0]

        if mode == "dense":
            hits = await self.store.dense_search(
                query_embedding=query_embedding,
                top_k=candidate_size,
                document_title=document_title,
                filters=filters,
            )
            hits = self._dedupe_hits(hits)
            hits, reranked = await self._rerank_hits(
                query=query,
                hits=hits,
                top_k=size,
                should_rerank=should_rerank,
            )
            hits = self._apply_query_intent_ranking(query=query, hits=hits)
            hits = self._balance_segment_types_for_query(query, hits, size)
            hits = self._finalize_ranks(hits)
            rerank_reason = self._last_rerank_reason
            payload = {
                "mode": mode,
                "top_k": size,
                "document_title": document_title,
                "results": hits,
                "reranked": reranked,
                "rerank_requested": should_rerank,
                "rerank_reason": rerank_reason,
            }
            _RESULT_CACHE[ck] = payload
            return payload

        hits = await self.store.hybrid_search(
            query=lexical_query,
            query_embedding=query_embedding,
            top_k=candidate_size,
            document_title=document_title,
            filters=filters,
        )
        if mode == "hybrid_kg":
            if self._should_use_graph(query):
                entry_hits, graph_reason = await self._entries_search(
                    query=lexical_query,
                    query_embedding=query_embedding,
                    top_k=candidate_size,
                    document_title=document_title,
                )
                hits = self._rrf_fuse_multi_source(
                    sources={
                        "chunk": hits,
                        "structmem": entry_hits,
                    },
                    top_k=candidate_size,
                    rrf_k=settings.RETRIEVAL_RRF_K,
                )
            else:
                graph_reason = "skipped_by_intent"
        else:
            graph_reason = "not_requested"
        # P2.9 Multi-vector retrieval: when query is image-intent AND
        # visual embedding enabled, also run CLIP visual kNN and RRF-fuse
        # those image hits into the candidate pool. Lets the retriever surface
        # X-quang / atlas images even when their text caption is sparse.
        if (
            getattr(settings, "VISUAL_EMBEDDING_ENABLED", False)
            and self._is_image_intent(query)
        ):
            try:
                visual_hits = await self.store.visual_search(
                    query_text=query,
                    top_k=candidate_size,
                    document_title=document_title,
                    filters=filters,
                )
            except Exception:
                visual_hits = []
            if visual_hits:
                hits = self._rrf_fuse_multi_source(
                    sources={
                        "primary": hits,
                        "visual": visual_hits,
                    },
                    top_k=candidate_size,
                    rrf_k=settings.RETRIEVAL_RRF_K,
                )
        hits = self._dedupe_hits(hits)
        hits, reranked = await self._rerank_hits(
            query=query,
            hits=hits,
            top_k=size,
            should_rerank=should_rerank,
        )
        hits = self._apply_query_intent_ranking(query=query, hits=hits)
        hits = self._balance_segment_types_for_query(query, hits, size)
        hits = self._finalize_ranks(hits)
        rerank_reason = self._last_rerank_reason
        payload = {
            "mode": mode,
            "top_k": size,
            "document_title": document_title,
            "results": hits,
            "reranked": reranked,
            "rerank_requested": should_rerank,
            "rerank_reason": rerank_reason,
            "graph_reason": graph_reason,
        }
        _RESULT_CACHE[ck] = payload
        return payload

    async def _entries_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        document_title: str | None,
    ) -> tuple[list[dict[str, Any]], str]:
        import asyncio

        group_ids: list[str] | None = None
        if document_title:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(Document.source_id).where(Document.title == document_title)
                )
                source_ids = [row[0] for row in result.all() if row[0]]
            if source_ids:
                group_ids = [
                    StructMemService.normalize_group_id(sid) for sid in source_ids
                ]

        try:
            entry_hits, synth_hits = await asyncio.gather(
                self.store.search_entries(
                    query_embedding=query_embedding,
                    query_text=query,
                    group_ids=group_ids,
                    top_k=top_k,
                ),
                self.store.search_synthesis(
                    query_embedding=query_embedding,
                    query_text=query,
                    group_ids=group_ids,
                    top_k=max(1, top_k // 2),
                ),
            )
        except Exception as exc:
            return [], f"entries_lookup_exception:{type(exc).__name__}"

        combined = entry_hits + synth_hits
        return combined, "ok" if combined else "empty"

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

    _IMAGE_INTENT_TOKENS = (
        "hình", "ảnh", "atlas", "sơ đồ", "biểu đồ", "đồ thị", "x-quang", "x quang",
        "siêu âm", "mri", "ct scan", "ecg", "điện tim", "vẽ", "minh hoạ", "minh họa",
        "show me", "image", "picture", "diagram", "chart", "figure", "scan", "x-ray",
    )

    @classmethod
    def _is_image_intent(cls, query: str) -> bool:
        """Detect queries that explicitly ask for visual content. When true,
        we relax the image-ratio cap so retrieval surfaces images first."""
        q = (query or "").lower()
        return any(tok in q for tok in cls._IMAGE_INTENT_TOKENS)

    @classmethod
    def _balance_segment_types_for_query(
        cls, query: str, hits: list[dict[str, Any]], top_k: int,
    ) -> list[dict[str, Any]]:
        """Wrapper that bumps the image cap when the query asks for visuals."""
        if cls._is_image_intent(query):
            # Allow up to 70% images when user explicitly wants visuals.
            return cls._balance_segment_types(hits, top_k, max_ratio_override=0.7)
        return cls._balance_segment_types(hits, top_k)

    @staticmethod
    def _balance_segment_types(
        hits: list[dict[str, Any]],
        top_k: int,
        max_ratio_override: float | None = None,
    ) -> list[dict[str, Any]]:
        """Cap the image-segment fraction inside the top_k window.

        Scanned-PDF docs often produce vision descriptions whose English-rich
        text out-matches the source Vietnamese chunks on dense kNN. The
        answer LLM then summarises figure captions instead of body text.
        Demote excess images by swapping them with the next text hit from
        the tail, preserving relative order otherwise.

        Pass `max_ratio_override` to relax the cap for image-intent queries.
        """
        if not hits or top_k <= 0:
            return hits
        max_ratio = (
            max_ratio_override
            if max_ratio_override is not None
            else getattr(settings, "RETRIEVAL_MAX_IMAGE_RATIO", 0.3)
        )
        max_images = max(0, int(top_k * max_ratio))
        head = hits[:top_k]
        tail = hits[top_k:]
        image_idxs_in_head = [
            i for i, h in enumerate(head)
            if h.get("segment_type") == "image"
        ]
        if len(image_idxs_in_head) <= max_images:
            return hits
        text_idxs_in_tail = [
            i for i, h in enumerate(tail)
            if h.get("segment_type") != "image"
        ]
        if not text_idxs_in_tail:
            return hits
        # Demote the worst-ranked excess images (head's later positions first).
        excess = len(image_idxs_in_head) - max_images
        # Take the `excess` worst-ranked image slots in head, then pair each
        # with text in head-rank order so the strongest text lands at the
        # earliest demoted slot.
        worst_image_slots = sorted(image_idxs_in_head, reverse=True)[:excess]
        worst_image_slots.sort()  # ascending → swap from earliest first
        to_promote = text_idxs_in_tail[: len(worst_image_slots)]
        if len(to_promote) < len(worst_image_slots):
            worst_image_slots = worst_image_slots[: len(to_promote)]
        for demote_i, promote_j in zip(worst_image_slots, to_promote, strict=True):
            head[demote_i], tail[promote_j] = tail[promote_j], head[demote_i]
        return head + tail

    def _rewrite_query(self, query: str) -> str:
        if not self._is_features_query(query):
            return query
        return f"{query} features capabilities tính năng chức năng"

    def _should_use_graph(self, query: str) -> bool:
        if self._is_features_query(query):
            return False
        q = query.lower()
        graph_intent_keywords = (
            "quan hệ",
            "relationship",
            "liên kết",
            "timeline",
            "temporal",
            "phụ thuộc",
            "depends on",
        )
        return any(keyword in q for keyword in graph_intent_keywords)
