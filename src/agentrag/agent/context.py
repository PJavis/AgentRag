from __future__ import annotations

from typing import Any

from src.agentrag.config import settings


def apply_relevance_floor(items: list[dict[str, Any]], floor: float) -> list[dict[str, Any]]:
    """Drop candidates whose reranker relevance (rerank_score, sigmoid 0–1) is
    below floor. No-op when NO item carries a score (reranker off / non
    cross-encoder backend) — never gate on a missing signal. Items lacking a
    score while others have one are KEPT (conservative — don't drop on missing)."""
    if not any(it.get("rerank_score") is not None for it in items):
        return items
    return [
        it for it in items
        if it.get("rerank_score") is None or float(it.get("rerank_score") or 0.0) >= floor
    ]


def _estimate_tokens(text: str) -> int:
    """Cheap char-density token estimate; matches observability/cost.py."""
    if not text:
        return 0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return int((len(text) - non_ascii) / 4 + non_ascii * 0.55) + 1


def _lost_in_middle_reorder(items: list[Any]) -> list[Any]:
    """Reorder a ranked list so highest-ranked entries sit at the start and
    the end of the sequence (weak in the middle). Empirically improves LLM
    attention on long contexts (Liu et al., "Lost in the Middle", 2023).

    With ranks [r1, r2, r3, r4, r5] (best→worst) the output is
    [r1, r3, r5, r4, r2] — best first, second-best last, etc.
    """
    if len(items) <= 2:
        return list(items)
    front: list[Any] = []
    back: list[Any] = []
    for i, item in enumerate(items):
        (front if i % 2 == 0 else back).append(item)
    return front + list(reversed(back))


class ContextAssembler:
    def __init__(self) -> None:
        # Reused for the GLOBAL rerank pass below. Per-mode rerank (inside each
        # retriever.search) is discarded once modes are combined + re-sorted, so
        # we re-rank the final candidate set here in one shot.
        from src.agentrag.retrieval.reranker import LLMReranker

        self._reranker = LLMReranker()
        # Lazily-connected ES retriever for the raw-question pool injection below.
        from src.agentrag.retrieval.elasticsearch_retriever import ElasticsearchRetriever

        self._retriever = ElasticsearchRetriever()

    async def _inject_raw_query_hits(
        self, question: str, retrieved: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Merge a plain-hybrid retrieval on the RAW question into the candidate pool.

        The agent's decide-step rewrites can retrieve worse chunks than the raw question
        (hybrid_kg + query variants), dropping the best chunk out of the pool and the
        packed-context max rerank below the abstain floor → flaky false-abstain. Adding the
        raw-question hits guarantees rewrites only ADD; dedupe merges overlaps, the global
        rerank (also keyed on the raw question) then scores the gold chunk correctly. Failure
        is non-fatal — fall back to the tool-only pool."""
        if not question.strip():
            return retrieved
        try:
            res = await self._retriever.search(
                query=question, mode="hybrid", top_k=settings.RETRIEVAL_RAW_QUERY_TOP_K
            )
            hits = res.get("results", []) if isinstance(res, dict) else []
        except Exception:
            return retrieved
        return retrieved + [h for h in hits if isinstance(h, dict)]

    async def assemble(self, question: str, tool_results: list[dict[str, Any]]) -> dict[str, Any]:
        retrieved = self._stage_retrieve(tool_results)
        if settings.RETRIEVAL_INCLUDE_RAW_QUERY:
            retrieved = await self._inject_raw_query_hits(question, retrieved)
        deduped = self._stage_dedupe(retrieved)
        # Rerank the FULL deduped pool FIRST, then trim — so the cross-encoder,
        # not the weak pre-rerank rrf heuristic, decides answer-context MEMBERSHIP.
        # A wide recall pool surfaces gold that ranks deep on rrf (fusion rank
        # 8–17); reranking before the trim lets that gold win a packed slot while
        # high-rrf distractors are trimmed by their low rerank score. (2026-07-16
        # agent-path dig: rrf-keyed trim + wide pool crowded gold out of the pack.)
        reranked = await self._stage_global_rerank(question, deduped)
        ranked = self._stage_rank_trim(question, reranked)
        # Relevance-floor gate (default OFF): drop distractors below the rerank
        # floor BEFORE the answer node sees them. No-op unless a cross-encoder
        # rerank_score is present, so it only bites with local_cross_encoder on.
        if settings.RETRIEVAL_RELEVANCE_GATE_ENABLED:
            ranked = apply_relevance_floor(ranked, settings.RETRIEVAL_RELEVANCE_FLOOR)
        packed = self._stage_citation_pack(ranked)
        return {
            "retrieved": retrieved,
            "deduped": deduped,
            "ranked": ranked,
            "packed_context": packed,
        }

    async def _stage_global_rerank(
        self, question: str, items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not settings.RETRIEVAL_RERANK_ENABLED or len(items) <= 1:
            return items
        try:
            reranked, ok, _reason = await self._reranker.maybe_rerank(
                question, items, top_k=len(items), force=True
            )
            return reranked if ok else items
        except Exception:
            return items

    def _stage_retrieve(self, tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for item in tool_results:
            results = item.get("results")
            if isinstance(results, list):
                candidates.extend(results)
            if isinstance(item.get("segments"), list):
                candidates.extend(item["segments"])
            if isinstance(item.get("hybrid", {}).get("results"), list):
                candidates.extend(item["hybrid"]["results"])
            if isinstance(item.get("sparse", {}).get("results"), list):
                candidates.extend(item["sparse"]["results"])
            if isinstance(item.get("dense", {}).get("results"), list):
                candidates.extend(item["dense"]["results"])
        return candidates

    def _stage_dedupe(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            key = (
                candidate.get("content_hash")
                or f"{candidate.get('document_title')}|{candidate.get('section_path')}|{candidate.get('position')}"
                or candidate.get("id")
            )
            if key not in seen:
                seen[key] = candidate
                continue
            current_score = candidate.get("rrf_score") or candidate.get("score") or 0.0
            existing_score = seen[key].get("rrf_score") or seen[key].get("score") or 0.0
            if current_score > existing_score:
                seen[key] = candidate
        return list(seen.values())

    def _stage_rank_trim(self, question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_tokens = self._tokenize(question)

        def rank_score(item: dict[str, Any]) -> float:
            # When the cross-encoder ran (rerank_score present), IT decides order
            # and membership — the weak rrf/overlap heuristic below is only the
            # fallback for a rerank-disabled / non-cross-encoder pool.
            rr = item.get("rerank_score")
            if rr is not None:
                return float(rr)
            base = float(item.get("rrf_score") or item.get("score") or 0.0)
            section = (item.get("section_path") or "").lower()
            content = (item.get("content") or "").lower()
            combined = f"{section} {content[:500]}"
            overlap = len(query_tokens & self._tokenize(combined))
            source = item.get("source")
            source_boost = {
                "graph": 0.08,
                "structmem": 0.08,
                "synthesis": 0.07,
                "hybrid": 0.06,
                "sparse": 0.03,
            }.get(source or "", 0.0)
            return base + overlap * 0.2 + source_boost

        ranked = sorted(
            candidates,
            key=rank_score,
            reverse=True,
        )

        # Token-budget trim — keep chunks until the accumulated content tokens
        # exceed AGENT_MAX_CONTEXT_TOKENS. Falls back to chunk-count cap when
        # budget is 0 (preserves old behaviour).
        token_budget = settings.AGENT_MAX_CONTEXT_TOKENS
        if token_budget > 0:
            selected: list[dict[str, Any]] = []
            used = 0
            deferred: list[dict[str, Any]] = []
            per_bucket: dict[Any, int] = {}
            # Coverage diversity: a long/detailed answer needs material from the
            # WHOLE document, but pure relevance-order fill stacks the top-scoring
            # section (e.g. the intro) and starves later parts. Cap how many chunks
            # come from the same page/section in the first pass, then backfill any
            # remaining budget. Spreads context across the doc so each part can be
            # detailed instead of padding page 1.
            def _bucket(it: dict[str, Any]) -> Any:
                # Key by (document, page/section) so coverage spreads across BOTH
                # documents and within-document parts — multi-doc queries must not
                # let one file's pages crowd out the others (doc A p1 ≠ doc B p1).
                doc = it.get("document_title") or ""
                part = it.get("page_start") or it.get("section_path") or it.get("position") or id(it)
                return (doc, part)

            for item in ranked:  # relevance order
                t = _estimate_tokens(item.get("content") or "")
                b = _bucket(item)
                if per_bucket.get(b, 0) >= 3:
                    deferred.append(item)
                    continue
                if selected and used + t > token_budget:
                    continue
                selected.append(item)
                used += t
                per_bucket[b] = per_bucket.get(b, 0) + 1
            # Backfill leftover budget with the deferred (still relevance-ordered).
            for item in deferred:
                t = _estimate_tokens(item.get("content") or "")
                if used + t > token_budget:
                    continue
                selected.append(item)
                used += t
            if not selected and ranked:
                selected = ranked[:1]
        else:
            selected = ranked[: settings.AGENT_MAX_CONTEXT_CHUNKS]

        _structmem_sources = {"graph", "structmem", "synthesis"}
        has_graph_candidate = any(item.get("source") in _structmem_sources for item in ranked)
        has_graph_selected = any(item.get("source") in _structmem_sources for item in selected)
        if has_graph_candidate and not has_graph_selected and selected:
            best_graph = next((item for item in ranked if item.get("source") in _structmem_sources), None)
            if best_graph is not None:
                selected[-1] = best_graph
        return selected

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token
            for token in "".join(ch if ch.isalnum() else " " for ch in text.lower()).split()
            if len(token) > 2
        }

    def _stage_citation_pack(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        packed = []
        for idx, item in enumerate(candidates):
            page_start = item.get("page_start")
            page_end = item.get("page_end")
            # image_url may live at top level (ES segment doc) OR inside metadata.
            image_url = item.get("image_url") or (item.get("metadata") or {}).get("image_url")
            entry: dict[str, Any] = {
                # Stable citation number = position in this relevance-ordered list.
                # The answer prompt shows this and the model cites it as [source],
                # so [n] aligns with retrieval_context[n-1] regardless of any
                # lost-in-middle display reordering in the prompt.
                "source": idx + 1,
                "document_title": item.get("document_title"),
                "section_path": item.get("section_path"),
                "position": item.get("position"),
                "content_hash": item.get("content_hash"),
                "segment_type": item.get("segment_type", "text"),
                "node_level": item.get("node_level", 0),
                # Cross-encoder relevance (sigmoid 0–1) when the local reranker
                # ran; None otherwise. Exposed for gate tuning / eval.
                "rerank_score": item.get("rerank_score"),
                "context_text": item.get("context_text"),
                "page_start": page_start,
                "page_end": page_end,
                "excerpt": (item.get("content") or "")[:1500],
                # Preserve image_url so the answer node can pass bytes to a
                # multimodal LLM (Gemini 2.5 Flash) for visual grounding.
                "image_url": image_url,
            }
            # Convenience field: human-readable page reference
            if page_start is not None:
                entry["page"] = page_start if page_start == page_end else f"{page_start}-{page_end}"
            packed.append(entry)
        return packed
