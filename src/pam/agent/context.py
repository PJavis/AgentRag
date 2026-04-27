from __future__ import annotations

from typing import Any

from src.pam.config import settings


class ContextAssembler:
    def assemble(self, question: str, tool_results: list[dict[str, Any]]) -> dict[str, Any]:
        retrieved = self._stage_retrieve(tool_results)
        deduped = self._stage_dedupe(retrieved)
        ranked = self._stage_rank_trim(question, deduped)
        packed = self._stage_citation_pack(ranked)
        return {
            "retrieved": retrieved,
            "deduped": deduped,
            "ranked": ranked,
            "packed_context": packed,
        }

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
        for item in candidates:
            packed.append(
                {
                    "document_title": item.get("document_title"),
                    "section_path": item.get("section_path"),
                    "position": item.get("position"),
                    "content_hash": item.get("content_hash"),
                    "excerpt": (item.get("content") or "")[:1500],
                }
            )
        return packed
