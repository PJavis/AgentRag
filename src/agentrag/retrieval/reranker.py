from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from urllib.parse import urlparse

import aiohttp
from openai import AsyncOpenAI

from src.agentrag.config import settings

logger = logging.getLogger(__name__)


class LLMReranker:
    def __init__(self):
        self.enabled = settings.RETRIEVAL_RERANK_ENABLED
        self.top_n = settings.RETRIEVAL_RERANK_TOP_N
        self.backend = settings.RETRIEVAL_RERANK_BACKEND
        self._local_cross_encoder: Any | None = None
        self.temperature = settings.RETRIEVAL_RERANK_TEMPERATURE

        if self.backend == "local_cross_encoder":
            self.provider = "local_cross_encoder"
            self.model = (
                settings.RETRIEVAL_RERANK_MODEL
                or "dengcao/bge-reranker-v2-m3"
            )
            self.base_url = None
            self.client = None
            return

        self.provider, self.model, base_url, api_key = self._resolve_backend()
        self.base_url = base_url
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def candidate_size(self, requested_top_k: int, force: bool = False) -> int:
        if not self.enabled and not force:
            return requested_top_k
        return max(requested_top_k, self.top_n)

    async def maybe_rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
        force: bool = False,
    ) -> tuple[list[dict[str, Any]], bool, str]:
        if not self.enabled and not force:
            return candidates[:top_k], False, "disabled_by_config"
        if self.model is None:
            return candidates[:top_k], False, "reranker_model_not_initialized"
        if self.backend != "local_cross_encoder" and self.client is None:
            return candidates[:top_k], False, "reranker_client_not_initialized"
        if len(candidates) <= 1:
            return candidates[:top_k], False, "not_enough_candidates"

        scoped = candidates[: self.top_n]
        payload = {
            "query": query,
            "candidates": [
                {
                    "id": str(item.get("id")),
                    "document_title": item.get("document_title"),
                    "section_path": item.get("section_path"),
                    "content": (item.get("content") or "")[:700],
                }
                for item in scoped
                if item.get("id") is not None
            ],
        }
        if not payload["candidates"]:
            return candidates[:top_k], False, "no_candidate_ids"

        if self.backend == "local_cross_encoder":
            ordered_ids, local_reason = await self._try_local_cross_encoder_rerank(
                query=query,
                scoped=scoped,
            )
            if ordered_ids:
                return (
                    self._apply_ordered_ids(candidates, scoped, ordered_ids, top_k),
                    True,
                    "ok_local_cross_encoder",
                )
            return candidates[:top_k], False, local_reason

        native_reason = ""
        if self.provider == "ollama":
            ordered_ids, native_reason = await self._try_ollama_native_rerank(query, scoped)
            if ordered_ids:
                return self._apply_ordered_ids(candidates, scoped, ordered_ids, top_k), True, "ok_ollama_native"

        system_prompt = (
            "You are a passage reranker. Rank candidate passages by relevance to the query. "
            "Return strict JSON with key ordered_ids (best first). Only include candidate ids."
        )
        user_prompt = json.dumps(payload, ensure_ascii=True)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            ranking = json.loads(content)
            ordered_ids = self._extract_ordered_ids(ranking, scoped)
            if not ordered_ids:
                if native_reason:
                    return candidates[:top_k], False, f"invalid_reranker_output_unrecognized_schema|{native_reason}"
                return candidates[:top_k], False, "invalid_reranker_output_unrecognized_schema"
        except Exception as exc:
            logger.warning("Reranker failed, fallback to original ranking: %s", exc)
            if native_reason:
                return candidates[:top_k], False, f"reranker_exception:{type(exc).__name__}|{native_reason}"
            return candidates[:top_k], False, f"reranker_exception:{type(exc).__name__}"

        return self._apply_ordered_ids(candidates, scoped, ordered_ids, top_k), True, "ok"

    def _apply_ordered_ids(
        self,
        candidates: list[dict[str, Any]],
        scoped: list[dict[str, Any]],
        ordered_ids: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        id_to_item = {str(item.get("id")): item for item in scoped if item.get("id") is not None}
        reranked: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for item_id in ordered_ids:
            key = str(item_id)
            item = id_to_item.get(key)
            if item is None or key in used_ids:
                continue
            used_ids.add(key)
            reranked.append(item)

        for item in scoped:
            key = str(item.get("id"))
            if key in used_ids:
                continue
            reranked.append(item)

        final = reranked + candidates[self.top_n :]
        for rank, item in enumerate(final, start=1):
            item["rerank_rank"] = rank
        return final[:top_k]

    async def _try_ollama_native_rerank(
        self,
        query: str,
        scoped: list[dict[str, Any]],
    ) -> tuple[list[str], str]:
        if not self.base_url:
            return [], "ollama_native_no_base_url"
        endpoint = self._to_ollama_rerank_endpoint(self.base_url)
        documents = [(item.get("content") or "")[:1200] for item in scoped]
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(self.top_n, len(documents)),
        }
        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, json=payload) as response:
                    if response.status >= 400:
                        body = await response.text()
                        return [], f"ollama_native_http_{response.status}:{body[:120]}"
                    data = await response.json()
        except Exception as exc:
            return [], f"ollama_native_exception:{type(exc).__name__}"

        results = data.get("results")
        if not isinstance(results, list):
            return [], "ollama_native_missing_results"

        ordered_ids: list[str] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            idx_value = item.get("index")
            if not isinstance(idx_value, int):
                idx_value = item.get("document_index")
            if not isinstance(idx_value, int) or idx_value < 0 or idx_value >= len(scoped):
                continue
            candidate_id = scoped[idx_value].get("id")
            if candidate_id is not None:
                ordered_ids.append(str(candidate_id))

        if not ordered_ids:
            return [], "ollama_native_unrecognized_results"
        return ordered_ids, "ok"

    def _get_local_cross_encoder(self):
        if self._local_cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._local_cross_encoder = CrossEncoder(
                self.model,
                trust_remote_code=True,
            )
        return self._local_cross_encoder

    async def _try_local_cross_encoder_rerank(
        self,
        query: str,
        scoped: list[dict[str, Any]],
    ) -> tuple[list[str], str]:
        try:
            model = await asyncio.to_thread(self._get_local_cross_encoder)
            pairs = [
                (query, (item.get("content") or "")[:1600])
                for item in scoped
            ]
            scores = await asyncio.to_thread(model.predict, pairs)
        except Exception as exc:
            return [], f"local_cross_encoder_exception:{type(exc).__name__}"

        try:
            indexed = list(enumerate(scores))
            indexed.sort(key=lambda item: float(item[1]), reverse=True)
        except Exception as exc:
            return [], f"local_cross_encoder_invalid_scores:{type(exc).__name__}"

        ordered_ids: list[str] = []
        for idx, _score in indexed:
            if not (0 <= idx < len(scoped)):
                continue
            item_id = scoped[idx].get("id")
            if item_id is not None:
                ordered_ids.append(str(item_id))

        if not ordered_ids:
            return [], "local_cross_encoder_no_rankable_candidates"
        return ordered_ids, "ok"

    @staticmethod
    def _to_ollama_rerank_endpoint(base_url: str) -> str:
        parsed = urlparse(base_url)
        base_path = parsed.path or ""
        if base_path.endswith("/v1/"):
            base_path = base_path[:-4]
        elif base_path.endswith("/v1"):
            base_path = base_path[:-3]
        if not base_path.endswith("/"):
            base_path += "/"
        return f"{parsed.scheme}://{parsed.netloc}{base_path}api/rerank"

    def _extract_ordered_ids(
        self,
        ranking: Any,
        scoped: list[dict[str, Any]],
    ) -> list[str]:
        if not isinstance(ranking, dict):
            return []

        def _as_str_list(values: Any) -> list[str]:
            if not isinstance(values, list):
                return []
            return [str(item) for item in values]

        # 1) Preferred strict schema
        ordered_ids = _as_str_list(ranking.get("ordered_ids"))
        if ordered_ids:
            return ordered_ids

        # 2) Common aliases
        for key in ("ranked_ids", "ordered", "ids"):
            ids = _as_str_list(ranking.get(key))
            if ids:
                return ids

        # 3) Structured item lists: results/rankings/items
        id_to_candidate = {str(item.get("id")): item for item in scoped if item.get("id") is not None}
        for key in ("results", "rankings", "items"):
            entries = ranking.get(key)
            if not isinstance(entries, list):
                continue
            normalized: list[tuple[float, str]] = []
            for pos, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                item_id = entry.get("id")
                if item_id is None:
                    # many rerank APIs return index/document_index/corpus_id instead of id
                    for idx_key in ("index", "document_index", "corpus_id"):
                        idx_value = entry.get(idx_key)
                        if isinstance(idx_value, int) and 0 <= idx_value < len(scoped):
                            item_id = scoped[idx_value].get("id")
                            break
                if item_id is None:
                    continue
                score = entry.get("score")
                if not isinstance(score, (int, float)):
                    score = entry.get("relevance_score")
                sort_key = float(score) if isinstance(score, (int, float)) else float(-pos)
                candidate_id = str(item_id)
                if candidate_id in id_to_candidate:
                    normalized.append((sort_key, candidate_id))
            if normalized:
                normalized.sort(key=lambda item: item[0], reverse=True)
                return [candidate_id for _, candidate_id in normalized]

        return []

    def _resolve_backend(self) -> tuple[str, str, str | None, str]:
        provider = (
            settings.RETRIEVAL_RERANK_PROVIDER
            or settings.AGENT_PROVIDER
            or settings.EXTRACTION_PROVIDER
        )
        model = (
            settings.RETRIEVAL_RERANK_MODEL
            or settings.AGENT_MODEL
            or settings.EXTRACTION_MODEL
        )
        base_override = settings.RETRIEVAL_RERANK_BASE_URL

        if not model:
            raise ValueError("RETRIEVAL_RERANK_MODEL or AGENT_MODEL or EXTRACTION_MODEL is required")

        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required when reranker provider resolves to openai")
            return (
                provider,
                model,
                base_override or settings.AGENT_BASE_URL or settings.EXTRACTION_BASE_URL,
                settings.OPENAI_API_KEY,
            )
        if provider == "ollama":
            return (
                provider,
                model,
                base_override or settings.AGENT_BASE_URL or settings.EXTRACTION_BASE_URL or settings.OLLAMA_BASE_URL,
                settings.OLLAMA_API_KEY,
            )
        if provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is required when reranker provider resolves to gemini")
            return (
                provider,
                model,
                base_override
                or settings.AGENT_BASE_URL
                or settings.EXTRACTION_BASE_URL
                or "https://generativelanguage.googleapis.com/v1beta/openai/",
                settings.GEMINI_API_KEY,
            )
        if provider == "hf_inference":
            if not settings.HF_TOKEN:
                raise ValueError("HF_TOKEN is required when reranker provider resolves to hf_inference")
            return (
                provider,
                model,
                base_override
                or settings.AGENT_BASE_URL
                or settings.EXTRACTION_BASE_URL
                or settings.HF_OPENAI_BASE_URL,
                settings.HF_TOKEN,
            )

        raise ValueError(f"Unsupported reranker provider: {provider}")
