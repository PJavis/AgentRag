from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Awaitable, Callable

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from openai import AsyncOpenAI

from src.pam.config import settings
from src.pam.graph.hf_embedder import HFGraphEmbedder

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    api_key: str
    base_url: str | None
    model: str


class GraphitiService:
    DEFAULT_EXTRACTION_INSTRUCTIONS = (
        "When extracting relationships, every source_entity_name and "
        "target_entity_name must exactly match an extracted node name from the current "
        "episode. If a relationship references a concept not yet listed as a node, "
        "add that concept as a node first. Do not emit edges with missing endpoints. "
        "Only mark duplicate_facts indexes that actually exist in the provided EXISTING FACTS list."
    )

    def __init__(self):
        llm_backend = self._build_llm_backend()

        llm_config = LLMConfig(
            api_key=llm_backend.api_key,
            model=llm_backend.model,
            base_url=llm_backend.base_url,
            temperature=settings.EXTRACTION_TEMPERATURE,
        )
        llm_client = OpenAIGenericClient(
            client=AsyncOpenAI(
                api_key=llm_backend.api_key,
                base_url=llm_backend.base_url,
            ),
            config=llm_config,
        )
        embedder = self._build_graph_embedder()
        reranker = OpenAIRerankerClient(client=llm_client, config=llm_config)

        self.graph = Graphiti(
            uri=settings.NEO4J_URI,
            user=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=reranker,
            store_raw_episode_content=True,
        )
        self._cache_dir = Path(settings.GRAPH_CACHE_DIR)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._provider_signature = sha256(
            "|".join(
                [
                    settings.EXTRACTION_PROVIDER,
                    settings.EXTRACTION_MODEL,
                    settings.EXTRACTION_BASE_URL or "",
                    settings.GRAPH_EMBEDDING_PROVIDER,
                    settings.GRAPH_EMBEDDING_MODEL,
                    settings.GRAPH_EMBEDDING_BASE_URL or "",
                ]
            ).encode("utf-8")
        ).hexdigest()

    def _build_llm_backend(self) -> OpenAICompatibleConfig:
        provider = settings.EXTRACTION_PROVIDER
        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is required when EXTRACTION_PROVIDER=openai"
                )
            return OpenAICompatibleConfig(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.EXTRACTION_BASE_URL,
                model=settings.EXTRACTION_MODEL,
            )

        if provider == "ollama":
            return OpenAICompatibleConfig(
                api_key=settings.OLLAMA_API_KEY,
                base_url=settings.EXTRACTION_BASE_URL or settings.OLLAMA_BASE_URL,
                model=settings.EXTRACTION_MODEL,
            )

        if provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError(
                    "GEMINI_API_KEY is required when EXTRACTION_PROVIDER=gemini"
                )
            return OpenAICompatibleConfig(
                api_key=settings.GEMINI_API_KEY,
                base_url=(
                    settings.EXTRACTION_BASE_URL
                    or "https://generativelanguage.googleapis.com/v1beta/openai/"
                ),
                model=settings.EXTRACTION_MODEL,
            )

        if provider == "hf_inference":
            if not settings.HF_TOKEN:
                raise ValueError(
                    "HF_TOKEN is required when EXTRACTION_PROVIDER=hf_inference"
                )
            return OpenAICompatibleConfig(
                api_key=settings.HF_TOKEN,
                base_url=settings.EXTRACTION_BASE_URL or settings.HF_OPENAI_BASE_URL,
                model=settings.EXTRACTION_MODEL,
            )

        raise ValueError(f"Unsupported extraction provider: {provider}")

    def _build_graph_embedder(self) -> EmbedderClient:
        provider = settings.GRAPH_EMBEDDING_PROVIDER
        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is required when GRAPH_EMBEDDING_PROVIDER=openai"
                )
            config = OpenAICompatibleConfig(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.GRAPH_EMBEDDING_BASE_URL,
                model=settings.GRAPH_EMBEDDING_MODEL,
            )
            return self._build_openai_embedder(config)

        if provider == "ollama":
            config = OpenAICompatibleConfig(
                api_key=settings.OLLAMA_API_KEY,
                base_url=(
                    settings.GRAPH_EMBEDDING_BASE_URL or settings.OLLAMA_BASE_URL
                ),
                model=settings.GRAPH_EMBEDDING_MODEL,
            )
            return self._build_openai_embedder(config)

        if provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError(
                    "GEMINI_API_KEY is required when GRAPH_EMBEDDING_PROVIDER=gemini"
                )
            config = OpenAICompatibleConfig(
                api_key=settings.GEMINI_API_KEY,
                base_url=(
                    settings.GRAPH_EMBEDDING_BASE_URL
                    or "https://generativelanguage.googleapis.com/v1beta/openai/"
                ),
                model=settings.GRAPH_EMBEDDING_MODEL,
            )
            return self._build_openai_embedder(config)

        if provider == "hf_inference":
            if not settings.HF_TOKEN:
                raise ValueError(
                    "HF_TOKEN is required when GRAPH_EMBEDDING_PROVIDER=hf_inference"
                )
            return HFGraphEmbedder(
                model=settings.GRAPH_EMBEDDING_MODEL,
                api_key=settings.HF_TOKEN,
            )

        raise ValueError(f"Unsupported graph embedding provider: {provider}")

    def _build_openai_embedder(
        self, backend: OpenAICompatibleConfig
    ) -> OpenAIEmbedder:
        embedder_config = OpenAIEmbedderConfig(
            api_key=backend.api_key,
            embedding_model=backend.model,
            base_url=backend.base_url,
        )
        return OpenAIEmbedder(config=embedder_config)

    async def extract_and_sync_chunk(
        self,
        chunk: dict[str, Any],
        group_id: str,
        chunk_id: str,
        document_ref: str,
    ) -> dict[str, Any]:
        episode = await self.graph.add_episode(
            name=f"Episode: {chunk_id}",
            episode_body=chunk["content"],
            source_description=(
                f"Extracted from document {document_ref}, segment {chunk_id}"
            ),
            reference_time=datetime.utcnow(),
            group_id=group_id,
            custom_extraction_instructions=self.DEFAULT_EXTRACTION_INSTRUCTIONS,
        )

        return {
            "episode_id": getattr(episode, "uuid", "unknown"),
            "entities": [node.name for node in getattr(episode, "nodes", [])],
            "relationships": [
                (
                    f"{edge.source_node_uuid} --"
                    f"{edge.fact or edge.name or 'RELATES_TO'}--> "
                    f"{edge.target_node_uuid}"
                )
                for edge in getattr(episode, "edges", [])
            ],
            "content_hash": chunk["content_hash"],
        }

    async def sync_chunks(
        self,
        chunks: list[dict[str, Any]],
        group_id: str,
        document_ref: str,
        progress_callback: ProgressCallback | None = None,
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []

        normalized_group_id = self.normalize_group_id(group_id)
        semaphore = asyncio.Semaphore(max(settings.GRAPH_MAX_CONCURRENCY, 1))

        async def run(index: int, chunk: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            async with semaphore:
                result = await self._sync_chunk_with_retry(
                    chunk=chunk,
                    group_id=normalized_group_id,
                    document_ref=document_ref,
                    chunk_index=index,
                )
                return index, result

        tasks = [asyncio.create_task(run(index, chunk)) for index, chunk in enumerate(chunks)]
        ordered_results: list[dict[str, Any] | None] = [None] * len(tasks)

        completed = 0
        try:
            for task in asyncio.as_completed(tasks):
                index, result = await task
                ordered_results[index] = result
                completed += 1
                if completed % 5 == 0 or completed == len(tasks):
                    logger.info(
                        "Graph extraction progress %s/%s for group %s",
                        completed,
                        len(tasks),
                        normalized_group_id,
                    )
                if progress_callback is not None:
                    payload = {
                        "completed": completed,
                        "total": len(tasks),
                        "group_id": normalized_group_id,
                        "chunk_index": index,
                        "cached": result.get("cached", False),
                        "duration_ms": result.get("duration_ms"),
                    }
                    maybe_awaitable = progress_callback(payload)
                    if maybe_awaitable is not None:
                        await maybe_awaitable
        except Exception:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        return [result for result in ordered_results if result is not None]

    async def _sync_chunk_with_retry(
        self,
        chunk: dict[str, Any],
        group_id: str,
        document_ref: str,
        chunk_index: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        cached = self._load_cached_result(
            group_id=group_id, chunk_hash=chunk["content_hash"]
        )
        if cached is not None:
            cached["cached"] = True
            cached["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return cached

        last_error: Exception | None = None
        chunk_id = f"{group_id}_chunk_{chunk_index}"

        for attempt in range(settings.GRAPH_CHUNK_RETRIES + 1):
            try:
                result = await asyncio.wait_for(
                    self.extract_and_sync_chunk(
                        chunk=chunk,
                        group_id=group_id,
                        chunk_id=chunk_id,
                        document_ref=document_ref,
                    ),
                    timeout=settings.GRAPH_CHUNK_TIMEOUT_SECONDS,
                )
                self._store_cached_result(
                    group_id=group_id,
                    chunk_hash=chunk["content_hash"],
                    payload=result,
                )
                result["cached"] = False
                result["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
                return result
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Graph extraction failed for chunk %s attempt %s/%s: %s",
                    chunk_id,
                    attempt + 1,
                    settings.GRAPH_CHUNK_RETRIES + 1,
                    exc,
                )
                if attempt < settings.GRAPH_CHUNK_RETRIES:
                    await asyncio.sleep(min(2 ** attempt, 5))

        assert last_error is not None
        raise last_error

    def _cache_path(self, group_id: str, chunk_hash: str) -> Path:
        key = sha256(
            f"{group_id}|{chunk_hash}|{self._provider_signature}".encode("utf-8")
        ).hexdigest()
        return self._cache_dir / f"{key}.json"

    @staticmethod
    def normalize_group_id(value: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
        return normalized or "default"

    def _load_cached_result(
        self, group_id: str, chunk_hash: str
    ) -> dict[str, Any] | None:
        if not settings.GRAPH_ENABLE_CACHE:
            return None
        path = self._cache_path(group_id=group_id, chunk_hash=chunk_hash)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _store_cached_result(
        self, group_id: str, chunk_hash: str, payload: dict[str, Any]
    ) -> None:
        if not settings.GRAPH_ENABLE_CACHE:
            return
        path = self._cache_path(group_id=group_id, chunk_hash=chunk_hash)
        path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    async def build_indices(self):
        logger.info("Building Graphiti indices and constraints")
        await self.graph.build_indices_and_constraints(delete_existing=False)
