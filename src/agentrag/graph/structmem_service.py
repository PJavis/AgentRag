"""StructMem extraction service — thay thế GraphitiService.

Mỗi chunk được xử lý với 2 LLM calls song song (factual + relational),
không cần Neo4j. Cache pattern giữ nguyên từ GraphitiService.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

from src.agentrag.config import settings

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

_FACTUAL_SYSTEM_PROMPT = """\
You are a knowledge extraction specialist. Extract precise, self-contained factual statements from a document passage.

A factual entry must be:
- Standalone (interpretable without surrounding context — include all specific names, numbers, versions)
- Grounded in specific details from the text (do not invent facts)
- Written as a declarative third-person statement

Output ONLY valid JSON with this exact schema:
{"factual_entries": [{"content": "<complete declarative statement>", "subject": "<primary entity or concept>", "fact_type": "<definition|property|event|measurement|constraint|procedure>", "confidence": "<high|medium|low>"}]}

Rules:
- Extract 3 to 12 entries depending on information density
- If a passage is a table or list, extract each meaningful row/item as a separate entry
- Do not output anything outside the JSON object\
"""

_RELATIONAL_SYSTEM_PROMPT = """\
You are a knowledge extraction specialist. Extract causal and relational dynamics between concepts in a document passage.

A relational entry captures how entities interact:
- Causal: A causes B, A enables B, A prevents B
- Dependency: A requires B, A is built on B
- Contrast: A differs from B in dimension C
- Sequence: A precedes B, A triggers B
- Comparison: A outperforms B on metric M

Output ONLY valid JSON with this exact schema:
{"relational_entries": [{"content": "<full natural-language description of the relationship>", "source_entity": "<left-side entity>", "target_entity": "<right-side entity>", "relation_type": "<causes|enables|requires|contrasts|precedes|compares|generalizes>", "confidence": "<high|medium|low>"}]}

Rules:
- Extract 2 to 8 entries per passage
- Both source_entity and target_entity must be clearly present or directly implied in the text
- content must be a full readable sentence, not just a label
- Do not infer relationships not supported by the text
- Do not output anything outside the JSON object\
"""


@dataclass(frozen=True)
class _LLMBackend:
    api_key: str
    base_url: str | None
    model: str


class StructMemService:
    """Dual-perspective extraction: 2 parallel LLM calls per chunk (factual + relational)."""

    def __init__(self):
        backend = self._build_llm_backend()
        self._client = AsyncOpenAI(
            api_key=backend.api_key,
            base_url=backend.base_url,
        )
        self._model = backend.model
        self._temperature = settings.EXTRACTION_TEMPERATURE

        self._cache_dir = Path(settings.STRUCTMEM_CACHE_DIR)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._provider_signature = sha256(
            "|".join(
                [
                    settings.EXTRACTION_PROVIDER,
                    settings.EXTRACTION_MODEL,
                    settings.EXTRACTION_BASE_URL or "",
                    "structmem_v1",  # versión tag to separate from Graphiti cache
                ]
            ).encode("utf-8")
        ).hexdigest()

    # ------------------------------------------------------------------
    # Provider routing — identical logic to GraphitiService._build_llm_backend
    # ------------------------------------------------------------------

    def _build_llm_backend(self) -> _LLMBackend:
        provider = settings.EXTRACTION_PROVIDER
        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required when EXTRACTION_PROVIDER=openai")
            return _LLMBackend(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.EXTRACTION_BASE_URL,
                model=settings.EXTRACTION_MODEL,
            )
        if provider == "ollama":
            return _LLMBackend(
                api_key=settings.OLLAMA_API_KEY,
                base_url=settings.EXTRACTION_BASE_URL or settings.OLLAMA_BASE_URL,
                model=settings.EXTRACTION_MODEL,
            )
        if provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is required when EXTRACTION_PROVIDER=gemini")
            return _LLMBackend(
                api_key=settings.GEMINI_API_KEY,
                base_url=(
                    settings.EXTRACTION_BASE_URL
                    or "https://generativelanguage.googleapis.com/v1beta/openai/"
                ),
                model=settings.EXTRACTION_MODEL,
            )
        if provider == "hf_inference":
            if not settings.HF_TOKEN:
                raise ValueError("HF_TOKEN is required when EXTRACTION_PROVIDER=hf_inference")
            return _LLMBackend(
                api_key=settings.HF_TOKEN,
                base_url=settings.EXTRACTION_BASE_URL or settings.HF_OPENAI_BASE_URL,
                model=settings.EXTRACTION_MODEL,
            )
        raise ValueError(f"Unsupported extraction provider: {provider}")

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    async def extract_chunk(
        self,
        chunk: dict[str, Any],
        group_id: str,
        chunk_id: str,
        chunk_index: int,
        document_ref: str,
    ) -> dict[str, Any]:
        """Chạy 2 LLM calls song song, trả về factual + relational entries."""
        content = chunk["content"]
        factual, relational = await asyncio.gather(
            self._call_factual_extraction(content, document_ref),
            self._call_relational_extraction(content, document_ref),
        )
        return {
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "chunk_position": chunk_index,
            "content_hash": chunk["content_hash"],
            "factual_entries": factual,
            "relational_entries": relational,
        }

    async def _call_factual_extraction(
        self, content: str, doc_ref: str
    ) -> list[dict[str, Any]]:
        user_msg = f"Document: {doc_ref}\n\nPassage:\n{content}\n\nExtract factual entries from this passage."
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _FACTUAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            entries = data.get("factual_entries", [])
            if not isinstance(entries, list):
                return []
            return [e for e in entries if isinstance(e, dict) and e.get("content")]
        except Exception as exc:
            logger.warning("Factual extraction failed for doc %s: %s", doc_ref, exc)
            return []

    async def _call_relational_extraction(
        self, content: str, doc_ref: str
    ) -> list[dict[str, Any]]:
        user_msg = f"Document: {doc_ref}\n\nPassage:\n{content}\n\nExtract relational entries showing how concepts interact in this passage."
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _RELATIONAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            entries = data.get("relational_entries", [])
            if not isinstance(entries, list):
                return []
            return [e for e in entries if isinstance(e, dict) and e.get("content")]
        except Exception as exc:
            logger.warning("Relational extraction failed for doc %s: %s", doc_ref, exc)
            return []

    # ------------------------------------------------------------------
    # Batch sync (mirrors GraphitiService.sync_chunks)
    # ------------------------------------------------------------------

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
        semaphore = asyncio.Semaphore(max(settings.STRUCTMEM_MAX_CONCURRENCY, 1))

        async def run(index: int, chunk: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            async with semaphore:
                result = await self._sync_chunk_with_retry(
                    chunk=chunk,
                    group_id=normalized_group_id,
                    document_ref=document_ref,
                    chunk_index=index,
                )
                return index, result

        tasks = [asyncio.create_task(run(i, c)) for i, c in enumerate(chunks)]
        ordered_results: list[dict[str, Any] | None] = [None] * len(tasks)

        completed = 0
        try:
            for task in asyncio.as_completed(tasks):
                index, result = await task
                ordered_results[index] = result
                completed += 1
                if completed % 5 == 0 or completed == len(tasks):
                    logger.info(
                        "StructMem extraction progress %s/%s for group %s",
                        completed, len(tasks), normalized_group_id,
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

        return [r for r in ordered_results if r is not None]

    async def _sync_chunk_with_retry(
        self,
        chunk: dict[str, Any],
        group_id: str,
        document_ref: str,
        chunk_index: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        cached = self._load_cached_result(group_id=group_id, chunk_hash=chunk["content_hash"])
        if cached is not None:
            cached["cached"] = True
            cached["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return cached

        last_error: Exception | None = None
        chunk_id = f"{group_id}_chunk_{chunk_index}"

        for attempt in range(settings.STRUCTMEM_CHUNK_RETRIES + 1):
            try:
                result = await asyncio.wait_for(
                    self.extract_chunk(
                        chunk=chunk,
                        group_id=group_id,
                        chunk_id=chunk_id,
                        chunk_index=chunk_index,
                        document_ref=document_ref,
                    ),
                    timeout=settings.STRUCTMEM_CHUNK_TIMEOUT_SECONDS,
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
                    "StructMem extraction failed for chunk %s attempt %s/%s: %s: %s",
                    chunk_id, attempt + 1, settings.STRUCTMEM_CHUNK_RETRIES + 1,
                    type(exc).__name__, exc,
                )
                if attempt < settings.STRUCTMEM_CHUNK_RETRIES:
                    # Exponential backoff — gives Ollama time to finish other requests
                    await asyncio.sleep(min(2 ** (attempt + 2), 30))

        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------
    # Cache (same key format as GraphitiService, different provider_sig suffix)
    # ------------------------------------------------------------------

    def _cache_path(self, group_id: str, chunk_hash: str) -> Path:
        key = sha256(
            f"{group_id}|{chunk_hash}|{self._provider_signature}".encode("utf-8")
        ).hexdigest()
        return self._cache_dir / f"{key}.json"

    def _load_cached_result(self, group_id: str, chunk_hash: str) -> dict[str, Any] | None:
        if not settings.STRUCTMEM_ENABLE_CACHE:
            return None
        path = self._cache_path(group_id=group_id, chunk_hash=chunk_hash)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _store_cached_result(self, group_id: str, chunk_hash: str, payload: dict[str, Any]) -> None:
        if not settings.STRUCTMEM_ENABLE_CACHE:
            return
        path = self._cache_path(group_id=group_id, chunk_hash=chunk_hash)
        path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    @staticmethod
    def normalize_group_id(value: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
        return normalized or "default"
