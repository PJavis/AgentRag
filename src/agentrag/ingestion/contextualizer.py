from __future__ import annotations

import asyncio
import logging
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agentrag.config import settings

if TYPE_CHECKING:
    from src.agentrag.services.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You situate a short passage within its source document for retrieval.
You are given the whole document, then one passage from it.
Write a single concise sentence (max ~80 tokens, same language as the passage)
that states what section/topic the passage belongs to and what it is about, so
the passage can be found by search even out of context. Output ONLY that
sentence, no preamble, no quotes."""


class Contextualizer:
    """WS1 — generate a situating context sentence per chunk.

    The whole document goes in the system prompt (a stable prefix that the
    provider's context cache, e.g. DeepSeek, reuses across the document's
    chunks); only the per-chunk passage varies. Results are file-cached keyed
    by (provider_signature, doc_hash, chunk_hash) so backfill is idempotent.
    """

    def __init__(self, gateway: "LLMGateway", cache_dir: str | None = None) -> None:
        self._gateway = gateway
        self._cache_dir = Path(cache_dir or settings.CONTEXTUAL_CACHE_DIR)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._sig = sha256(
            f"contextual_v1|{settings.CONTEXTUAL_RETRIEVAL_TASK}".encode("utf-8")
        ).hexdigest()[:12]

    async def contextualize_chunks(
        self, doc_text: str, chunks: list[dict[str, Any]], document_title: str
    ) -> list[dict[str, Any]]:
        if not chunks:
            return chunks
        doc_clip = doc_text[: settings.CONTEXTUAL_MAX_DOC_CHARS]
        doc_hash = sha256(doc_clip.encode("utf-8")).hexdigest()
        system = f"{_SYSTEM_PROMPT}\n\n<document title=\"{document_title}\">\n{doc_clip}\n</document>"

        sem = asyncio.Semaphore(max(settings.EMBEDDING_BATCH_SIZE // 4, 4))

        async def one(chunk: dict[str, Any]) -> None:
            chunk_hash = chunk.get("content_hash") or sha256(
                chunk["content"].encode("utf-8")
            ).hexdigest()
            cached = self._load(doc_hash, chunk_hash)
            if cached is not None:
                chunk["context_text"] = cached
                return
            async with sem:
                try:
                    text = await self._gateway.text_response(
                        system_prompt=system,
                        user_prompt=f"Passage:\n{chunk['content']}",
                        task=settings.CONTEXTUAL_RETRIEVAL_TASK,
                    )
                except Exception as exc:
                    logger.warning("contextualize failed (%s): %s", document_title, exc)
                    text = ""
            text = (text or "").strip()
            chunk["context_text"] = text or None
            if text:
                self._store(doc_hash, chunk_hash, text)

        await asyncio.gather(*(one(c) for c in chunks))
        return chunks

    def _path(self, doc_hash: str, chunk_hash: str) -> Path:
        key = sha256(f"{self._sig}|{doc_hash}|{chunk_hash}".encode("utf-8")).hexdigest()
        return self._cache_dir / f"{key}.txt"

    def _load(self, doc_hash: str, chunk_hash: str) -> str | None:
        try:
            p = self._path(doc_hash, chunk_hash)
            if not p.exists():
                return None
            return p.read_text(encoding="utf-8")
        except Exception:
            return None

    def _store(self, doc_hash: str, chunk_hash: str, text: str) -> None:
        try:
            self._path(doc_hash, chunk_hash).write_text(text, encoding="utf-8")
        except Exception:
            pass
