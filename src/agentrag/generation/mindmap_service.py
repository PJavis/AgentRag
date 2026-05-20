"""MindmapService: generate a Mermaid mindmap from document chunks via LLM."""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from src.agentrag.config import settings
from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore
from src.agentrag.services.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)

_MINDMAP_SYSTEM = """\
You are an educational content organizer. Given document chunks, extract a hierarchical concept map.

Rules:
- Central node: document title (wrap in (( )) in Mermaid)
- Max depth as specified
- Use the document's original language (Vietnamese if Vietnamese source)
- Nodes should be short concept names, not full sentences
- Mermaid mindmap format:
  mindmap
    root((Title))
      Branch
        Leaf

Return valid JSON with exactly these keys:
{
  "mermaid": "<the complete mermaid mindmap string>",
  "concepts": [{"name": "<concept>", "parent": "<parent name or null>", "level": <1|2|3>}]
}
Return ONLY the JSON object, no markdown fences.
"""

_CACHE_PREFIX = "agentrag:mindmap:v1:"
_CACHE_TTL_SECONDS = 86400  # 24h


class MindmapService:
    """Valkey-backed cache survives restarts and is shared across workers."""

    _valkey: Redis | None = None
    _valkey_disabled: bool = False

    def __init__(self) -> None:
        self._es = ElasticsearchStore()
        self._llm = LLMGateway()

    @classmethod
    def _client(cls) -> Redis | None:
        if cls._valkey_disabled:
            return None
        if cls._valkey is None and settings.REDIS_URL:
            cls._valkey = Redis.from_url(settings.REDIS_URL, decode_responses=True)
        return cls._valkey

    @staticmethod
    def _cache_key(document_title: str, focus_topic: str | None, max_depth: int) -> str:
        raw = f"{document_title}|{focus_topic or ''}|{max_depth}".encode("utf-8")
        return _CACHE_PREFIX + hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _doc_index_key(document_title: str) -> str:
        digest = hashlib.sha256(document_title.encode("utf-8")).hexdigest()
        return f"{_CACHE_PREFIX}doc:{digest}"

    async def _cache_get(self, key: str) -> dict[str, Any] | None:
        client = self._client()
        if client is None:
            return None
        try:
            raw = await client.get(key)
        except RedisError:
            type(self)._valkey_disabled = True
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    async def _cache_set(self, key: str, value: dict[str, Any], doc_index_key: str) -> None:
        client = self._client()
        if client is None:
            return
        try:
            pipe = client.pipeline()
            pipe.set(key, json.dumps(value, ensure_ascii=False), ex=_CACHE_TTL_SECONDS)
            pipe.sadd(doc_index_key, key)
            pipe.expire(doc_index_key, _CACHE_TTL_SECONDS)
            await pipe.execute()
        except RedisError:
            type(self)._valkey_disabled = True

    async def generate(
        self,
        document_title: str,
        focus_topic: str | None = None,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        cache_key = self._cache_key(document_title, focus_topic, max_depth)
        cached = await self._cache_get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}

        chunks = await self._fetch_chunks(document_title, focus_topic)
        if not chunks:
            return {
                "mermaid": f"mindmap\n  root(({document_title}))\n    Không tìm thấy nội dung",
                "concepts": [],
                "cached": False,
            }

        context = self._build_context(chunks, max_chunks=30)
        user_prompt = json.dumps(
            {
                "document_title": document_title,
                "focus_topic": focus_topic,
                "max_depth": max_depth,
                "chunks": context,
            },
            ensure_ascii=False,
        )

        result, _ = await self._llm.json_response(
            system_prompt=_MINDMAP_SYSTEM,
            user_prompt=user_prompt,
            task="mindmap",
        )

        output: dict[str, Any] = {
            "mermaid": result.get("mermaid", f"mindmap\n  root(({document_title}))"),
            "concepts": result.get("concepts", []),
            "cached": False,
        }
        await self._cache_set(cache_key, output, self._doc_index_key(document_title))
        return output

    async def invalidate(self, document_title: str) -> None:
        client = self._client()
        if client is None:
            return
        index_key = self._doc_index_key(document_title)
        try:
            members = await client.smembers(index_key)
            if members:
                await client.delete(*members)
            await client.delete(index_key)
        except RedisError:
            type(self)._valkey_disabled = True

    async def _fetch_chunks(
        self, document_title: str, focus_topic: str | None
    ) -> list[dict[str, Any]]:
        query = focus_topic or document_title
        hits = await self._es.sparse_search(
            query=query,
            top_k=30,
            document_title=document_title,
        )
        return hits

    @staticmethod
    def _build_context(chunks: list[dict[str, Any]], max_chunks: int = 30) -> list[dict]:
        return [
            {
                "section": c.get("section_path", ""),
                "content": (c.get("content") or "")[:600],
            }
            for c in chunks[:max_chunks]
        ]
