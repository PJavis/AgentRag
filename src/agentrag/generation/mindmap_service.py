"""MindmapService: generate a Mermaid mindmap from document chunks via LLM."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

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


class MindmapService:
    # Simple in-process TTL cache: key → (timestamp, result)
    _cache: dict[str, tuple[float, dict[str, Any]]] = {}
    _TTL = 86400.0  # 24 hours

    def __init__(self) -> None:
        self._es = ElasticsearchStore()
        self._llm = LLMGateway()

    async def generate(
        self,
        document_title: str,
        focus_topic: str | None = None,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        cache_key = f"{document_title}|{focus_topic or ''}|{max_depth}"
        cached = self._cache.get(cache_key)
        if cached and (time.time() - cached[0]) < self._TTL:
            return {**cached[1], "cached": True}

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
        self._cache[cache_key] = (time.time(), output)
        return output

    def invalidate(self, document_title: str) -> None:
        self._cache = {
            k: v for k, v in self._cache.items() if not k.startswith(document_title)
        }

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
