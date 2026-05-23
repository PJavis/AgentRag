"""SummaryService: generate structured medical summaries from document chunks."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore
from src.agentrag.services.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)

SummaryStyle = Literal["study_note", "clinical", "quick_review"]

_MEDICAL_TEMPLATE_VI = [
    "Định nghĩa & Phân loại",
    "Dịch tễ học",
    "Nguyên nhân & Yếu tố nguy cơ",
    "Sinh lý bệnh",
    "Triệu chứng lâm sàng",
    "Cận lâm sàng & Chẩn đoán",
    "Điều trị",
    "Biến chứng",
    "Tiên lượng & Theo dõi",
]

_OVERVIEW_SYSTEM = """\
You are a medical education expert. Write a concise overview (3-5 sentences) of the document.
LANGUAGE: respond in VIETNAMESE (tiếng Việt) by default — only switch language if the document content is clearly non-Vietnamese.
Highlight key medical terms in **bold** within the overview text.
Return JSON: {"overview": "<text>"}
Return ONLY JSON, no markdown fences.
"""

_SECTION_SYSTEM = """\
You are a medical education expert. Summarize the given document chunks for the specified section heading.
LANGUAGE: respond in VIETNAMESE (tiếng Việt) by default — only switch language if the document content is clearly non-Vietnamese.
Be specific and clinically precise. Wrap key medical/drug/dosage terms in **bold**.

Return JSON with exactly:
{
  "summary": "<2-4 sentence summary, with **bold** key terms>",
  "key_points": ["<point with **bold** terms>", ...],
  "important_terms": [{"term": "<term>", "definition": "<short def>"}, ...]
}
- key_points: 3-6 bullet points, each a complete claim from the content
- important_terms: up to 5 medical/technical terms with definitions
If the content does not cover this section, return {"summary": "", "key_points": [], "important_terms": []}.
Return ONLY JSON, no markdown fences.
"""

_QUICK_SYSTEM = """\
You are a medical education expert. Create a quick-review cheat sheet for the given document.
LANGUAGE: respond in VIETNAMESE (tiếng Việt) by default — only switch language if the document content is clearly non-Vietnamese.
Wrap key terms in **bold** throughout.
Return JSON:
{
  "overview": "<2-3 sentence summary with **bold** key terms>",
  "sections": [
    {
      "heading": "<heading>",
      "summary": "<1-2 sentences with **bold** terms>",
      "key_points": ["<point>", ...]
    }
  ]
}
Return ONLY JSON, no markdown fences.
"""


class SummaryService:
    def __init__(self) -> None:
        self._es = ElasticsearchStore()
        self._llm = LLMGateway()

    async def generate(
        self,
        document_title: str,
        style: SummaryStyle = "study_note",
    ) -> dict[str, Any]:
        if style == "quick_review":
            return await self._quick_review(document_title)

        # study_note and clinical both use medical template sections
        overview, sections = await asyncio.gather(
            self._generate_overview(document_title),
            self._generate_sections(document_title, style),
        )
        return {
            "title": document_title,
            "style": style,
            "overview": overview,
            "sections": sections,
        }

    # ── Overview ──────────────────────────────────────────────────────────────

    async def _generate_overview(self, document_title: str) -> str:
        chunks = await self._es.sparse_search(
            query=document_title, top_k=10, document_title=document_title
        )
        context = self._chunks_to_text(chunks, max_len=2000)
        result, _ = await self._llm.json_response(
            system_prompt=_OVERVIEW_SYSTEM,
            user_prompt=json.dumps(
                {"document_title": document_title, "content": context},
                ensure_ascii=False,
            ),
            task="summary",
        )
        return result.get("overview", "")

    # ── Per-section ───────────────────────────────────────────────────────────

    async def _generate_sections(
        self, document_title: str, style: SummaryStyle
    ) -> list[dict[str, Any]]:
        # Fetch image chunks once — associate with sections by page proximity
        image_chunks = await self._fetch_image_chunks(document_title)

        tasks = [
            self._summarize_section(document_title, heading, image_chunks)
            for heading in _MEDICAL_TEMPLATE_VI
        ]
        results = await asyncio.gather(*tasks)
        # Drop empty sections (document didn't cover that topic)
        return [s for s in results if s.get("summary") or s.get("key_points")]

    async def _summarize_section(
        self,
        document_title: str,
        heading: str,
        image_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        hits = await self._es.sparse_search(
            query=heading,
            top_k=8,
            document_title=document_title,
        )
        # Also include text chunks (exclude image chunks from context)
        text_hits = [h for h in hits if h.get("segment_type", "text") != "image"]
        context = self._chunks_to_text(text_hits, max_len=2000)

        result, _ = await self._llm.json_response(
            system_prompt=_SECTION_SYSTEM,
            user_prompt=json.dumps(
                {
                    "document_title": document_title,
                    "section_heading": heading,
                    "content": context,
                },
                ensure_ascii=False,
            ),
            task="summary",
        )

        # Find image chunks whose page overlaps with this section's text pages
        section_pages = {
            p
            for h in text_hits
            for p in self._page_range(h.get("page_start"), h.get("page_end"))
        }
        images = [
            {
                "url": img.get("metadata", {}).get("image_url", ""),
                "caption": (img.get("content") or "")[:200],
                "page": img.get("page_start"),
            }
            for img in image_chunks
            if img.get("page_start") in section_pages
            and img.get("metadata", {}).get("image_url")
        ]

        return {
            "heading": heading,
            "summary": result.get("summary", ""),
            "key_points": result.get("key_points", []),
            "important_terms": result.get("important_terms", []),
            "images": images,
        }

    # ── Quick review ──────────────────────────────────────────────────────────

    async def _quick_review(self, document_title: str) -> dict[str, Any]:
        chunks = await self._es.sparse_search(
            query=document_title, top_k=30, document_title=document_title
        )
        text_chunks = [c for c in chunks if c.get("segment_type", "text") != "image"]
        context = self._chunks_to_text(text_chunks, max_len=4000)
        result, _ = await self._llm.json_response(
            system_prompt=_QUICK_SYSTEM,
            user_prompt=json.dumps(
                {"document_title": document_title, "content": context},
                ensure_ascii=False,
            ),
            task="summary",
        )
        return {
            "title": document_title,
            "style": "quick_review",
            "overview": result.get("overview", ""),
            "sections": result.get("sections", []),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _fetch_image_chunks(self, document_title: str) -> list[dict[str, Any]]:
        try:
            hits = await self._es.sparse_search(
                query="image diagram anatomy",
                top_k=50,
                document_title=document_title,
            )
            return [h for h in hits if h.get("segment_type") == "image"]
        except Exception:
            return []

    @staticmethod
    def _chunks_to_text(chunks: list[dict[str, Any]], max_len: int = 2000) -> str:
        parts: list[str] = []
        total = 0
        for c in chunks:
            text = (c.get("content") or "").strip()
            if not text:
                continue
            section = c.get("section_path", "")
            piece = f"[{section}]\n{text}" if section else text
            if total + len(piece) > max_len:
                break
            parts.append(piece)
            total += len(piece)
        return "\n\n".join(parts)

    @staticmethod
    def _page_range(start: int | None, end: int | None) -> set[int]:
        if start is None:
            return set()
        return set(range(start, (end or start) + 1))
