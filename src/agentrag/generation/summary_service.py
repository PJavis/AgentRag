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
You are a medical education expert writing the OPENING SECTION of a study
guide. Write a flowing, natural-language intro (5-8 sentences, ONE paragraph)
that:
  1. Names the document topic and clinical context (1-2 sentences).
  2. Explains WHY this matters in practice — epidemiology, prevalence,
     impact on patient care, or learning objectives the handout covers.
  3. Briefly previews what the reader will learn — definition, classification,
     causes, diagnosis, treatment, prognosis — but written as prose, NOT a list.
  4. Ends with a sentence that bridges naturally into the detailed sections
     below (e.g. "Phần dưới đây trình bày chi tiết từng khía cạnh...").

LANGUAGE: respond in VIETNAMESE (tiếng Việt) by default — only switch
language if the document content is clearly non-Vietnamese.

STYLE: conversational and natural, like an attending physician explaining
to a junior. Do NOT start with "Đây là tóm tắt..." or "Tài liệu này nói về...".
Open with a real fact — a statistic, a key definition, or the central clinical
problem. Wrap key medical terms in **bold** but do NOT over-bold.

Return JSON: {"overview": "<text>"}
Return ONLY JSON, no markdown fences.
"""

_SECTION_SYSTEM = """\
You are a medical education expert. Summarize the given document chunks for the specified section heading.
LANGUAGE: respond in VIETNAMESE (tiếng Việt) by default — only switch language if the document content is clearly non-Vietnamese.
Be specific, comprehensive, and clinically precise. Cover EVERY relevant
detail from the content — numbers, percentages, criteria, mechanisms,
sub-classifications, exceptions. Do NOT summarize so aggressively that key
clinical details disappear. Prefer nested sub-bullets over flattening.

STYLE — natural narrative, NOT robotic:
- The `summary` field is PROSE (4-8 sentences flowing together), not a list.
- Open with a concrete fact, definition, mechanism, or statistic — NOT with
  the section name itself. BAD: "Định nghĩa là...". GOOD: "Theo WHO, hiếm
  muộn là tình trạng...".
- Use connectors (do đó, ngoài ra, tuy nhiên, mặt khác) to link sentences.
- Mix sentence lengths for readability.

AGGRESSIVELY wrap key medical terms, drug names, dosages, lab values,
anatomical structures, diagnoses, and red-flag warnings in **bold** within
the summary text and key_points. Wrap statistics ($15\\%$), thresholds
($< 15 \\text{ triệu/ml}$), and ranges in inline LaTeX ($...$). Do NOT
output a separate term/definition list.

Return JSON with exactly:
{
  "summary": "<4-8 sentence flowing prose, every key medical term **bolded**>",
  "key_points": ["<point with terms **bolded** and stats in $LaTeX$>", ...]
}
- key_points: 6-12 bullet points, each a complete claim from the content.
- Use nested bullets (sub-bullets indented 2 spaces with `-`) for hierarchical info.
- Use Markdown tables `| col | col |` when the content has structured comparisons.
- Use `> blockquote` ONLY for safety warnings / contraindications.
- Each bullet should have at least 1 **bold** term.
If the content does not cover this section, return {"summary": "", "key_points": []}.
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


_BATCH_SYSTEM = """\
You are a medical education expert building a DETAILED study guide from a long
document, one contiguous span at a time. Summarize the given chunks (a span of
consecutive pages) THOROUGHLY — this is one part of a whole-document summary, so
do NOT compress to a few lines.

LANGUAGE: respond in VIETNAMESE (tiếng Việt) unless the content is clearly
non-Vietnamese.

REQUIREMENTS:
- Infer a concise section heading naming the topic(s) of THIS span (e.g. the
  specific disorder, procedure, or theme covered). If the span covers a single
  named disorder/topic, use its name.
- Cover EVERY clinically relevant detail in the span: definitions,
  classifications, epidemiology, causes/risk factors, mechanisms, clinical
  features, diagnostic criteria (ICD/DSM codes, thresholds, durations),
  investigations, treatment (drugs, doses, lines of therapy), complications,
  prognosis, follow-up. Keep numbers, percentages, criteria, drug names, doses.
- Prefer nested bullets over flattening. Bold key terms with **term**.
- Do NOT invent content not present in the span. Summarize only what is given.

Return JSON:
{"heading": "<topic of this span>",
 "summary": "<detailed multi-paragraph markdown>",
 "key_points": ["<important takeaway>", ...]}
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
            top_k=15,
            document_title=document_title,
        )
        # Also include text chunks (exclude image chunks from context)
        text_hits = [h for h in hits if h.get("segment_type", "text") != "image"]
        context = self._chunks_to_text(text_hits, max_len=5000)

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
            # important_terms still parsed for back-compat, but consumers
            # render inline-bolded keywords instead of a separate term list.
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

    # ── Whole-document map-reduce ───────────────────────────────────────────────

    async def generate_full(
        self,
        document_title: str,
        on_progress: Any = None,
        char_budget: int = 14000,
        max_batches: int = 24,
        concurrency: int = 8,
    ) -> dict[str, Any]:
        """Map-reduce summary covering the ENTIRE document.

        Fetches every text segment, batches them into contiguous page spans,
        summarizes each span in detail (parallel), and returns ordered sections.
        Scales to arbitrarily long docs — unlike top-K retrieval, it sees 100%
        of the content. `on_progress(done, total)` is awaited per finished batch.
        """
        chunks = await self._es.fetch_all_segments(document_title)
        if not chunks:
            return {"title": document_title, "style": "full", "overview": "", "sections": []}

        batches = self._batch_chunks(chunks, char_budget=char_budget, max_batches=max_batches)
        total = len(batches)
        sem = asyncio.Semaphore(max(1, concurrency))
        done = 0
        lock = asyncio.Lock()

        async def run(batch: list[dict[str, Any]]) -> dict[str, Any]:
            nonlocal done
            async with sem:
                res = await self._summarize_batch(document_title, batch)
            if on_progress is not None:
                async with lock:
                    done += 1
                    try:
                        await on_progress(done, total)
                    except Exception:  # noqa: BLE001
                        pass
            return res

        results = await asyncio.gather(*(run(b) for b in batches))
        sections = [r for r in results if r and (r.get("summary") or r.get("key_points"))]
        return {
            "title": document_title,
            "style": "full",
            "overview": "",
            "sections": sections,
            "batch_count": total,
        }

    @staticmethod
    def _batch_chunks(
        chunks: list[dict[str, Any]], char_budget: int, max_batches: int
    ) -> list[list[dict[str, Any]]]:
        """Group page-ordered chunks into contiguous spans ~char_budget each.

        If the doc is huge, widen the budget so batch count stays ≤ max_batches
        (bounds LLM calls/cost while still covering everything)."""
        total_chars = sum(len((c.get("content") or "")) for c in chunks)
        budget = max(char_budget, (total_chars // max_batches) + 1)
        batches: list[list[dict[str, Any]]] = []
        cur: list[dict[str, Any]] = []
        cur_len = 0
        for c in chunks:
            piece = len((c.get("content") or ""))
            if cur and cur_len + piece > budget:
                batches.append(cur)
                cur, cur_len = [], 0
            cur.append(c)
            cur_len += piece
        if cur:
            batches.append(cur)
        return batches

    async def _summarize_batch(
        self, document_title: str, batch: list[dict[str, Any]]
    ) -> dict[str, Any]:
        pages = [c.get("page_start") for c in batch if c.get("page_start") is not None]
        page_start = min(pages) if pages else None
        page_end = max(
            [c.get("page_end") or c.get("page_start") for c in batch
             if (c.get("page_end") or c.get("page_start")) is not None],
            default=page_start,
        )
        context = "\n\n".join((c.get("content") or "").strip() for c in batch if (c.get("content") or "").strip())
        try:
            result, _ = await self._llm.json_response(
                system_prompt=_BATCH_SYSTEM,
                user_prompt=json.dumps(
                    {
                        "document_title": document_title,
                        "page_range": f"{page_start}-{page_end}" if page_start else "",
                        "content": context[:24000],
                    },
                    ensure_ascii=False,
                ),
                # Batch map = many parallel calls → use the FAST model (flash);
                # pro is too slow/costly at 10-40 batches. Quality is fine here
                # (condensing given text, not deep reasoning).
                task="answer",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("summary batch (pages %s-%s) failed: %s", page_start, page_end, exc)
            return {}
        return {
            "heading": result.get("heading", ""),
            "page_start": page_start,
            "page_end": page_end,
            "summary": result.get("summary", ""),
            "key_points": result.get("key_points", []),
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
