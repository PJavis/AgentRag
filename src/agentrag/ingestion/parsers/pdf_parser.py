"""PDFParser: page-aware PDF parsing using PyMuPDF + Tesseract/Vision OCR fallback.

Per page:
  1. PyMuPDF.get_text("text", sort=True). If ≥ PDF_OCR_MIN_TEXT_CHARS → use as-is.
  2. Else render page → Tesseract (lang=PDF_OCR_LANG). If ≥ PDF_OCR_VISION_THRESHOLD → use.
  3. Else, if PDF_OCR_VISION_FALLBACK on, send page image through VISION_PROVIDER.
  4. Marker `\x00P{N}\x00` survives the chunker and is resolved into
     page_start / page_end during chunking (NotebookLM-style citations).
"""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any

from src.agentrag.config import settings

logger = logging.getLogger(__name__)

_SAFE_DIRNAME_RE = re.compile(r"[^\w\-]")

PAGE_MARKER_PREFIX = "\x00P"
PAGE_MARKER_SUFFIX = "\x00"


def make_page_marker(page_num: int) -> str:
    return f"{PAGE_MARKER_PREFIX}{page_num}{PAGE_MARKER_SUFFIX}"


def _ocr_tesseract(image_bytes: bytes, lang: str) -> str:
    """OCR via Tesseract. Returns '' on failure (never raises)."""
    try:
        import pytesseract
        from PIL import Image
    except Exception as exc:
        logger.warning("Tesseract OCR libs missing: %s", exc)
        return ""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return (pytesseract.image_to_string(img, lang=lang) or "").strip()
    except Exception as exc:
        logger.warning("Tesseract OCR failed: %s", exc)
        return ""


def _append_table_markdown(page, text: str) -> str:
    """Probe arm B: append *structurally sound* tables as GFM markdown.

    Additive — flat text is preserved and the markdown is appended, so retrieval
    sees both. Tables failing `is_safe_to_markdown` are skipped: on this corpus
    77% of find_tables() detections are layout boxes whose markdown (mirrored
    cells, invented headers, one row per visual line) is worse than the flat text
    it would replace. Converting them would make arm B lose for reasons unrelated
    to tables. Any detection error leaves the text unchanged.

    Rendering goes through `render_markdown`, never `table.to_markdown()`: the
    gate judges the `extract()` cells, so the emitted text must come from those
    same cells. `to_markdown()` invents `ColN` headers and mirrors a single
    populated cell across columns — corruption the gate cannot see because it
    does not exist until that renderer runs. Blocks are packed to
    `SEARCH_CHUNK_MAX_TOKENS` and each repeats the header, so the chunker never
    has to cut inside a row.
    """
    from src.agentrag.config import settings
    from src.agentrag.ingestion.parsers.table_quality import render_markdown

    try:
        tables = list(page.find_tables().tables)
    except Exception:  # noqa: BLE001 — no table layer on this page
        return text

    blocks = []
    for table in tables:
        try:
            md = render_markdown(
                table.extract(), max_tokens=settings.SEARCH_CHUNK_MAX_TOKENS
            )
        except Exception:  # noqa: BLE001 — a table PyMuPDF cannot read
            continue
        if md.strip():
            blocks.append(md.strip())

    if not blocks:
        return text
    return text + "\n\n" + "\n\n".join(blocks)


class PDFParser:
    """Page-aware PDF parser with hybrid Tesseract + Vision LLM OCR fallback."""

    def parse(self, file_path: str) -> dict[str, Any]:
        """Per-page text extraction with OCR fallback.

        Returns:
          {
            "parsed_content": str,      # full text with embedded page markers
            "pages": int,               # total page count
            "page_data": list[dict],    # [{"page_num": N, "text": "...", "source": "text|ocr|vision"}]
          }
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError("PyMuPDF required. pip install pymupdf") from exc

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc = fitz.open(str(path))
        ocr_enabled = settings.PDF_OCR_FALLBACK_ENABLED
        ocr_min = settings.PDF_OCR_MIN_TEXT_CHARS
        ocr_dpi = settings.PDF_OCR_DPI
        ocr_lang = settings.PDF_OCR_LANG
        vision_fallback = settings.PDF_OCR_VISION_FALLBACK
        vision_threshold = settings.PDF_OCR_VISION_THRESHOLD

        parts: list[str] = []
        page_data: list[dict[str, Any]] = []

        for page_num, page in enumerate(doc, start=1):
            source = "text"
            text = page.get_text("text", sort=True)
            stripped = text.strip()

            if ocr_enabled and len(stripped) < ocr_min:
                # Render page to image (PNG bytes) for OCR
                try:
                    scale = ocr_dpi / 72.0
                    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                    img_bytes = pix.tobytes("png")
                except Exception as exc:
                    logger.warning("PDF page %d render failed: %s", page_num, exc)
                    img_bytes = b""

                if img_bytes:
                    ocr_text = _ocr_tesseract(img_bytes, lang=ocr_lang)
                    if len(ocr_text) >= vision_threshold:
                        text = ocr_text
                        source = "ocr"
                    elif vision_fallback:
                        vision_text = _ocr_via_vision_llm(img_bytes)
                        if len(vision_text) >= vision_threshold:
                            text = vision_text
                            source = "vision"
                        elif ocr_text:
                            text = ocr_text
                            source = "ocr"
                    elif ocr_text:
                        text = ocr_text
                        source = "ocr"
                    stripped = text.strip()

            # Arm B, AFTER the OCR block on purpose: appending pipes and a
            # header to a thin page would push it past PDF_OCR_MIN_TEXT_CHARS and
            # skip the OCR fallback arm A takes — arm B would then lose content
            # for reasons unrelated to tables.
            if settings.PDF_PRESERVE_TABLES:
                text = _append_table_markdown(page, text)

            if not stripped:
                continue
            page_data.append({"page_num": page_num, "text": text, "source": source})
            parts.append(f"{make_page_marker(page_num)}\n{text}")

        doc.close()
        full_content = "\n\n".join(parts)
        summary = {
            s: sum(1 for p in page_data if p["source"] == s)
            for s in ("text", "ocr", "vision")
        }
        logger.info(
            "PDFParser: %d pages from %s — sources=%s",
            len(page_data),
            path.name,
            summary,
        )
        return {
            "parsed_content": full_content,
            "pages": max(len(page_data), 1),
            "page_data": page_data,
        }

    def extract_images(
        self,
        file_path: str,
        document_title: str,
    ) -> list[dict[str, Any]]:
        """Extract raster images from a PDF.

        Returns a list of dicts:
          {"page": int, "path": str, "url": str, "bytes": bytes, "mime": str}

        Small images (icons, bullet glyphs) are filtered by IMAGE_MIN_SIZE_BYTES.
        """
        try:
            import fitz
        except ImportError as exc:
            raise ImportError("PyMuPDF required") from exc

        path = Path(file_path)
        safe_dir = _SAFE_DIRNAME_RE.sub("_", document_title)[:80]
        out_dir = Path(settings.IMAGE_STORAGE_DIR) / safe_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(path))
        images: list[dict[str, Any]] = []
        seen_xrefs: set[int] = set()
        # Perceptual-hash dedup: same diagram repeated across slides should
        # only be embedded + described once. We use a cheap content-hash
        # (SHA1 of raw jpeg bytes) as a first pass — handles identical bytes.
        # For visually-near-duplicates a true pHash would help, but adds an
        # extra Pillow dep. Bytes-hash catches the common slide-deck case.
        import hashlib as _hashlib
        seen_byte_hashes: set[str] = set()

        for page_num, page in enumerate(doc, start=1):
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_bytes = pix.tobytes("jpeg")
                    if len(img_bytes) < settings.IMAGE_MIN_SIZE_BYTES:
                        continue
                    byte_hash = _hashlib.sha1(img_bytes).hexdigest()
                    if byte_hash in seen_byte_hashes:
                        logger.debug(
                            "PDFParser: skip duplicate image hash=%s p%d xref=%d",
                            byte_hash[:8], page_num, xref,
                        )
                        continue
                    seen_byte_hashes.add(byte_hash)
                    filename = f"p{page_num}_{img_idx}.jpg"
                    save_path = out_dir / filename
                    save_path.write_bytes(img_bytes)
                    url = f"/images/{safe_dir}/{filename}"
                    images.append({
                        "page": page_num,
                        "path": str(save_path),
                        "url": url,
                        "bytes": img_bytes,
                        "mime": "image/jpeg",
                        "byte_hash": byte_hash,
                    })
                except Exception as exc:
                    logger.debug("Skipping image xref=%d p%d: %s", xref, page_num, exc)

        doc.close()
        logger.debug("PDFParser: extracted %d images from %s", len(images), path.name)
        return images


def _ocr_via_vision_llm(image_bytes: bytes) -> str:
    """Vision LLM as OCR fallback. Reuses VISION_PROVIDER; returns '' on
    fail/disabled. PDFParser.parse is sync but may be called from an async
    context (FastAPI / ARQ worker) — running the vision coroutine in a
    fresh thread + its own event loop avoids 'event loop is already
    running'."""
    if not settings.VISION_PROVIDER:
        return ""
    try:
        import asyncio
        import concurrent.futures

        from src.agentrag.services.container import get_container
        vision = get_container().vision
        if not vision.enabled:
            return ""
        prompt_ctx = (
            "Transcribe ALL Vietnamese / English text visible on this page "
            "verbatim. Preserve line breaks and bullet structure. Do NOT "
            "summarise, do NOT translate, do NOT describe the image — only "
            "transcribe text exactly as it appears. If no readable text, "
            "return an empty string."
        )

        def _runner() -> str:
            return asyncio.run(
                vision.describe(image_bytes, mime="image/png", context=prompt_ctx)
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            text = ex.submit(_runner).result(timeout=120)
        return (text or "").strip()
    except Exception as exc:
        logger.warning("Vision OCR fallback failed: %s", exc)
        return ""
