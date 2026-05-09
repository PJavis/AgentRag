"""PDFParser: page-aware PDF parsing using PyMuPDF.

Extracts text page by page and embeds thin page markers into the content
string. Markers (\x00P{N}\x00) are invisible to the chunker's heading/token
logic but resolved in HybridChunker's post-processing step to assign
page_start / page_end to each chunk — enabling NotebookLM-style citations.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.agentrag.config import settings

logger = logging.getLogger(__name__)

_SAFE_DIRNAME_RE = re.compile(r"[^\w\-]")

# Non-printable NUL-bounded markers that survive the chunker unchanged.
PAGE_MARKER_PREFIX = "\x00P"
PAGE_MARKER_SUFFIX = "\x00"


def make_page_marker(page_num: int) -> str:
    return f"{PAGE_MARKER_PREFIX}{page_num}{PAGE_MARKER_SUFFIX}"


class PDFParser:
    """Page-aware PDF parser using PyMuPDF (fitz).

    Unlike MarkItDownParser, this parser preserves per-page boundaries so
    that downstream chunks can be tagged with their source page number.
    """

    def parse(self, file_path: str) -> dict[str, Any]:
        """
        Returns:
          {
            "parsed_content": str,      # full text with embedded page markers
            "pages": int,               # total page count
            "page_data": list[dict],    # [{"page_num": N, "text": "..."}]
          }
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for PDFParser. "
                "Install it with: pip install pymupdf"
            ) from exc

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc = fitz.open(str(path))
        parts: list[str] = []
        page_data: list[dict[str, Any]] = []

        for page_num, page in enumerate(doc, start=1):
            # sort=True → reading order (left-to-right, top-to-bottom)
            text = page.get_text("text", sort=True)
            if not text.strip():
                continue
            page_data.append({"page_num": page_num, "text": text})
            # Embed marker at the very start of each page's text block
            parts.append(f"{make_page_marker(page_num)}\n{text}")

        doc.close()

        full_content = "\n\n".join(parts)
        logger.debug("PDFParser: %d pages from %s", len(page_data), path.name)

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

        for page_num, page in enumerate(doc, start=1):
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    pix = fitz.Pixmap(doc, xref)
                    # Convert CMYK / alpha-only to RGB
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    # Skip tiny images (icons, decorations)
                    img_bytes = pix.tobytes("jpeg")
                    if len(img_bytes) < settings.IMAGE_MIN_SIZE_BYTES:
                        continue
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
                    })
                except Exception as exc:
                    logger.debug("Skipping image xref=%d p%d: %s", xref, page_num, exc)

        doc.close()
        logger.debug("PDFParser: extracted %d images from %s", len(images), path.name)
        return images
