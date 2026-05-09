"""ImageParser: describe image files via a vision LLM.

Used for two scenarios:
  1. Standalone image files (.jpg, .png, .webp, …) uploaded directly.
  2. Images extracted from PDFs by PDFParser.extract_images().

The parser calls LLMGateway.vision_response() with a medical-education
system prompt and returns a text description suitable for embedding.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agentrag.config import settings

if TYPE_CHECKING:
    from src.agentrag.services.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)

_SAFE_DIRNAME_RE = re.compile(r"[^\w\-]")

_MIME_MAP: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}

_MEDICAL_VISION_SYSTEM = """\
You are a medical image analyst creating educational content for medical students.
Describe this image with clinical precision:

1. Image type (anatomy diagram, X-ray, CT scan, MRI, histology slide, ECG, ultrasound, chart, etc.)
2. Anatomical structures or regions visible — use correct anatomical/medical terminology
3. Labels, arrows, annotations, and what they point to
4. Key educational features and findings highlighted
5. Any pathological findings if present (abnormalities, lesions, fractures, etc.)
6. Orientation or view (e.g., anteroposterior, lateral, cross-section, sagittal)

Write a clear, factual description in 3-6 sentences. Use standard medical terminology.
If this is not a medical image, describe what it shows accurately.
"""


class ImageParser:
    """Parse an image file into a searchable text description using a vision LLM."""

    def __init__(self, llm_gateway: LLMGateway) -> None:
        self._llm = llm_gateway

    async def parse(
        self,
        file_path: str,
        document_title: str | None = None,
    ) -> dict[str, Any]:
        """Parse a standalone image file.

        Returns:
          {
            "parsed_content": str,   # vision LLM description
            "pages": 1,
            "image_url": str,        # local URL for serving via /images/...
            "image_path": str,       # filesystem path where image is saved
          }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")

        mime = _MIME_MAP.get(path.suffix.lower(), "image/jpeg")
        img_bytes = path.read_bytes()

        # Save to IMAGE_STORAGE_DIR under document_title subfolder
        safe_dir = _SAFE_DIRNAME_RE.sub("_", document_title or "standalone")[:80]
        out_dir = Path(settings.IMAGE_STORAGE_DIR) / safe_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / path.name
        if not dest.exists():
            dest.write_bytes(img_bytes)

        image_url = f"/images/{safe_dir}/{path.name}"
        description = await self._describe(img_bytes, mime)

        return {
            "parsed_content": description,
            "pages": 1,
            "image_url": image_url,
            "image_path": str(dest),
        }

    async def describe(
        self,
        image_bytes: bytes,
        mime: str = "image/jpeg",
        context: str = "",
    ) -> str:
        """Describe image bytes directly (used for PDF-extracted images)."""
        return await self._describe(image_bytes, mime, context)

    async def _describe(
        self,
        image_bytes: bytes,
        mime: str,
        context: str = "",
    ) -> str:
        text_prompt = (
            f"Document context: {context}\n\nDescribe this image."
            if context
            else "Describe this image."
        )
        try:
            description, _latency = await self._llm.vision_response(
                system_prompt=_MEDICAL_VISION_SYSTEM,
                text_prompt=text_prompt,
                image_bytes=image_bytes,
                mime_type=mime,
                task="vision",
            )
            return description or "[image — no description generated]"
        except Exception as exc:
            logger.warning("Vision description failed: %s", exc)
            return "[image — vision LLM unavailable]"
