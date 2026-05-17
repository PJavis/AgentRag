"""PDFParser hybrid OCR fallback — smoke tests."""
from __future__ import annotations

import io

import pytest

from src.agentrag.ingestion.parsers import pdf_parser as P


def test_tesseract_ocr_helper_returns_empty_on_garbage_bytes():
    """Tesseract on non-image bytes — must not raise, returns empty."""
    out = P._ocr_tesseract(b"not an image", lang="eng")
    assert out == ""


def test_tesseract_ocr_helper_reads_simple_png():
    """Render a tiny PIL image with text, OCR it, expect substring match."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (320, 80), color="white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    d.text((10, 20), "HELLO 12345", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    text = P._ocr_tesseract(buf.getvalue(), lang="eng")
    assert "HELLO" in text.upper() or "12345" in text


def test_vision_fallback_disabled_when_no_provider(monkeypatch):
    from src.agentrag.config import settings
    monkeypatch.setattr(settings, "VISION_PROVIDER", None)
    out = P._ocr_via_vision_llm(b"\x89PNG\r\n\x1a\n")
    assert out == ""
