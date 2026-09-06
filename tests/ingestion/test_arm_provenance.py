"""Segments must record which probe arm produced them.

Without this stamp, "ship arm B and measure it on production traffic" cannot be
done: an answer's cited segments could not be attributed to an arm after the
fact. The metric would be assertable, not measurable.
"""
from src.agentrag.config import settings
from src.agentrag.ingestion.pipeline import build_chunk_metadata


def test_pdf_segments_record_the_arm_when_the_flag_is_off(monkeypatch):
    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", False)
    meta = build_chunk_metadata("pdf", "doc.pdf")
    assert meta["pdf_preserve_tables"] is False
    assert meta["document_title"] == "doc.pdf"


def test_pdf_segments_record_the_arm_when_the_flag_is_on(monkeypatch):
    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", True)
    assert build_chunk_metadata("pdf", "doc.pdf")["pdf_preserve_tables"] is True


def test_non_pdf_segments_are_not_stamped(monkeypatch):
    """The flag only governs the PDF parser; stamping other sources would claim
    provenance the flag never had."""
    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", True)
    for source in ("markdown", "excel", "audio", "image", "word"):
        assert "pdf_preserve_tables" not in build_chunk_metadata(source, "x")
