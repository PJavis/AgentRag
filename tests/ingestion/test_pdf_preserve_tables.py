from unittest.mock import MagicMock, patch

from src.agentrag.ingestion.parsers.pdf_parser import PDFParser, _append_table_markdown

GRID = [["STT", "Liều"], ["1", "10mg"], ["2", "20mg"]]
# One populated cell per row: PyMuPDF reports 2 columns, extract() fills one.
# `to_markdown()` is what mirrors the text and invents "Col2".
SINGLE_COL = [["1. ĐỊNH NGHĨA", None], ["Lo lắng là phản ứng", None], ["của cơ thể", None]]


def _page(tables):
    tabs = []
    for rows in tables:
        t = MagicMock()
        t.extract.return_value = rows
        tabs.append(t)
    page = MagicMock()
    page.find_tables.return_value = MagicMock(tables=tabs)
    return page


def test_appends_markdown_for_a_real_grid():
    out = _append_table_markdown(_page([GRID]), "flat text")
    assert out.startswith("flat text")
    assert "| STT | Liều |" in out


def test_never_calls_pymupdf_to_markdown():
    """The gate judges extract(); emitting to_markdown() would emit ungated text."""
    page = _page([GRID])
    _append_table_markdown(page, "flat text")
    for t in page.find_tables.return_value.tables:
        t.to_markdown.assert_not_called()


def test_skips_layout_artifacts():
    """Converting these would corrupt the page and poison the A/B."""
    assert _append_table_markdown(_page([SINGLE_COL]), "flat text") == "flat text"


def test_mixed_page_keeps_only_the_safe_table():
    out = _append_table_markdown(_page([GRID, SINGLE_COL]), "flat text")
    assert "| STT | Liều |" in out
    assert "ĐỊNH NGHĨA" not in out


def test_noop_when_no_tables():
    assert _append_table_markdown(_page([]), "flat text") == "flat text"


def test_survives_find_tables_error():
    page = MagicMock()
    page.find_tables.side_effect = RuntimeError("no table layer")
    assert _append_table_markdown(page, "flat text") == "flat text"


def test_survives_extract_error():
    t = MagicMock()
    t.extract.side_effect = RuntimeError("bad table")
    page = MagicMock()
    page.find_tables.return_value = MagicMock(tables=[t])
    assert _append_table_markdown(page, "flat text") == "flat text"


def test_flag_defaults_off():
    """Probe arm B must never be on by default — this is throwaway measurement code."""
    from src.agentrag.config import Settings

    assert Settings.model_fields["PDF_PRESERVE_TABLES"].default is False


def test_parse_leaves_pages_untouched_when_the_flag_is_off(monkeypatch, tmp_path):
    import sys

    from src.agentrag.config import settings

    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", False)
    monkeypatch.setattr(settings, "PDF_OCR_FALLBACK_ENABLED", False)

    page = _page([GRID])
    page.get_text.return_value = "flat page text with plenty of characters here"
    doc = MagicMock()
    doc.__iter__.return_value = iter([page])
    fake_fitz = MagicMock()
    fake_fitz.open.return_value = doc

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with patch.dict(sys.modules, {"fitz": fake_fitz}):
        out = PDFParser().parse(str(pdf))

    assert "| STT | Liều |" not in out["parsed_content"]


def test_parse_appends_the_table_when_the_flag_is_on(monkeypatch, tmp_path):
    import sys

    from src.agentrag.config import settings

    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", True)
    monkeypatch.setattr(settings, "PDF_OCR_FALLBACK_ENABLED", False)

    page = _page([GRID])
    page.get_text.return_value = "flat page text with plenty of characters here"
    doc = MagicMock()
    doc.__iter__.return_value = iter([page])
    fake_fitz = MagicMock()
    fake_fitz.open.return_value = doc

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with patch.dict(sys.modules, {"fitz": fake_fitz}):
        out = PDFParser().parse(str(pdf))

    assert "| STT | Liều |" in out["parsed_content"]
    assert "flat page text" in out["parsed_content"]  # additive, never a replacement


def test_table_append_does_not_change_ocr_routing(monkeypatch, tmp_path):
    """THE load-bearing test for the '25% blind spot' scope claim.

    A page with a thin text layer must take the OCR path in BOTH arms. If the
    table markdown is appended before the OCR length check, arm B pushes the
    page over PDF_OCR_MIN_TEXT_CHARS, skips the OCR/vision fallback arm A takes,
    and loses content for reasons that have nothing to do with tables.

    Note the two mechanics this test depends on, both verified against
    pdf_parser.py: `import fitz` is INSIDE `PDFParser.parse`, so the module has no
    `fitz` attribute to patch — the import must be intercepted in `sys.modules`.
    And `parse` raises FileNotFoundError before opening anything, so the path
    argument must exist on disk.
    """
    import sys

    from src.agentrag.config import settings

    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", True)
    monkeypatch.setattr(settings, "PDF_OCR_FALLBACK_ENABLED", True)
    monkeypatch.setattr(settings, "PDF_OCR_MIN_TEXT_CHARS", 50)
    monkeypatch.setattr(settings, "PDF_OCR_VISION_FALLBACK", False)
    monkeypatch.setattr(settings, "PDF_OCR_VISION_THRESHOLD", 1)

    thin_page = _page([GRID])                     # 30 chars: under the threshold
    thin_page.get_text.return_value = "x" * 30
    thin_page.get_pixmap.return_value.tobytes.return_value = b"png"
    doc = MagicMock()
    doc.__iter__.return_value = iter([thin_page])
    fake_fitz = MagicMock()
    fake_fitz.open.return_value = doc

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with patch.dict(sys.modules, {"fitz": fake_fitz}), \
         patch("src.agentrag.ingestion.parsers.pdf_parser._ocr_tesseract",
               return_value="recovered OCR text") as ocr:
        PDFParser().parse(str(pdf))

    ocr.assert_called_once()   # arm B still took the OCR path
