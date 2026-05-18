"""MinerU parser shim — smoke + fallback tests (no real CLI invocation)."""
from __future__ import annotations

import json
from unittest.mock import patch


def test_is_available_returns_bool():
    from src.agentrag.ingestion.parsers import mineru_parser
    assert isinstance(mineru_parser.is_available(), bool)


def test_parse_pdf_returns_none_when_cli_missing(tmp_path):
    from src.agentrag.ingestion.parsers import mineru_parser
    fake_pdf = tmp_path / "x.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")
    with patch.object(mineru_parser, "is_available", return_value=False):
        out = mineru_parser.parse_pdf(str(fake_pdf))
    assert out is None


def test_collect_mineru_output_uses_content_list(tmp_path):
    from src.agentrag.ingestion.parsers import mineru_parser
    out_root = tmp_path / "doc" / "auto"
    out_root.mkdir(parents=True)
    (out_root / "doc.md").write_text("# stuff", encoding="utf-8")
    (out_root / "doc_content_list.json").write_text(json.dumps([
        {"page_idx": 0, "text": "Page one body."},
        {"page_idx": 1, "text": "Page two body."},
    ]), encoding="utf-8")
    result = mineru_parser._collect_mineru_output(tmp_path, stem="doc")
    assert result is not None
    assert result["pages"] == 2
    assert result["page_data"][0]["page_num"] == 1
    assert result["page_data"][0]["source"] == "mineru"
    assert "Page one body." in result["parsed_content"]


def test_collect_falls_back_to_single_page_when_no_json(tmp_path):
    from src.agentrag.ingestion.parsers import mineru_parser
    md_dir = tmp_path / "doc" / "auto"
    md_dir.mkdir(parents=True)
    (md_dir / "doc.md").write_text("# only md", encoding="utf-8")
    result = mineru_parser._collect_mineru_output(tmp_path, stem="doc")
    assert result is not None
    assert result["pages"] == 1
    assert result["page_data"][0]["source"] == "mineru"
