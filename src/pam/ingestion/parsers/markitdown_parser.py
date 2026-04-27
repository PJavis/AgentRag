"""MarkItDownParser: chuyển PDF / DOCX / PPTX / Excel / HTML thành Markdown.

Dùng microsoft/markitdown — pure-Python, không cần ML model.
Thay thế DoclingParser + ImageDescriber.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MarkItDownParser:
    def __init__(self) -> None:
        from markitdown import MarkItDown
        self._mid = MarkItDown()

    def parse(self, file_path: str) -> dict[str, Any]:
        """
        Returns:
          {
            "parsed_content": str,   # markdown text
            "pages": int,            # best-effort (1 nếu không detect được)
          }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        result = self._mid.convert(str(path))
        content = result.text_content or ""

        # Đếm trang thô: đếm số lần xuất hiện form-feed hoặc "Page N" marker
        pages = max(1, content.count("\f") or content.count("<!-- page"))

        return {
            "parsed_content": content,
            "pages": pages,
        }
