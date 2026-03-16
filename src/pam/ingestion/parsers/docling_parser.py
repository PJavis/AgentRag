# src/pam/ingestion/parsers/docling_parser.py
from docling.document_converter import DocumentConverter
from pathlib import Path
from typing import Dict, Any

class DoclingParser:
    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, file_path: str, source_type: str = "markdown") -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Convert từ path
        result = self.converter.convert(path)

        # Export text giữ cấu trúc (export_to_markdown là chuẩn)
        parsed_text = result.document.export_to_markdown()

        # Lấy metadata cơ bản (không dùng to_dict vì không tồn tại)
        structure = {
            "title": result.document.title if hasattr(result.document, "title") else path.stem,
            "num_pages": len(result.document.pages) if hasattr(result.document, "pages") else 1,
            "format": result.document.format if hasattr(result.document, "format") else "unknown",
        }

        return {
            "parsed_content": parsed_text,
            "structure": structure,
            "pages": structure["num_pages"],
        }