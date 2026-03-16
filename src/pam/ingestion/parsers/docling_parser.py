# src/pam/ingestion/parsers/docling_parser.py
from docling.document_converter import DocumentConverter
from typing import Dict, Any

class DoclingParser:
    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, content: str, source_type: str = "markdown") -> Dict[str, Any]:
        # Docling hỗ trợ Markdown trực tiếp
        result = self.converter.convert(content, format="text")  # hoặc tùy format
        return {
            "parsed_content": result.document.export_to_markdown(),  # hoặc export_to_text()
            "structure": result.document.to_dict()  # giữ metadata layout nếu cần
        }
