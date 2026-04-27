"""
FolderConnector: scan thư mục cho tất cả định dạng tài liệu được hỗ trợ.
Trả về cùng format dict như MarkdownConnector để pipeline không cần thay đổi.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

# Mapping ext → source_type
_EXT_TO_SOURCE_TYPE: dict[str, str] = {
    ".md":   "markdown",
    ".pdf":  "pdf",
    ".docx": "word",
    ".doc":  "word",
    ".xlsx": "excel",
    ".xls":  "excel",
    ".csv":  "csv",
}

SUPPORTED_EXTENSIONS = set(_EXT_TO_SOURCE_TYPE.keys())


class FolderConnector:
    """Scan thư mục đệ quy cho tất cả định dạng được hỗ trợ."""

    def __init__(self, folder_path: str, extensions: set[str] | None = None):
        self.folder_path = Path(folder_path).resolve()
        self.extensions = extensions or SUPPORTED_EXTENSIONS

    def list_documents(self) -> List[Dict]:
        documents: list[dict] = []
        for path in sorted(self.folder_path.rglob("*")):
            if path.suffix.lower() not in self.extensions:
                continue
            if not path.is_file():
                continue
            file_path = path.resolve()
            content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            documents.append(
                {
                    "source_id": str(path.relative_to(self.folder_path)),
                    "title": path.stem,
                    "file_path": str(file_path),
                    "content_hash": content_hash,
                    "source_type": _EXT_TO_SOURCE_TYPE.get(path.suffix.lower(), "unknown"),
                }
            )
        return documents
