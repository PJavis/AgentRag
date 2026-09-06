"""
FolderConnector: scan thư mục cho tất cả định dạng tài liệu được hỗ trợ.
Trả về cùng format dict như MarkdownConnector để pipeline không cần thay đổi.
"""
from __future__ import annotations

import hashlib

from src.agentrag.config import settings
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
    # Standalone image files — described via Vision LLM and indexed as image chunks
    ".jpg":  "image",
    ".jpeg": "image",
    ".png":  "image",
    ".webp": "image",
    ".bmp":  "image",
    ".gif":  "image",
    # Audio — transcribed via Whisper, indexed as text chunks with timestamps
    ".mp3":  "audio",
    ".wav":  "audio",
    ".m4a":  "audio",
    ".ogg":  "audio",
    ".flac": "audio",
    ".aac":  "audio",
    ".opus": "audio",
}

SUPPORTED_EXTENSIONS = set(_EXT_TO_SOURCE_TYPE.keys())


def _document_cache_key(file_path: Path, suffix: str) -> str:
    """Hash identifying a document's *stored representation*, not just its bytes.

    `save_document_and_segments` skips a document whose `content_hash` already
    matches, so this value is the re-ingest cache key. Hashing the file alone is
    wrong whenever a setting changes how the file is PARSED: the bytes are
    identical, every document reports "skipped", and the new setting silently
    never takes effect — a flag flip that looks successful and changes nothing.

    So a non-default parser setting is mixed in. Only non-default ones, so that
    existing corpora keep their current hashes and are not re-ingested for no
    reason; turning the setting back off restores the original key.
    """
    digest = hashlib.sha256(file_path.read_bytes())
    if suffix == ".pdf" and settings.PDF_PRESERVE_TABLES:
        digest.update(b"|pdf_preserve_tables=1")
    return digest.hexdigest()


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
            content_hash = _document_cache_key(file_path, path.suffix.lower())
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
