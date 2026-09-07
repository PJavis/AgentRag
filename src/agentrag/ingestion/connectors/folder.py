"""
FolderConnector: scan thư mục cho tất cả định dạng tài liệu được hỗ trợ.
Trả về cùng format dict như MarkdownConnector để pipeline không cần thay đổi.
"""
from __future__ import annotations

import hashlib
import json

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


#: Settings that change what a PDF parse PRODUCES. Any of these differing from
#: its shipped default changes the stored representation, so it belongs in the
#: re-ingest cache key. Discovered the hard way: pointing VISION_BASE_URL at a
#: host without the configured vision model silently degraded every scanned page
#: to ~40 characters, and a re-ingest would have reported "skipped" for all of
#: them because the file bytes had not changed.
_PARSE_AFFECTING_SETTINGS = (
    "PDF_PRESERVE_TABLES",
    "PDF_OCR_FALLBACK_ENABLED",
    "PDF_OCR_LANG",
    "PDF_OCR_DPI",
    "PDF_OCR_MIN_TEXT_CHARS",
    "PDF_OCR_VISION_FALLBACK",
    "PDF_OCR_VISION_THRESHOLD",
    "VISION_PROVIDER",
    "VISION_MODEL",
    "VISION_BASE_URL",
)


def _parse_config_fingerprint() -> str:
    """Canonical string for parse-affecting settings that differ from default.

    Only non-default values are included, so an untouched deployment keeps the
    plain file hash and is never re-ingested merely because this function was
    added. Flipping a setting back to its default restores the original key.
    """
    from src.agentrag.config import Settings

    changed = {}
    for name in _PARSE_AFFECTING_SETTINGS:
        field = Settings.model_fields.get(name)
        if field is None:
            continue
        current = getattr(settings, name, None)
        if current != field.default:
            changed[name] = current
    if not changed:
        return ""
    return json.dumps(changed, sort_keys=True, default=str)


def _document_cache_key(file_path: Path, suffix: str) -> str:
    """Hash identifying a document's *stored representation*, not just its bytes.

    `save_document_and_segments` skips a document whose `content_hash` already
    matches, so this value is the re-ingest cache key. Hashing the file alone is
    wrong whenever a setting changes how the file is PARSED: the bytes are
    identical, every document reports "skipped", and the new setting silently
    never takes effect — a change that looks successful and does nothing.

    Only PDFs carry the fingerprint: every setting in
    `_PARSE_AFFECTING_SETTINGS` governs the PDF path, so mixing it into other
    source types would invalidate them for a change that cannot reach them.
    """
    digest = hashlib.sha256(file_path.read_bytes())
    if suffix == ".pdf":
        fingerprint = _parse_config_fingerprint()
        if fingerprint:
            digest.update(b"|parse:" + fingerprint.encode("utf-8"))
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
            raw_sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
            content_hash = _document_cache_key(file_path, path.suffix.lower())
            documents.append(
                {
                    "source_id": str(path.relative_to(self.folder_path)),
                    "title": path.stem,
                    "file_path": str(file_path),
                    "content_hash": content_hash,
                    # Identity of the FILE, independent of parser settings.
                    # Derived artefacts that do NOT depend on the parse (the
                    # contextualizer's per-chunk blurb) key on this instead, so
                    # a parser-flag flip does not discard them.
                    "source_bytes_sha": raw_sha,
                    "source_type": _EXT_TO_SOURCE_TYPE.get(path.suffix.lower(), "unknown"),
                }
            )
        return documents
