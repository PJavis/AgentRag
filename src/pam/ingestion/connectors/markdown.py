# src/pam/ingestion/connectors/markdown.py
from pathlib import Path
import hashlib
from typing import List, Dict

class MarkdownConnector:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)

    def list_documents(self) -> List[Dict]:
        documents = []
        for md_file in self.folder_path.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            documents.append({
                "source_id": str(md_file.relative_to(self.folder_path)),
                "title": md_file.stem,
                "content": content,
                "content_hash": content_hash,
                "source_type": "markdown",
            })
        return documents
