from pathlib import Path
import hashlib
from typing import List, Dict

class MarkdownConnector:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path).resolve()

    def list_documents(self) -> List[Dict]:
        documents = []
        for md_file in self.folder_path.rglob("*.md"):
            file_path = md_file.resolve()
            content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            documents.append({
                "source_id": str(md_file.relative_to(self.folder_path)),
                "title": md_file.stem,
                "file_path": str(file_path),  # <-- thêm đường dẫn file
                "content_hash": content_hash,
                "source_type": "markdown",
            })
        return documents