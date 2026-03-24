# src/pam/ingestion/stores/elasticsearch_store.py
from elasticsearch import AsyncElasticsearch
from typing import List, Dict
import uuid

class ElasticsearchStore:
    def __init__(self):
        self.client = AsyncElasticsearch(["http://localhost:9200"])
        self.index_name = "pam_segments"

    async def index_segments(self, chunks: List[Dict], document_title: str):
        actions = []

        for chunk in chunks:
            # Format đúng cho bulk API
            actions.append({
                "index": {
                    "_index": self.index_name,
                    "_id": str(uuid.uuid4())
                }
            })
            actions.append({
                "content": chunk["content"],
                "embedding": chunk["embedding"],
                "document_title": document_title,
                "section_path": chunk.get("section_path"),
                "position": chunk.get("position"),
                "content_hash": chunk.get("content_hash"),
                "metadata": chunk.get("metadata", {}),
            })

        if actions:
            try:
                response = await self.client.bulk(body=actions, refresh=True)
                if response.get("errors"):
                    print("[ES Error] Bulk had some errors:", response)
                else:
                    print(f"[ES Success] Indexed {len(chunks)} segments for '{document_title}'")
            except Exception as e:
                print(f"[ES Error] Failed to bulk index: {e}")