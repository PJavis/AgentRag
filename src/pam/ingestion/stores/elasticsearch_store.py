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
            action = {
                "_index": self.index_name,
                "_id": str(uuid.uuid4()),
                "_source": {
                    "content": chunk["content"],
                    "embedding": chunk["embedding"],
                    "document_title": document_title,
                    "section_path": chunk["section_path"],
                    "position": chunk["position"],
                    "content_hash": chunk["content_hash"],
                    "metadata": chunk["metadata"],
                }
            }
            actions.append(action)

        if actions:
            await self.client.bulk(body=actions, refresh=True)
            print(f"Indexed {len(actions)} segments to ES")
