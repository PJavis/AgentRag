# src/pam/graph/graphiti_service.py
from graphiti import Graphiti
from ollama import Client  # hoặc HF nếu muốn
from typing import List, Dict
from src.pam.config import settings

class GraphitiService:
    def __init__(self):
        self.graph = Graphiti(
            uri="bolt://localhost:7687",
            user="neo4j",
            password=settings.NEO4J_PASSWORD or "neo4j123456",
            llm_provider="ollama",  # hoặc "openai" nếu fallback
            llm_model="llama3.1:70b-instruct-q4_K_M",  # dùng Ollama
            embedding_model="nomic-embed-text"  # embedding local
        )

    async def extract_and_sync_chunk(self, chunk: Dict, document_id: str, chunk_id: str):
        """
        Extract entity/relation từ chunk và sync vào Neo4j
        - chunk: {"content": "...", "embedding": [...], "section_path": "..."}
        - Tự động tạo episode với bitemporal
        """
        episode = await self.graph.add_episode(
            content=chunk["content"],
            source=document_id,
            episode_id=chunk_id,
            valid_at=datetime.utcnow(),  # thời gian sự kiện hợp lệ
            invalid_at=None  # chưa hết hạn
        )

        return {
            "episode_id": episode.id,
            "entities": episode.entities,
            "relationships": episode.relationships
        }

    async def sync_chunks(self, chunks: List[Dict], document_id: str):
        results = []
        for i, chunk in enumerate(chunks):
            result = await self.extract_and_sync_chunk(chunk, document_id, f"{document_id}_chunk_{i}")
            results.append(result)
        return results
