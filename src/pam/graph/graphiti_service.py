# src/pam/graph/graphiti_service.py
from graphiti_core import Graphiti
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from openai import AsyncOpenAI
from datetime import datetime
from typing import List, Dict
from src.pam.config import settings

class GraphitiService:
    def __init__(self):
        # 1. Setup LLM Config for Ollama
        llm_config = LLMConfig(
            api_key="ollama", 
            model="llama3.1:8b-instruct-q5_K_M",
            base_url="http://127.0.0.1:11434/v1/",
        )

        # 2. Ollama LLM Client
        ollama_llm = AsyncOpenAI(
            base_url=llm_config.base_url,
            api_key=llm_config.api_key
        )
        llm_client = OpenAIGenericClient(client=ollama_llm, config=llm_config)

        # 3. Ollama Embedder
        embedder_config = OpenAIEmbedderConfig(
            api_key="ollama",
            embedding_model="nomic-embed-text",
            base_url="http://127.0.0.1:11434/v1/",
        )
        embedder = OpenAIEmbedder(config=embedder_config)

        # 4. Create a reranker client pointing to Ollama
        reranker = OpenAIRerankerClient(client=llm_client, config=llm_config)

        # 5. Initialize Graphiti
        self.graph = Graphiti(
            uri="bolt://localhost:7687",
            user="neo4j",
            password=settings.NEO4J_PASSWORD or "neo4j123456",
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=reranker,
            store_raw_episode_content=True,
        )

    async def extract_and_sync_chunk(self, chunk: Dict, document_id: str, chunk_id: str):
        episode = await self.graph.add_episode(
            name=f"Episode from {document_id}",
            episode_body=chunk["content"],
            source_description="Extracted from document chunk",
            reference_time=datetime.utcnow(),
            group_id="default",
        )

        return {
            "episode_id": getattr(episode, 'uuid', 'unknown'),
            "entities": [node.name for node in getattr(episode, 'nodes', [])],
            "relationships": [
                f"{edge.source_node_uuid} --{edge.fact or edge.name or 'RELATES_TO'}--> {edge.target_node_uuid}"
                for edge in getattr(episode, 'edges', [])
            ]
        }

    async def sync_chunks(self, chunks: List[Dict], document_id: str):
        results = []
        for i, chunk in enumerate(chunks):
            result = await self.extract_and_sync_chunk(chunk, document_id, f"{document_id}_chunk_{i}")
            results.append(result)
        return results
    
    # Thêm vào class GraphitiService
    async def build_indices(self):
        """Chạy một lần sau khi init Graphiti để tạo index và constraint"""
        print("🔧 Đang build indices và constraints cho Graphiti...")
        await self.graph.build_indices_and_constraints(delete_existing=False)
        print("✅ Graphiti indices đã được tạo!")