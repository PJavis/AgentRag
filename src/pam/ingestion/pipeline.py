# src/pam/ingestion/pipeline.py
from pathlib import Path
import asyncio
from datetime import datetime

# SQLAlchemy async
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy import update

# Models & DB
from src.pam.database.models import Document
from src.pam.database import AsyncSessionLocal  # engine + session factory

# Config
from src.pam.config import settings

# Ingestion components
from .connectors.markdown import MarkdownConnector
from .parsers.docling_parser import DoclingParser
from .chunkers.hybrid_chunker import HybridChunker
from .embedders.hf_inference_embedder import HFInferenceEmbedder
from .stores.postgres_store import PostgresStore
from .stores.elasticsearch_store import ElasticsearchStore

# Graphiti (tạm thời comment nếu chưa tạo GraphitiService)
# from src.pam.graph.graphiti_service import GraphitiService

engine = create_async_engine(settings.DATABASE_URL)
# AsyncSessionLocal đã được định nghĩa ở src/pam/database/__init__.py

async def ingest_folder(folder_path: str):
    connector = MarkdownConnector(folder_path)
    documents = connector.list_documents()

    parser = DoclingParser()
    chunker = HybridChunker(max_tokens=512)
    embedder = HFInferenceEmbedder(model="intfloat/multilingual-e5-large-instruct")
    pg_store = PostgresStore()
    es_store = ElasticsearchStore()

    # graph_service = GraphitiService()  # Uncomment khi đã tạo file GraphitiService

    ingested_count = 0

    async with AsyncSessionLocal() as session:
        for doc in documents:
            file_path = doc["file_path"]

            # Parse từ path
            parsed = parser.parse(file_path)

            # Đọc content để chunk
            content = Path(file_path).read_text(encoding="utf-8")
            chunks = chunker.chunk(content, metadata={"document_title": doc["title"]})

            # Embed
            texts = [c["content"] for c in chunks]
            embeddings = await embedder.embed(texts)
            for c, emb in zip(chunks, embeddings):
                c["embedding"] = emb

            # Lưu Postgres
            doc_id, status = await pg_store.save_document_and_segments(
                session, doc, chunks
            )

            if status == "ingested":
                # Index ES
                await es_store.index_segments(chunks, doc["title"])

                # Graphiti sync (comment tạm nếu chưa có GraphitiService)
                # graph_results = await graph_service.sync_chunks(chunks, doc_id)
                # await session.execute(
                #     update(Document)
                #     .where(Document.id == doc_id)
                #     .values(graph_synced=True)
                # )
                # await session.commit()

                ingested_count += 1

    return {"status": "success", "ingested": ingested_count, "total": len(documents)}