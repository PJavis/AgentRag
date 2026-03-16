# src/pam/ingestion/pipeline.py
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from src.pam.config import settings
from .connectors.markdown import MarkdownConnector
from .parsers.docling_parser import DoclingParser
from .chunkers.hybrid_chunker import HybridChunker
from .embedders.openai_embedder import OpenAIEmbedder
from .stores.postgres_store import PostgresStore
from .stores.elasticsearch_store import ElasticsearchStore
import asyncio

engine = create_async_engine(settings.DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def ingest_folder(folder_path: str):
    connector = MarkdownConnector(folder_path)
    documents = connector.list_documents()

    parser = DoclingParser()
    chunker = HybridChunker(max_tokens=512)
    embedder = OpenAIEmbedder()
    pg_store = PostgresStore()
    es_store = ElasticsearchStore()

    ingested_count = 0

    async with AsyncSessionLocal() as session:
        for doc in documents:
            # Parse
            parsed = parser.parse(doc["content"])

            # Chunk
            chunks = chunker.chunk(parsed["parsed_content"], metadata={"document_title": doc["title"]})

            # Embed
            texts = [c["content"] for c in chunks]
            embeddings = embedder.embed(texts)
            for c, emb in zip(chunks, embeddings):
                c["embedding"] = emb

            # Lưu Postgres atomic
            doc_id, status = await pg_store.save_document_and_segments(
                session, doc, chunks
            )

            if status == "ingested":
                # Index ES
                await es_store.index_segments(chunks, doc["title"])
                ingested_count += 1

    return {"status": "success", "ingested": ingested_count, "total": len(documents)}