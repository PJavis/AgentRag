# src/pam/ingestion/pipeline.py
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from src.pam.config import settings
from .connectors.markdown import MarkdownConnector
from .parsers.docling_parser import DoclingParser
from .chunkers.hybrid_chunker import HybridChunker
from .embedders.openai_embedder import OpenAIEmbedder
from .embedders.hf_inference_embedder import HFInferenceEmbedder
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
    # embedder = OpenAIEmbedder()
    embedder = HFInferenceEmbedder(model="intfloat/multilingual-e5-large-instruct")
    pg_store = PostgresStore()
    es_store = ElasticsearchStore()

    ingested_count = 0

    async with AsyncSessionLocal() as session:
        for doc in documents:
            file_path = doc["file_path"]

            # Parse từ path (Docling tự đọc file)
            parsed = parser.parse(file_path)

            # Đọc content từ file để chunk (vì parsed_content có thể khác định dạng)
            content = Path(file_path).read_text(encoding="utf-8")
            chunks = chunker.chunk(content, metadata={"document_title": doc["title"]})

            # Embed
            texts = [c["content"] for c in chunks]
            embeddings = await embedder.embed(texts)  # await vì async
            for c, emb in zip(chunks, embeddings):
                c["embedding"] = emb

            # Lưu Postgres
            doc_id, status = await pg_store.save_document_and_segments(
                session, doc, chunks
            )

            if status == "ingested":
                await es_store.index_segments(chunks, doc["title"])
                ingested_count += 1

    return {"status": "success", "ingested": ingested_count, "total": len(documents)}