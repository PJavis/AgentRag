# main.py
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Body, HTTPException
from sqlalchemy import select

from src.pam.config import settings
from src.pam.config_validation import validate_settings
from src.pam.database import AsyncSessionLocal
from src.pam.database.models import Document
from src.pam.health.providers import collect_provider_health
from src.pam.ingestion.pipeline import ingest_folder
from src.pam.graph.graph_jobs import run_graph_worker
from src.pam.retrieval.elasticsearch_retriever import ElasticsearchRetriever
from src.pam.agent.service import AgentService


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_settings(settings)
    stop = asyncio.Event()
    worker = asyncio.create_task(run_graph_worker(stop))
    yield
    stop.set()
    await worker


app = FastAPI(lifespan=lifespan)


@app.get("/config/validate")
async def config_validate():
    try:
        validate_settings(settings)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "providers": {
            "embedding": settings.EMBEDDING_PROVIDER,
            "extraction": settings.EXTRACTION_PROVIDER,
            "graph_embedding": settings.GRAPH_EMBEDDING_PROVIDER,
        },
    }


@app.get("/health/providers")
async def provider_health():
    report = collect_provider_health(settings)
    if not report["ok"]:
        raise HTTPException(status_code=400, detail=report)
    return report


@app.post("/ingest/folder")
async def ingest(payload: dict = Body(...)):
    folder_path = payload.get("folder_path")
    if not folder_path:
        return {"error": "folder_path is required"}
    mode = payload.get("graph_ingest_mode")
    if mode is not None and mode not in ("sync", "async"):
        return {"error": "graph_ingest_mode must be 'sync' or 'async'"}
    result = await ingest_folder(folder_path, graph_ingest_mode=mode)
    return result


@app.post("/search")
async def search(payload: dict = Body(...)):
    query = payload.get("query")
    if not query:
        return {"error": "query is required"}

    mode = payload.get("mode", "hybrid_kg")
    top_k = payload.get("top_k")
    document_title = payload.get("document_title")
    rerank = payload.get("rerank")

    try:
        retriever = ElasticsearchRetriever()
        result = await retriever.search(
            query=query,
            mode=mode,
            top_k=top_k,
            document_title=document_title,
            rerank=rerank,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.get("/documents/{document_id}/graph-status")
async def graph_status(document_id: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()

    if document is None:
        raise HTTPException(status_code=404, detail="document not found")

    total = document.graph_total_chunks or 0
    processed = document.graph_processed_chunks or 0
    failed = document.graph_failed_chunks or 0
    progress = (processed / total) if total else 0.0

    return {
        "document_id": str(document.id),
        "title": document.title,
        "graph_status": document.graph_status,
        "graph_synced": document.graph_synced,
        "graph_total_chunks": total,
        "graph_processed_chunks": processed,
        "graph_failed_chunks": failed,
        "graph_progress": progress,
        "graph_last_error": document.graph_last_error,
    }


@app.post("/chat")
async def chat(payload: dict = Body(...)):
    question = payload.get("question")
    if not question:
        return {"error": "question is required"}
    document_title = payload.get("document_title")
    agent = AgentService()
    return await agent.chat(question=question, document_title=document_title)
