# main.py
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Body, HTTPException

from src.pam.config import settings
from src.pam.config_validation import validate_settings
from src.pam.health.providers import collect_provider_health
from src.pam.ingestion.pipeline import ingest_folder
from src.pam.graph.graph_jobs import run_graph_worker


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
