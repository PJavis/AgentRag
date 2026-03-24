# main.py
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Body

from src.pam.ingestion.pipeline import ingest_folder
from src.pam.graph.graph_jobs import run_graph_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop = asyncio.Event()
    worker = asyncio.create_task(run_graph_worker(stop))
    yield
    stop.set()
    await worker


app = FastAPI(lifespan=lifespan)


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
