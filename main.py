# main.py
from fastapi import FastAPI, Body
from src.pam.ingestion.pipeline import ingest_folder

app = FastAPI()

@app.post("/ingest/folder")
async def ingest(payload: dict = Body(...)):
    folder_path = payload.get("folder_path")
    if not folder_path:
        return {"error": "folder_path is required"}
    result = await ingest_folder(folder_path)
    return result