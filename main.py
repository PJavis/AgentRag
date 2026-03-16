# main.py
from fastapi import FastAPI
from src.pam.ingestion.pipeline import ingest_folder

app = FastAPI()

@app.post("/ingest/folder")
async def ingest(folder_path: str):
    result = await ingest_folder(folder_path)
    return result