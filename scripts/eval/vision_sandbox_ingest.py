"""Slice B sandbox ingest: caption 5 image-heavy PDFs with local qwen2.5vl:7b.

Non-destructive: writes ONLY to rag_scratch DB + agentrag_segments_scratch ES index
+ /tmp/vis_images. Captioning is 100% local (Ollama). Env MUST be exported before
this process starts (settings is an import-time singleton)."""
import asyncio, sys
sys.path.insert(0, ".")
from src.agentrag.config import settings
from src.agentrag.database import engine, Base
from src.agentrag.ingestion.pipeline import ingest_folder


async def main():
    # Hard guards — refuse to run against prod DB/index/image dir.
    assert settings.POSTGRES_DB == "rag_scratch", f"NOT scratch DB: {settings.POSTGRES_DB}"
    assert settings.ELASTICSEARCH_INDEX_NAME == "agentrag_segments_scratch", settings.ELASTICSEARCH_INDEX_NAME
    assert settings.VISION_PROVIDER in ("ollama", "gemini") and settings.VISION_INGEST_MODE == "sync", \
        (settings.VISION_PROVIDER, settings.VISION_INGEST_MODE)
    assert settings.IMAGE_STORAGE_DIR == "/tmp/vis_images", settings.IMAGE_STORAGE_DIR
    assert settings.VISION_PROVIDER in ("ollama", "gemini"), settings.VISION_PROVIDER
    print("GUARDS OK | vision_model=%s provider=%s ingest_mode=%s img_dir=%s db=%s index=%s" % (
        settings.VISION_MODEL, settings.VISION_PROVIDER, settings.VISION_INGEST_MODE,
        settings.IMAGE_STORAGE_DIR, settings.POSTGRES_DB, settings.ELASTICSEARCH_INDEX_NAME))
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # scratch schema (TAGGING off → no ontology_terms need)
    rep = await ingest_folder("/tmp/vis_docs")
    if isinstance(rep, dict):
        print("INGEST:", {k: rep.get(k) for k in ("status", "ingested", "total", "pdf_images")})
    else:
        print("INGEST:", rep)


asyncio.run(main())
