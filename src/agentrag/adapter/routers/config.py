"""Open-notebook config, auth/status, and settings endpoints."""
from __future__ import annotations

from fastapi import APIRouter
from src.agentrag.config import settings

router = APIRouter()


@router.get("/config")
async def get_config():
    return {
        "version": settings.ADAPTER_VERSION,
        "latestVersion": settings.ADAPTER_VERSION,
        "hasUpdate": False,
        "dbStatus": "connected",
    }


@router.get("/settings")
async def get_settings():
    return {
        "default_content_processing_engine_doc": "agentrag",
        "default_content_processing_engine_url": "agentrag",
        "default_embedding_option": settings.EMBEDDING_PROVIDER,
        "auto_delete_files": False,
        "youtube_preferred_languages": ["vi", "en"],
    }


@router.put("/settings")
async def update_settings(payload: dict):
    return await get_settings()


@router.get("/languages")
async def get_languages():
    return [
        {"code": "vi", "name": "Tiếng Việt"},
        {"code": "en", "name": "English"},
    ]


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/metrics/cost")
async def metrics_cost():
    """Per-task + per-model LLM cost summary (in-memory, cleared on restart)."""
    from src.agentrag.observability.cost import cost_summary
    return cost_summary()


@router.post("/metrics/cost/reset")
async def metrics_cost_reset():
    from src.agentrag.observability.cost import reset_ledger
    reset_ledger()
    return {"ok": True}
