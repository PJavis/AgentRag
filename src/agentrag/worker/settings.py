"""ARQ WorkerSettings — used by the arq CLI to start background workers.

Start a worker process:
    arq src.agentrag.worker.settings.WorkerSettings

Or with uvicorn + worker in separate terminals:
    uvicorn main:app --reload          # API server
    arq src.agentrag.worker.settings.WorkerSettings   # background worker
"""
from __future__ import annotations

from arq.connections import RedisSettings

from src.agentrag.common.build_info import log_build_banner
from src.agentrag.config import settings as app_settings
from src.agentrag.worker.functions import chat_memory, consolidate, graph_ingest, vision_extract


async def _on_startup(ctx: dict) -> None:
    """Which code is actually running. The worker ingests documents, so a stale
    image here corrupts a whole corpus silently rather than erroring."""
    log_build_banner("worker")


class WorkerSettings:
    functions = [graph_ingest, consolidate, chat_memory, vision_extract]
    on_startup = _on_startup
    redis_settings = RedisSettings.from_dsn(
        app_settings.REDIS_URL or "redis://127.0.0.1:6379/0"
    )
    # Max concurrent document jobs per worker process (keep 1 for local Ollama)
    max_jobs = app_settings.STRUCTMEM_WORKER_MAX_JOBS
    # Timeout per job (seconds) — covers the full document, not a single chunk
    job_timeout = app_settings.STRUCTMEM_JOB_TIMEOUT_SECONDS
    # Keep job result in Redis for 1 hour (for status inspection)
    keep_result = 3600
    # Retry failed jobs once
    max_tries = 2
