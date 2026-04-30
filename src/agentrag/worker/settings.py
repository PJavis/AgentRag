"""ARQ WorkerSettings — used by the arq CLI to start background workers.

Start a worker process:
    arq src.agentrag.worker.settings.WorkerSettings

Or with uvicorn + worker in separate terminals:
    uvicorn main:app --reload          # API server
    arq src.agentrag.worker.settings.WorkerSettings   # background worker
"""
from __future__ import annotations

from arq.connections import RedisSettings

from src.agentrag.config import settings as app_settings
from src.agentrag.worker.functions import chat_memory, consolidate, graph_ingest


class WorkerSettings:
    functions = [graph_ingest, consolidate, chat_memory]
    redis_settings = RedisSettings.from_dsn(
        app_settings.REDIS_URL or "redis://127.0.0.1:6379/0"
    )
    # Max concurrent jobs per worker process
    max_jobs = app_settings.STRUCTMEM_MAX_CONCURRENCY or 4
    # Timeout per job (seconds) — long for heavy ingest jobs
    job_timeout = app_settings.STRUCTMEM_CHUNK_TIMEOUT_SECONDS or 600
    # Keep job result in Redis for 1 hour (for status inspection)
    keep_result = 3600
    # Retry failed jobs once
    max_tries = 2
