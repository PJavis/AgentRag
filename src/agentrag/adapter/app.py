"""Open-notebook compatible FastAPI sub-application."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.agentrag.config import settings
from src.agentrag.adapter.auth import OpenNotebookAuthMiddleware
from src.agentrag.adapter.rate_limit import RateLimitMiddleware
from src.agentrag.adapter.routers.auth import router as auth_router
from src.agentrag.adapter.routers.chat import router as chat_router
from src.agentrag.adapter.routers.chat import source_router as source_chat_router
from src.agentrag.adapter.routers.config import router as config_router
from src.agentrag.adapter.routers.notebooks import router as notebooks_router
from src.agentrag.adapter.routers.notes import router as notes_router
from src.agentrag.adapter.routers.search import router as search_router
from src.agentrag.adapter.routers.sources import router as sources_router
from src.agentrag.adapter.routers.stubs import router as stubs_router
from src.agentrag.adapter.routers.transformations import router as transformations_router
from src.agentrag.adapter.routers.insights import router as insights_router
from src.agentrag.adapter.routers.models import router as models_router
from src.agentrag.adapter.routers.ontology import router as ontology_router

adapter = FastAPI(title="AgentRag — open-notebook adapter", docs_url="/adapter/docs")

adapter.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Order matters: rate-limit AFTER auth so we can throttle per-user.
# In starlette, middlewares added later run first → add rate-limit BEFORE auth.
adapter.add_middleware(RateLimitMiddleware)
adapter.add_middleware(OpenNotebookAuthMiddleware)

# API routes under /api prefix — auth router REGISTERED FIRST so its
# /api/auth/status overrides the stub in config_router (if any).
adapter.include_router(auth_router, prefix="/api")
adapter.include_router(config_router, prefix="/api")
adapter.include_router(notebooks_router, prefix="/api")
adapter.include_router(sources_router, prefix="/api")
adapter.include_router(source_chat_router, prefix="/api")
adapter.include_router(chat_router, prefix="/api")
adapter.include_router(notes_router, prefix="/api")
adapter.include_router(search_router, prefix="/api")
adapter.include_router(transformations_router, prefix="/api")
adapter.include_router(insights_router, prefix="/api")
adapter.include_router(models_router, prefix="/api")
adapter.include_router(ontology_router, prefix="/api")
adapter.include_router(stubs_router, prefix="/api")

# Stored image bytes (extracted from PDFs + standalone uploads) served under
# /on/api/images/* so the frontend can resolve them via the same origin as the
# rest of the API (markdown <img src> rewrites point here).
adapter.mount(
    "/api/images",
    StaticFiles(directory=settings.IMAGE_STORAGE_DIR),
    name="adapter-images",
)


@adapter.get("/")
async def root():
    return {"status": "ok", "service": "agentrag-open-notebook-adapter"}


@adapter.get("/health")
async def health():
    return {"status": "ok"}
