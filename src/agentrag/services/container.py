"""ServiceContainer — Execution Plane singleton (S4).

One container per process. Lazy-instantiates each service on first access
so that import time stays cheap and infra-less unit tests don't try to
build live clients.

Reasoning Plane code fetches services by attribute:

    from src.agentrag.services.container import get_container
    container = get_container()
    embedder = container.embedding
    retriever = container.retrieval

This is the only place concrete service classes should be constructed in
production code. Tests can pass their own container instance with mocks.
"""
from __future__ import annotations

from typing import Any

from src.agentrag.config import Settings, settings as global_settings


class ServiceContainer:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or global_settings
        self._llm: Any | None = None
        self._embedding: Any | None = None
        self._vision: Any | None = None
        self._storage: Any | None = None
        self._retrieval: Any | None = None
        self._domain_router: Any | None = None

    # ---- Lazy properties ---------------------------------------------------

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def llm(self):
        if self._llm is None:
            from src.agentrag.services.llm_gateway import LLMGateway
            self._llm = LLMGateway()
        return self._llm

    @property
    def embedding(self):
        if self._embedding is None:
            from src.agentrag.services.embedding_service import EmbeddingService
            self._embedding = EmbeddingService(self._settings)
        return self._embedding

    @property
    def vision(self):
        if self._vision is None:
            from src.agentrag.services.vision_service import VisionService
            self._vision = VisionService(self.llm, self._settings)
        return self._vision

    @property
    def storage(self):
        if self._storage is None:
            from src.agentrag.services.storage_service import StorageService
            self._storage = StorageService()
        return self._storage

    @property
    def retrieval(self):
        if self._retrieval is None:
            from src.agentrag.services.retrieval_service import RetrievalService
            self._retrieval = RetrievalService()
        return self._retrieval

    @property
    def domain_router(self):
        """Reasoning Plane service — but lives here so it's a single instance.
        Reasoning code calls `domain_router.classify(query)` and forwards the
        result to `retrieval.search(system_override=…)`.
        """
        if self._domain_router is None:
            from src.agentrag.orchestration.domain_router import DomainRouter
            self._domain_router = DomainRouter()
        return self._domain_router

    # ---- Test helpers ------------------------------------------------------

    def override(self, **services: Any) -> None:
        """Inject mocks for tests. Keys must match property names."""
        for key, value in services.items():
            attr = f"_{key}"
            if not hasattr(self, attr):
                raise KeyError(f"Unknown service: {key}")
            setattr(self, attr, value)


_container: ServiceContainer | None = None


def get_container() -> ServiceContainer:
    """Return the process-wide ServiceContainer (lazy-initialised)."""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def set_container(container: ServiceContainer | None) -> None:
    """Test/bootstrap hook to swap or reset the singleton."""
    global _container
    _container = container


def reset_container() -> None:
    """Force re-instantiation on next `get_container()`."""
    global _container
    _container = None
