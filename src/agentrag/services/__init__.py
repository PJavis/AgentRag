from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agentrag.services.context_assembly_service import ContextAssemblyService
    from src.agentrag.services.knowledge_service import KnowledgeService
    from src.agentrag.services.llm_gateway import LLMGateway
    from src.agentrag.services.security_service import SecurityService

__all__ = [
    "ContextAssemblyService",
    "KnowledgeService",
    "LLMGateway",
    "SecurityService",
]

# Lazy re-exports (PEP 562) so that importing a leaf submodule such as
# `src.agentrag.services.semantic_cache` does not eagerly pull
# `knowledge_service` -> `agent.tools` -> `elasticsearch_retriever`, which
# would create a circular import.
_LAZY = {
    "ContextAssemblyService": "src.agentrag.services.context_assembly_service",
    "KnowledgeService": "src.agentrag.services.knowledge_service",
    "LLMGateway": "src.agentrag.services.llm_gateway",
    "SecurityService": "src.agentrag.services.security_service",
}


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    attr = getattr(module, name)
    globals()[name] = attr  # cache for subsequent lookups
    return attr
