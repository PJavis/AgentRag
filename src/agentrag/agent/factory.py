"""Resolve agent backend based on AGENT_BACKEND config.

Callers should use `get_agent_service()` instead of importing AgentService
directly so we can flip the backend in one place. Both backends expose the
same `chat()` and `chat_stream()` surface.
"""
from __future__ import annotations

from src.agentrag.config import settings


def get_agent_service():
    """Return an instance of the active agent backend."""
    if settings.AGENT_BACKEND == "langgraph":
        from src.agentrag.agent.graph_service import GraphAgentService
        return GraphAgentService()
    # default: hand-rolled loop
    from src.agentrag.agent.service import AgentService
    return AgentService()
