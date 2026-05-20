"""Resolve the agent service.

Single backend now (LangGraph). The factory is kept as a shim so callers
import via one stable entrypoint; removing it would touch every endpoint
+ CLI without changing behavior.
"""
from __future__ import annotations


def get_agent_service():
    from src.agentrag.agent.graph_service import GraphAgentService
    return GraphAgentService()
