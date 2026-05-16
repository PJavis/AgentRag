"""Per-turn ContextVars for retrieval — propagate UI overrides through
async call trees without threading kwargs everywhere.

Set in adapter.routers.chat → AgentService.chat. Read in
KnowledgeService.bootstrap_search / execute_tool when calling the retriever.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_domain_filter: ContextVar[dict[str, Any] | None] = ContextVar(
    "agentrag_domain_filter", default=None
)


def set_domain_filter(value: dict[str, Any] | None):
    """Set current turn's domain_filter; returns Token for later reset (optional)."""
    return _domain_filter.set(value)


def get_domain_filter() -> dict[str, Any] | None:
    return _domain_filter.get()
