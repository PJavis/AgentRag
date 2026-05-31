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


# Per-turn document scope — when the user ticks a subset of notebook sources,
# every retrieval in the turn is restricted to those document titles (faster +
# focused). Empty/None = search all documents.
_document_scope: ContextVar[list[str] | None] = ContextVar(
    "agentrag_document_scope", default=None
)


def set_document_scope(titles: list[str] | None):
    """Restrict this turn's retrieval to the given document titles."""
    return _document_scope.set(titles or None)


def get_document_scope() -> list[str] | None:
    return _document_scope.get()
