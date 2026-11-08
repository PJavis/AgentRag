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


# Per-turn document scope — every retrieval in the turn is restricted to these
# document titles (notebook isolation, or a ticked subset of sources).
# None = unscoped (search everything). [] = scoped to NOTHING (an empty
# notebook must not fall back to the whole corpus).
_document_scope: ContextVar[list[str] | None] = ContextVar(
    "agentrag_document_scope", default=None
)


def set_document_scope(titles: list[str] | None):
    """Restrict this turn's retrieval to the given document titles.

    Pass None for "no scope". An empty list means the scope is empty and every
    scoped search returns nothing — it used to be coerced to None, so an empty
    notebook searched the entire corpus."""
    return _document_scope.set(list(titles) if titles is not None else None)


def get_document_scope() -> list[str] | None:
    return _document_scope.get()


def current_retrieval_filters() -> dict[str, Any] | None:
    """The turn's retrieval filters (domain tags + document scope) in the shape
    the retriever expects. Every retrieval made on behalf of a turn must use
    this — the context assembler's raw-question search skipped it and leaked
    other notebooks' documents into every scoped answer."""
    df = get_domain_filter() or {}
    out: dict[str, Any] = {}
    if df.get("system"):
        out["systems"] = [df["system"]]
    if df.get("specialties"):
        out["specialties"] = list(df["specialties"])
    scope = get_document_scope()
    if scope is not None:
        out["document_titles"] = list(scope)
    return out or None
