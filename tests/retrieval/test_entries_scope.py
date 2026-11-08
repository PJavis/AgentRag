"""StructMem entries honour the notebook document scope.

hybrid_kg fused StructMem entries searched WITHOUT the scope, so a 1-document
notebook returned citations from 14 documents (reproduced on master
2026-11-08).
"""
from __future__ import annotations

import inspect

import pytest

from src.agentrag.retrieval import elasticsearch_retriever as er


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Session:
    def __init__(self, rows):
        self.rows = rows
        self.stmts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt):
        self.stmts.append(stmt)
        return _Rows(self.rows)


class _Store:
    def __init__(self):
        self.group_ids = "unset"

    async def search_entries(self, query_embedding, query_text, group_ids, top_k):
        self.group_ids = group_ids
        return [{"content": "e"}]

    async def search_synthesis(self, query_embedding, query_text, group_ids, top_k):
        return []


def _retriever(monkeypatch, rows):
    r = er.ElasticsearchRetriever.__new__(er.ElasticsearchRetriever)
    r.store = _Store()
    session = _Session(rows)
    monkeypatch.setattr(er, "AsyncSessionLocal", lambda: session)
    return r, session


async def test_scope_titles_restrict_entries_to_their_groups(monkeypatch):
    r, _ = _retriever(monkeypatch, [("quasar-manual.md",)])
    hits, reason = await r._entries_search("q", [0.0], 5, None, scope_titles=["quasar-manual"])
    assert reason == "ok"
    assert r.store.group_ids == [er.StructMemService.normalize_group_id("quasar-manual.md")]


async def test_scope_resolving_to_nothing_returns_no_entries(monkeypatch):
    # Must NOT widen to every document's memory.
    r, _ = _retriever(monkeypatch, [])
    hits, reason = await r._entries_search("q", [0.0], 5, None, scope_titles=["gone"])
    assert (hits, reason) == ([], "scope_empty")
    assert r.store.group_ids == "unset"


async def test_unscoped_search_stays_global(monkeypatch):
    r, session = _retriever(monkeypatch, [])
    await r._entries_search("q", [0.0], 5, None)
    assert r.store.group_ids is None
    assert session.stmts == []


def test_hybrid_kg_passes_scope_to_entries_search():
    src = inspect.getsource(er.ElasticsearchRetriever)
    assert 'scope_titles=(filters or {}).get("document_titles")' in src
