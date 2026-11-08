"""StructMem entries honour the notebook document scope.

hybrid_kg fused StructMem entries searched WITHOUT the scope, so a 1-document
notebook returned citations from 14 documents (reproduced on master
2026-11-08).
"""
from __future__ import annotations

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


def _sql(stmt) -> str:
    return str(stmt.compile(compile_kwargs={"literal_binds": True}))


async def test_document_title_outside_scope_gets_no_entries(monkeypatch):
    # The planner may pass a document_title from another notebook: the lookup
    # must require BOTH the scope and the title, so it resolves to nothing.
    r, session = _retriever(monkeypatch, [])
    hits, reason = await r._entries_search("q", [0.0], 5, "other", scope_titles=["quasar-manual"])
    assert (hits, reason) == ([], "scope_empty")
    sql = _sql(session.stmts[0])
    assert "documents.title IN ('quasar-manual')" in sql and "documents.title = 'other'" in sql


A = "11111111-1111-4111-8111-111111111111"


async def test_scope_ids_restrict_entries_by_document_id(monkeypatch):
    # Ids, not titles: two users' "report.pdf" share a title.
    r, session = _retriever(monkeypatch, [(f"{A}/report.pdf",)])
    hits, reason = await r._entries_search("q", [0.0], 5, None, scope_ids=[A])
    assert reason == "ok"
    assert r.store.group_ids == [er.StructMemService.normalize_group_id(f"{A}/report.pdf")]
    sql = _sql(session.stmts[0])
    assert "documents.id IN" in sql and "documents.title" not in sql


async def test_empty_or_invalid_scope_ids_return_nothing(monkeypatch):
    r, session = _retriever(monkeypatch, [("x",)])
    assert await r._entries_search("q", [0.0], 5, None, scope_ids=[]) == ([], "scope_empty")
    assert await r._entries_search("q", [0.0], 5, None, scope_ids=["not-a-uuid"]) == ([], "scope_empty")
    assert session.stmts == []


async def test_document_title_inside_scope_narrows(monkeypatch):
    r, session = _retriever(monkeypatch, [("quasar-manual.md",)])
    await r._entries_search("q", [0.0], 5, "quasar-manual", scope_titles=["quasar-manual", "b"])
    assert r.store.group_ids == [er.StructMemService.normalize_group_id("quasar-manual.md")]


async def test_empty_scope_list_returns_nothing(monkeypatch):
    r, session = _retriever(monkeypatch, [("x",)])
    assert await r._entries_search("q", [0.0], 5, None, scope_titles=[]) == ([], "scope_empty")
    assert session.stmts == []


async def test_scope_lookup_failure_fails_soft(monkeypatch):
    r = er.ElasticsearchRetriever.__new__(er.ElasticsearchRetriever)
    r.store = _Store()

    class Broken:
        async def __aenter__(self):
            raise ConnectionError("db")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(er, "AsyncSessionLocal", lambda: Broken())
    hits, reason = await r._entries_search("q", [0.0], 5, None, scope_titles=["a"])
    assert hits == [] and reason.startswith("entries_scope_lookup_exception")
