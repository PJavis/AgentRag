"""Every retrieval made for a turn honours the turn's document scope.

A 1-document notebook returned 33 citations from 14 documents (reproduced on
master 2026-11-08): the context assembler's raw-question search ran without
filters, and empty scopes were coerced to "search everything".
"""
from __future__ import annotations

import pytest

from src.agentrag.agent.context import ContextAssembler
from src.agentrag.agent import service as svc
from src.agentrag.ingestion.stores.elasticsearch_store import _tag_filter_clauses
from src.agentrag.retrieval.context import (
    current_retrieval_filters,
    set_document_scope,
    set_domain_filter,
)


DOC_A = "11111111-1111-4111-8111-111111111111"
DOC_B = "22222222-2222-4222-8222-222222222222"
DOC_C = "33333333-3333-4333-8333-333333333333"


@pytest.fixture(autouse=True)
def _reset():
    set_domain_filter(None)
    set_document_scope(None)
    yield
    set_domain_filter(None)
    set_document_scope(None)


async def test_raw_query_hits_use_the_turn_scope(monkeypatch):
    monkeypatch.setattr(svc.settings, "RETRIEVAL_INCLUDE_RAW_QUERY", True, raising=False)
    seen = {}

    class Retriever:
        async def search(self, query, mode="hybrid", top_k=8, filters=None):
            seen["filters"] = filters
            return {"results": []}

    a = ContextAssembler()
    a._retriever = Retriever()
    set_document_scope([DOC_A])
    await a._inject_raw_query_hits("q?", [])
    assert seen["filters"] == {"document_ids": [DOC_A]}


def test_empty_scope_is_kept_not_coerced_to_none():
    set_document_scope([])
    assert current_retrieval_filters() == {"document_ids": []}
    set_document_scope(None)
    assert current_retrieval_filters() is None


def test_empty_scope_matches_nothing_in_es():
    nothing = [{"bool": {"must_not": [{"match_all": {}}]}}]
    assert _tag_filter_clauses({"document_ids": []}) == nothing
    assert _tag_filter_clauses({"document_titles": []}) == nothing
    assert _tag_filter_clauses({"document_ids": [DOC_A]}) == [{"terms": {"document_id": [DOC_A]}}]
    assert _tag_filter_clauses({"document_titles": ["a"]}) == [{"terms": {"document_title.keyword": ["a"]}}]
    assert _tag_filter_clauses({"systems": ["x"]}) == [{"terms": {"system_tag": ["x"]}}]


def test_store_applies_the_turn_scope_when_the_caller_passes_none():
    # Summary / mindmap / tool helpers call the store directly with no filters.
    set_document_scope([DOC_A])
    assert _tag_filter_clauses(None) == [{"terms": {"document_id": [DOC_A]}}]
    assert {"terms": {"document_id": [DOC_A]}} in _tag_filter_clauses({"systems": ["x"]})
    set_document_scope([])
    assert _tag_filter_clauses(None) == [{"bool": {"must_not": [{"match_all": {}}]}}]
    set_document_scope(None)
    assert _tag_filter_clauses(None) == []


def test_tools_use_the_shared_filters():
    from src.agentrag.agent.tools import AgentTools

    set_document_scope([DOC_A])
    set_domain_filter({"system": "cardio"})
    tools = AgentTools.__new__(AgentTools)
    assert tools._current_filters() == {"systems": ["cardio"], "document_ids": [DOC_A]}


async def test_planner_tools_filter_by_scope_ids_not_title(monkeypatch):
    # Two users' "report.pdf" share the title. A title check passed the other
    # user's document; the SQL must restrict by the turn's document ids.
    from src.agentrag.agent import tools as tools_mod

    stmts = []

    class Result:
        def all(self):
            return []

        def scalars(self):
            return self

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, stmt):
            stmts.append(stmt)
            return Result()

    monkeypatch.setattr(tools_mod, "AsyncSessionLocal", lambda: Session())
    t = tools_mod.AgentTools.__new__(tools_mod.AgentTools)
    set_document_scope([DOC_A])
    await t.get_document_segments({"document_title": "report"})
    await t.get_document_graph_status({"document_title": "report"})
    await t.list_documents({})
    assert len(stmts) == 3
    for stmt in stmts:
        sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "documents.id IN" in sql, sql
        assert DOC_A.replace("-", "") in sql.replace("-", "")


async def test_chunk_lookup_by_hash_is_scoped(monkeypatch):
    from src.agentrag.agent import tools as tools_mod

    seen = {}

    class Store:
        async def get_chunks_by_hashes(self, hashes, filters=None):
            seen["filters"] = filters
            return []

    t = tools_mod.AgentTools.__new__(tools_mod.AgentTools)
    t.es_store = Store()
    set_document_scope([DOC_A])
    set_domain_filter({"system": "cardio"})  # soft filters must not drop the lookup
    await t.get_chunk_by_hash({"content_hashes": ["h"]})
    assert seen["filters"] == {"document_ids": [DOC_A]}


async def test_turn_scope_fails_closed(monkeypatch):
    from src.agentrag.adapter.routers import chat

    async def boom(_nb):
        raise ConnectionError("db")

    async def none_title(_sid):
        return None

    monkeypatch.setattr(chat, "_list_notebook_documents", boom)
    monkeypatch.setattr(chat, "_get_document_title", none_title)
    assert await chat._resolve_turn_scope("nb", None) == []          # lookup error → nothing
    assert await chat._resolve_turn_scope("nb", "lec10") == []       # ... even with a hint
    assert await chat._resolve_turn_scope("nb", None, [DOC_A]) == []  # ticked but deleted
    assert await chat._resolve_turn_scope(None, None) is None        # no notebook → unscoped


async def test_turn_scope_is_document_ids(monkeypatch):
    from src.agentrag.adapter.routers import chat

    async def nb_docs(_nb):
        return [(DOC_A, "report"), (DOC_B, "notes")]

    async def title_of(sid):
        return {DOC_A: "report", DOC_B: "notes"}.get(sid)

    async def titled(title):
        return [DOC_A, DOC_C] if title == "report" else []

    monkeypatch.setattr(chat, "_list_notebook_documents", nb_docs)
    monkeypatch.setattr(chat, "_get_document_title", title_of)
    monkeypatch.setattr(chat, "_document_ids_titled", titled)
    assert await chat._resolve_turn_scope("nb", None) == [DOC_A, DOC_B]
    # A hint narrows within the notebook: DOC_C (same title, other notebook) is out.
    assert await chat._resolve_turn_scope("nb", "report") == [DOC_A]
    assert await chat._resolve_turn_scope("nb", None, [DOC_B, "not-a-uuid"]) == [DOC_B]
    assert await chat._resolve_turn_scope(None, "report") == [DOC_A, DOC_C]


def test_answer_cache_key_includes_scope():
    from src.agentrag.services.answer_cache import AnswerCache

    c = AnswerCache.__new__(AnswerCache)
    k = lambda scope: c.key("q", "v1", None, None, None, "m", document_scope=scope)
    assert k([DOC_A]) != k([DOC_B])
    assert k([DOC_A, DOC_B]) == k([DOC_B, DOC_A])
    assert k(None) != k([])


async def test_filename_hint_never_picks_another_notebooks_document(monkeypatch):
    # "tóm tắt chương 3" in a notebook without such a document used to fall back
    # to a GLOBAL title match: the turn was narrowed to someone else's document.
    from src.agentrag.adapter.routers import chat

    stmts = []

    class Rows:
        def all(self):
            return [] if len(stmts) == 1 else [("Chuong 3 (other user)",)]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, stmt):
            stmts.append(stmt)
            return Rows()

    monkeypatch.setattr(chat, "AsyncSessionLocal", lambda: Session())
    nb = "44444444-4444-4444-8444-444444444444"
    assert await chat._resolve_document_hint("tóm tắt chương 3", nb) is None
    assert len(stmts) == 1  # the notebook query only, no global fallback
    # Outside a notebook (admin routes) the global match still applies.
    stmts.clear()
    stmts.append("pre")  # make the next query return the row
    assert await chat._resolve_document_hint("tóm tắt chương 3", None) == "Chuong 3 (other user)"
