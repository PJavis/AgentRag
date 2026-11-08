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


@pytest.fixture(autouse=True)
def _reset():
    set_domain_filter(None)
    set_document_scope(None)
    yield
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
    set_document_scope(["quasar-manual"])
    await a._inject_raw_query_hits("q?", [])
    assert seen["filters"] == {"document_titles": ["quasar-manual"]}


def test_empty_scope_is_kept_not_coerced_to_none():
    set_document_scope([])
    assert current_retrieval_filters() == {"document_titles": []}
    set_document_scope(None)
    assert current_retrieval_filters() is None


def test_empty_scope_matches_nothing_in_es():
    assert _tag_filter_clauses({"document_titles": []}) == [{"bool": {"must_not": [{"match_all": {}}]}}]
    assert _tag_filter_clauses({"document_titles": ["a"]}) == [{"terms": {"document_title.keyword": ["a"]}}]
    assert _tag_filter_clauses({"systems": ["x"]}) == [{"terms": {"system_tag": ["x"]}}]


def test_tools_use_the_shared_filters():
    from src.agentrag.agent.tools import AgentTools

    set_document_scope(["d"])
    set_domain_filter({"system": "cardio"})
    tools = AgentTools.__new__(AgentTools)
    assert tools._current_filters() == {"systems": ["cardio"], "document_titles": ["d"]}
