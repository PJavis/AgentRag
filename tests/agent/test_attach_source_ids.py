"""Citations link to their document so the UI can open the cited page."""
from __future__ import annotations

import inspect

import pytest

from src.agentrag.agent.service import _attach_source_ids


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Session:
    def __init__(self, rows):
        self.rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, _stmt):
        return _Rows(self.rows)


@pytest.fixture
def db(monkeypatch):
    def install(rows):
        import src.agentrag.database as database

        monkeypatch.setattr(database, "AsyncSessionLocal", lambda: _Session(rows))
    return install


async def test_unique_title_gets_id_and_type(db):
    db([("id-1", "Lec 10", "pdf")])
    cites = [{"document_title": "Lec 10", "page_start": 4}]
    await _attach_source_ids(cites)
    assert cites == [{"document_title": "Lec 10", "page_start": 4, "source_id": "id-1", "source_type": "pdf"}]


async def test_ambiguous_title_left_unlinked(db):
    # Used to link to whichever row came last.
    db([("a", "Dup", "pdf"), ("b", "Dup", "docx")])
    cites = [{"document_title": "Dup"}]
    await _attach_source_ids(cites)
    assert "source_id" not in cites[0]


async def test_lookup_failure_leaves_citations_untouched(monkeypatch):
    import src.agentrag.database as database

    class Broken:
        async def __aenter__(self):
            raise ConnectionError("db down")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(database, "AsyncSessionLocal", lambda: Broken())
    cites = [{"document_title": "A"}]
    await _attach_source_ids(cites)
    assert cites == [{"document_title": "A"}]


def test_source_chat_links_citations():
    # _direct_rag bypasses the graph's ground node, where this normally runs.
    from src.agentrag.adapter.routers import chat

    assert "_attach_source_ids(citations)" in inspect.getsource(chat.send_message)


async def test_document_id_links_even_when_the_title_is_shared(db):
    # Two users' "report.pdf": the title is ambiguous, the segment's id is not.
    a = "11111111-1111-4111-8111-111111111111"
    b = "22222222-2222-4222-8222-222222222222"
    db([(a, "report", "pdf"), (b, "report", "md")])
    cites = [{"document_title": "report", "document_id": b}]
    await _attach_source_ids(cites)
    assert cites[0]["source_id"] == b and cites[0]["source_type"] == "md"


async def test_unknown_document_id_stays_unlinked(db):
    db([])
    cites = [{"document_title": "report", "document_id": "33333333-3333-4333-8333-333333333333"}]
    await _attach_source_ids(cites)
    assert "source_id" not in cites[0]
