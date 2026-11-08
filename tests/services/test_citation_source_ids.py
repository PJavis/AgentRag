"""Citations carry the id of their document so the UI can open the cited page.

ES segments only hold `document_title`; without this every [n] badge was
unclickable (`source_id` was never set).
"""
from __future__ import annotations

from src.agentrag.chat.turns import attach_source_ids


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Session:
    def __init__(self, rows):
        self.rows = rows
        self.queries = 0

    async def execute(self, _stmt):
        self.queries += 1
        return _Rows(self.rows)


async def test_unique_title_gets_source_id():
    s = _Session([("Lec 10", "id-1")])
    out = await attach_source_ids(s, [{"document_title": "Lec 10", "page_start": 4}])
    assert out == [{"document_title": "Lec 10", "page_start": 4, "source_id": "id-1"}]


async def test_ambiguous_title_is_left_unlinked():
    s = _Session([("Dup", "a"), ("Dup", "b")])
    out = await attach_source_ids(s, [{"document_title": "Dup"}])
    assert "source_id" not in out[0]


async def test_existing_source_id_kept_and_no_query():
    s = _Session([("T", "other")])
    out = await attach_source_ids(s, [{"document_title": "T", "source_id": "mine"}])
    assert out[0]["source_id"] == "mine"
    assert s.queries == 0


async def test_one_query_for_many_citations():
    s = _Session([("A", "1"), ("B", "2")])
    out = await attach_source_ids(s, [{"document_title": "A"}, {"document_title": "B"}, {"document_title": "A"}])
    assert [c["source_id"] for c in out] == ["1", "2", "1"]
    assert s.queries == 1


async def test_lookup_failure_returns_citations_unchanged():
    class Broken:
        async def execute(self, _stmt):
            raise ConnectionError("db down")

    cites = [{"document_title": "A"}]
    assert await attach_source_ids(Broken(), cites) == cites


def test_append_message_links_citations():
    import inspect

    from src.agentrag.chat.history import ConversationStore

    assert "attach_source_ids" in inspect.getsource(ConversationStore.append_message)
