import asyncio
from src.agentrag.ingestion.pipeline import _build_and_index_raptor


class _Store:
    def __init__(self):
        self.indexed = None

    async def index_segments(self, chunks, title):  # noqa: ANN001
        self.indexed = chunks


class _Builder:
    async def build(self, leaves, title):  # noqa: ANN001
        return [{"content": "S", "content_hash": "s1", "embedding": [1.0],
                 "node_level": 1, "segment_type": "raptor_summary"}]


def test_indexes_summary_nodes_when_builder_returns_them():
    store = _Store()
    leaves = [{"content": "x", "embedding": [1.0]}]
    asyncio.run(_build_and_index_raptor(_Builder(), store, leaves, "Doc"))
    assert store.indexed and store.indexed[0]["segment_type"] == "raptor_summary"


def test_no_index_when_builder_returns_empty():
    store = _Store()

    class _Empty:
        async def build(self, leaves, title):  # noqa: ANN001
            return []

    asyncio.run(_build_and_index_raptor(_Empty(), store, [{"content": "x"}], "Doc"))
    assert store.indexed is None
