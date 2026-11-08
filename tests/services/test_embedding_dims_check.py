"""Startup check: the live embedder must produce vectors the index can search.

docker-compose.yml's `tei` used to serve bge-m3 (1024-dim) under the same
container name as deploy/tei.compose.yml's e5-FT (768-dim). Whichever one was up
silently decided the query space.
"""
from __future__ import annotations

import asyncio

import pytest

from src.agentrag.health.embedding_dims import EmbeddingDimMismatch, verify_embedding_dims


class _Indices:
    def __init__(self, mapping=None, exc=None):
        self._mapping, self._exc = mapping, exc

    async def get_mapping(self, index):
        if self._exc:
            raise self._exc
        return self._mapping


class _ES:
    def __init__(self, mapping=None, exc=None):
        self.indices = _Indices(mapping, exc)


class _Embedder:
    def __init__(self, dim=None, exc=None):
        self._dim, self._exc = dim, exc

    async def embed(self, texts):
        if self._exc:
            raise self._exc
        return [[0.0] * self._dim for _ in texts]


def _mapping(dims, index="agentrag_segments"):
    return {index: {"mappings": {"properties": {"embedding": {"type": "dense_vector", "dims": dims}}}}}


def _run(es, embedder, index="agentrag_segments"):
    return asyncio.run(verify_embedding_dims(es=es, embedder=embedder, index=index))


def test_matching_dims_pass():
    assert _run(_ES(_mapping(768)), _Embedder(768)) == "embedding dims match: 768"


def test_mismatch_raises_with_both_numbers():
    with pytest.raises(EmbeddingDimMismatch) as exc:
        _run(_ES(_mapping(768)), _Embedder(1024))
    assert "1024" in str(exc.value) and "768" in str(exc.value)


def test_missing_index_is_not_an_error():
    from elasticsearch import NotFoundError

    class _NotFound(NotFoundError):
        def __init__(self):
            pass

        def __str__(self):
            return "index_not_found"

    result = _run(_ES(exc=_NotFound()), _Embedder(768))
    assert "no index" in result


def test_unreachable_embedder_is_skipped_not_raised():
    result = _run(_ES(_mapping(768)), _Embedder(exc=ConnectionError("refused")))
    assert result.startswith("skipped")


def test_unreachable_es_is_skipped_not_raised():
    result = _run(_ES(exc=ConnectionError("refused")), _Embedder(768))
    assert result.startswith("skipped")


def test_index_without_embedding_field_is_skipped():
    mapping = {"agentrag_segments": {"mappings": {"properties": {"content": {"type": "text"}}}}}
    assert _run(_ES(mapping), _Embedder(768)).startswith("skipped")
