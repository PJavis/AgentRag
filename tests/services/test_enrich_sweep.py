"""Sweep for deferred enrichment that never landed (enrich_sweep.py).

Live 2026-11-28 dry run: 43 of 121 documents had no context_text on any leaf
and/or no RAPTOR node — the corpus was ingested 2026-09-07, the day the
DeepSeek balance ran out. Nothing re-ran those; the sweep is the retry path.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.agentrag.ingestion import enrich_sweep as sw


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    monkeypatch.setattr(sw.settings, "CONTEXTUAL_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(sw.settings, "RAPTOR_ENABLED", True)
    monkeypatch.setattr(sw.settings, "RAPTOR_MIN_LEAVES", 8)
    monkeypatch.setattr(sw.settings, "CONTEXTUAL_MAX_CHUNKS", 1500)


def test_needs_mirrors_ingest_eligibility():
    assert sw.needs("pdf", {"leaves": 20, "contextualized": 0, "summaries": 0}) == (True, True)
    # partly contextualized: the job counts that as success — leave it
    assert sw.needs("pdf", {"leaves": 20, "contextualized": 3, "summaries": 5}) == (False, False)
    # too small for RAPTOR
    assert sw.needs("pdf", {"leaves": 4, "contextualized": 0, "summaries": 0}) == (True, False)
    # spreadsheets never get either; images never get either
    assert sw.needs("excel", {"leaves": 50, "contextualized": 0, "summaries": 0}) == (False, False)
    assert sw.needs("image", {"leaves": 50, "contextualized": 0, "summaries": 0}) == (False, False)
    # over the contextualize cap
    assert sw.needs("pdf", {"leaves": 2000, "contextualized": 0, "summaries": 1}) == (False, False)
    # the ingest cap counted image chunks too
    assert sw.needs("pdf", {"leaves": 1400, "images": 200, "contextualized": 0, "summaries": 1}) == (False, False)


def test_needs_respects_disabled_features(monkeypatch):
    monkeypatch.setattr(sw.settings, "CONTEXTUAL_RETRIEVAL_ENABLED", False)
    monkeypatch.setattr(sw.settings, "RAPTOR_ENABLED", False)
    assert sw.needs("pdf", {"leaves": 20, "contextualized": 0, "summaries": 0}) == (False, False)


class _Store:
    def __init__(self, counts):
        self.counts = counts
        self.asked: list[str] = []

    async def enrichment_counts(self, ids):
        self.asked += ids
        return {i: self.counts[i] for i in ids if i in self.counts}


def _doc(i, status="failed", st="pdf"):
    return {"id": i, "title": f"t{i}", "source_type": st, "content_hash": f"h{i}", "graph_status": status}


def test_find_candidates_keeps_graph_failed_skips_in_flight():
    # graph_status "failed" is the StructMem graph step: segments are indexed
    # and it must be swept. pending/parsing are still being ingested.
    store = _Store({
        "a": {"leaves": 20, "contextualized": 0, "summaries": 0},
        "b": {"leaves": 20, "contextualized": 20, "summaries": 4},
        "c": {"leaves": 20, "contextualized": 0, "summaries": 0},
    })
    docs = [_doc("a"), _doc("b", "done"), _doc("c", "parsing"), _doc("d", "done")]
    got = asyncio.run(sw.find_candidates(documents=docs, es_store=store))
    assert [(c.document_id, c.contextualize, c.raptor) for c in got] == [("a", True, True)]
    assert "c" not in store.asked
    only = asyncio.run(sw.find_candidates(documents=docs, es_store=store, only="raptor"))
    assert [(c.contextualize, c.raptor) for c in only] == [(False, True)]


class _Pool:
    """ARQ semantics: an id is refused while its job is queued/running OR its
    result is kept; `delete` drops a kept result."""

    def __init__(self):
        self.active: set[str] = set()
        self.results: set[str] = set()
        self.calls: list[dict] = []

    async def delete(self, key):
        from arq.constants import result_key_prefix

        self.results.discard(key.removeprefix(result_key_prefix))

    async def enqueue_job(self, name, **kw):
        assert name == "enrich_segments"
        self.calls.append(kw)
        jid = kw["_job_id"]
        if jid in self.active or jid in self.results:
            return None
        self.active.add(jid)
        return object()


def test_enqueue_dedupes_on_the_pipeline_job_id():
    pool = _Pool()
    # the upload's job is still queued under the shared id
    pool.active.add(sw.enrich_job_id("a", "ha"))
    cands = [sw.Candidate("a", "ta", "ha", True, True, 20), sw.Candidate("b", "tb", "hb", True, False, 9)]
    assert asyncio.run(sw.enqueue(cands, pool)) == (1, 1)
    sent = pool.calls[1]
    assert sent["document_id"] == "b" and sent["contextualize"] is True and sent["raptor"] is False


def test_a_kept_result_does_not_block_a_needed_rerun():
    # Review of this change: a failed run (or one whose segments a same-row
    # re-ingest replaced) keeps its result for keep_result=1h; ARQ would
    # refuse the id and nothing would be queued.
    pool = _Pool()
    pool.results.add(sw.enrich_job_id("a", "ha"))
    assert asyncio.run(sw.enqueue_enrich(pool, document_id="a", content_hash="ha", title="t")) is True


def test_pipeline_queues_through_the_shared_helper(tmp_path, monkeypatch):
    """Drive the real pipeline enqueue block: it must go through
    enqueue_enrich (shared id, stale result dropped) and report a refused
    enqueue honestly."""
    src = Path("src/agentrag/ingestion/pipeline.py").read_text(encoding="utf-8")
    assert "await enqueue_enrich(" in src
    assert '"queued" if queued else "already_queued"' in src
    assert '"enrich_segments",' not in src.split("from .enrich_sweep import enqueue_enrich")[1].split("except Exception")[0]


def test_enrichment_counts_query_and_parse():
    from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore

    store = ElasticsearchStore.__new__(ElasticsearchStore)
    store.index_name = "idx"
    seen = {}

    class _Client:
        async def search(self, index, **body):
            seen.update(body, index=index)
            return {"aggregations": {"docs": {"buckets": [{
                "key": "a", "leaves": {"doc_count": 9}, "images": {"doc_count": 2},
                "contextualized": {"doc_count": 0}, "summaries": {"doc_count": 0},
            }]}}}

    store.client = _Client()
    assert asyncio.run(store.enrichment_counts([])) == {}
    out = asyncio.run(store.enrichment_counts(["a", "b"]))
    assert out == {"a": {"leaves": 9, "images": 2, "contextualized": 0, "summaries": 0}}
    assert seen["index"] == "idx" and seen["size"] == 0
    assert seen["query"] == {"terms": {"document_id": ["a", "b"]}}
    aggs = seen["aggs"]["docs"]["aggs"]
    leaf = aggs["leaves"]["filter"]["bool"]["must_not"]
    # old segments without node_level still count as leaves (must_not range)
    assert {"range": {"node_level": {"gt": 0}}} in leaf and {"term": {"segment_type": "image"}} in leaf
    assert {"exists": {"field": "context_text"}} in aggs["contextualized"]["filter"]["bool"]["filter"]


def test_apply_refuses_over_the_call_budget(monkeypatch, capsys):
    big = [sw.Candidate("a", "ta", "ha", True, False, 600)]

    async def fake_find(**_kw):
        return big

    async def no_close():
        return None

    monkeypatch.setattr(sw, "find_candidates", fake_find)
    import src.agentrag.ingestion.stores.elasticsearch_store as es

    monkeypatch.setattr(es, "close_shared_es_client", no_close)

    def boom(*_a, **_k):
        raise AssertionError("must not touch the queue")

    import src.agentrag.worker.pool as pool_mod

    monkeypatch.setattr(pool_mod, "init_pool", boom)
    assert asyncio.run(sw._main(["--apply", "--max-calls", "500"])) == 2
    assert "refusing" in capsys.readouterr().out
    # dry run never queues either
    assert asyncio.run(sw._main([])) == 0
