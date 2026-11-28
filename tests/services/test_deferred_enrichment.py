"""An upload is chattable before its contextual retrieval / RAPTOR pass.

Inline, contextualize (one LLM call per chunk) and RAPTOR ran ahead of
"searchable". Uploads now index plain chunks and queue `enrich_segments`,
which rewrites the same segments in place.
"""
from __future__ import annotations

import inspect

from src.agentrag.ingestion import pipeline
from src.agentrag.ingestion.enrich_jobs import EnrichJob, process_enrich_job


class _Store:
    def __init__(self, leaves, missing=0, refetch=None, fail=False):
        self.leaves = leaves
        self.missing = missing
        self.refetch = refetch
        self.fail = fail
        self.fetches = 0
        self.updates: list = []
        self.deleted_trees = 0
        self.indexed: list = []

    async def fetch_leaf_segments(self, document_id):
        if self.fail:
            raise RuntimeError("es down")
        self.fetches += 1
        if self.fetches > 1 and self.refetch is not None:
            return self.refetch
        return [dict(x) for x in self.leaves]

    async def update_segments(self, updates):
        self.updates.extend(updates)
        return self.missing

    async def delete_summary_nodes(self, document_id):
        self.deleted_trees += 1
        return 0

    async def index_segments(self, chunks, title, document_id=None):
        self.indexed.extend((c, document_id) for c in chunks)


class _Embedder:
    def __init__(self):
        self.seen: list[str] = []

    async def embed(self, texts):
        self.seen.extend(texts)
        return [[float(len(t))] for t in texts]


class _Gateway:
    def __init__(self):
        self.calls = 0

    async def text_response(self, system_prompt, user_prompt, task="general"):  # noqa: ANN001
        self.calls += 1
        return "ctx"


def _leaves(n=2):
    return [
        {"_id": f"es{i}", "content": f"c{i}", "content_hash": f"h{i}", "embedding": [0.0]}
        for i in range(n)
    ]


def _job(tmp_path, **kw):
    cache = tmp_path / "doc.enrich.txt"
    cache.write_text("full document text", encoding="utf-8")
    base = dict(document_id="00000000-0000-0000-0000-000000000001", title="T",
                content_hash="H", doc_key="K", parsed_cache_path=str(cache))
    base.update(kw)
    return EnrichJob(**base)


async def _current(job):
    return True


async def _superseded(job):
    return False


class _FakeRaptor:
    nodes = [{"content": "summary", "embedding": [1.0], "node_level": 1}]
    on_build = None

    def __init__(self, gateway, embedder):
        pass

    async def build(self, leaves, title):
        assert all("_id" not in leaf for leaf in leaves)  # ES ids never re-indexed as fields
        if _FakeRaptor.on_build:
            _FakeRaptor.on_build()
        return [dict(n) for n in _FakeRaptor.nodes]


def _patch_raptor(monkeypatch, nodes=None, on_build=None):
    from src.agentrag.ingestion import raptor

    monkeypatch.setattr(_FakeRaptor, "nodes", _FakeRaptor.nodes if nodes is None else nodes)
    monkeypatch.setattr(_FakeRaptor, "on_build", on_build)
    monkeypatch.setattr(raptor, "RaptorBuilder", _FakeRaptor)


async def test_segments_are_contextualized_in_place(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.settings, "CONTEXTUAL_CACHE_DIR", str(tmp_path / "ctx"))
    _patch_raptor(monkeypatch)
    store, emb, gw = _Store(_leaves()), _Embedder(), _Gateway()
    job = _job(tmp_path)
    out = await process_enrich_job(job, es_store=store, embedder=emb, gateway=gw, is_current=_current)

    assert out["status"] == "done" and out["contextualized"] == 2
    # Same ES ids rewritten, with the contextualized text embedded — what an
    # inline ingest would have indexed.
    assert [u[0] for u in store.updates] == ["es0", "es1"]
    assert all(u[1]["context_text"] == "ctx" for u in store.updates)
    assert emb.seen == ["ctx\n\nc0", "ctx\n\nc1"]
    assert store.updates[0][1]["embedding"] == [float(len("ctx\n\nc0"))]
    # RAPTOR replaces any earlier tree instead of stacking a second one, and
    # its nodes carry the document id (delete-by-id must reach them).
    assert store.deleted_trees == 1
    assert [(c["content"], d) for c, d in store.indexed] == [("summary", job.document_id)]
    assert not (tmp_path / "doc.enrich.txt").exists()


async def test_superseded_document_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.settings, "CONTEXTUAL_CACHE_DIR", str(tmp_path / "ctx"))
    store, gw = _Store(_leaves()), _Gateway()
    out = await process_enrich_job(_job(tmp_path), es_store=store, embedder=_Embedder(),
                                   gateway=gw, is_current=_superseded)
    assert out["status"] == "stale"
    assert gw.calls == 0 and not store.updates and store.deleted_trees == 0


async def test_segments_purged_mid_run_stop_before_raptor(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.settings, "CONTEXTUAL_CACHE_DIR", str(tmp_path / "ctx"))
    store = _Store(_leaves(), missing=2)
    out = await process_enrich_job(_job(tmp_path), es_store=store, embedder=_Embedder(),
                                   gateway=_Gateway(), is_current=_current)
    assert out["status"] == "stale" and out["missing"] == 2
    assert store.deleted_trees == 0 and not store.indexed


async def test_reindexed_leaves_stop_raptor(tmp_path, monkeypatch):
    # Same doc re-populated (failed/skeleton path keeps id + hash) → new ES ids.
    _patch_raptor(monkeypatch)
    store = _Store(_leaves(), refetch=[{"_id": "new0"}, {"_id": "new1"}])
    out = await process_enrich_job(_job(tmp_path, contextualize=False), es_store=store,
                                   embedder=_Embedder(), gateway=_Gateway(), is_current=_current)
    assert out["status"] == "stale" and store.deleted_trees == 0


async def test_failure_keeps_document_searchable_and_cleans_cache(tmp_path):
    job = _job(tmp_path)
    out = await process_enrich_job(job, es_store=_Store([], fail=True), embedder=_Embedder(),
                                   gateway=_Gateway(), is_current=_current)
    assert out["status"] == "failed" and "es down" in out["error"]
    assert not (tmp_path / "doc.enrich.txt").exists()


async def test_raptor_only_job_uses_indexed_embeddings(tmp_path, monkeypatch):
    _patch_raptor(monkeypatch)
    store, gw = _Store(_leaves()), _Gateway()
    out = await process_enrich_job(_job(tmp_path, contextualize=False), es_store=store,
                                   embedder=_Embedder(), gateway=gw, is_current=_current)
    assert out["status"] == "done" and not store.updates and store.deleted_trees == 1


async def test_document_deleted_during_raptor_build_gets_no_nodes(tmp_path, monkeypatch):
    # Review finding: nodes written under a dead document_id are never purged.
    state = {"current": True}
    _patch_raptor(monkeypatch, on_build=lambda: state.update(current=False))

    async def is_current(job):
        return state["current"]

    store = _Store(_leaves())
    out = await process_enrich_job(_job(tmp_path, contextualize=False), es_store=store,
                                   embedder=_Embedder(), gateway=_Gateway(), is_current=is_current)
    assert out["status"] == "stale"
    assert not store.indexed and store.deleted_trees == 0


async def test_empty_rebuild_keeps_the_old_tree(tmp_path, monkeypatch):
    _patch_raptor(monkeypatch, nodes=[])
    store = _Store(_leaves())
    out = await process_enrich_job(_job(tmp_path, contextualize=False), es_store=store,
                                   embedder=_Embedder(), gateway=_Gateway(), is_current=_current)
    assert out["status"] == "done" and store.deleted_trees == 0 and not store.indexed


async def test_all_llm_calls_failing_is_not_done(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.settings, "CONTEXTUAL_CACHE_DIR", str(tmp_path / "ctx"))

    class Down:
        async def text_response(self, **kw):
            raise RuntimeError("llm down")

    store = _Store(_leaves())
    out = await process_enrich_job(_job(tmp_path), es_store=store, embedder=_Embedder(),
                                   gateway=Down(), is_current=_current)
    assert out["status"] == "failed" and not store.updates


async def test_stand_in_text_is_not_cached_under_the_file_key(tmp_path, monkeypatch):
    from src.agentrag.ingestion import contextualizer

    seen = {}

    class Spy:
        def __init__(self, gateway):
            pass

        async def contextualize_chunks(self, doc_text, chunks, document_title, doc_key=None):
            seen["doc_key"] = doc_key
            for c in chunks:
                c["context_text"] = "ctx"
            return chunks

    monkeypatch.setattr(contextualizer, "Contextualizer", Spy)
    job = _job(tmp_path, raptor=False, parsed_cache_path=str(tmp_path / "gone.txt"))
    await process_enrich_job(job, es_store=_Store(_leaves()), embedder=_Embedder(),
                             gateway=_Gateway(), is_current=_current)
    assert seen["doc_key"] is None
    await process_enrich_job(_job(tmp_path, raptor=False), es_store=_Store(_leaves()),
                             embedder=_Embedder(), gateway=_Gateway(), is_current=_current)
    assert seen["doc_key"] == "K"


class _Client:
    def __init__(self, response):
        self.response = response
        self.body = None

    async def bulk(self, body, refresh):
        self.body = body
        return self.response


async def test_update_segments_counts_missing_and_raises_on_other_errors():
    import pytest

    from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore

    store = ElasticsearchStore.__new__(ElasticsearchStore)
    store.index_name = "segs"
    store.client = _Client({"errors": True, "items": [
        {"update": {"status": 404, "error": {"type": "document_missing_exception"}}},
        {"update": {"status": 200}},
    ]})
    assert await store.update_segments([("a", {"x": 1}), ("b", {"x": 2})]) == 1
    assert store.client.body[0] == {"update": {"_index": "segs", "_id": "a"}}
    assert store.client.body[1] == {"doc": {"x": 1}}

    store.client = _Client({"errors": True, "items": [
        {"update": {"status": 400, "error": {"type": "mapper_parsing_exception"}}},
    ]})
    with pytest.raises(RuntimeError):
        await store.update_segments([("a", {"x": 1})])


def test_only_uploads_defer():
    # CLI / eval ingest stays inline so a benchmark never queries
    # half-enriched chunks.
    assert inspect.signature(pipeline.ingest_folder).parameters["defer_enrichment"].default is False
    from src.agentrag.ingestion import upload_jobs

    assert "defer_enrichment=settings.INGEST_DEFER_ENRICHMENT" in inspect.getsource(
        upload_jobs.run_upload_ingest
    )


def test_pipeline_skips_inline_work_when_deferring():
    src = inspect.getsource(pipeline.ingest_folder)
    assert 'defer_mode = defer_enrichment and mode == "async"' in src
    assert "defer = defer_mode and source_type not in _IMAGE_SOURCE_TYPES" in src
    assert 'f"{doc_id}.enrich.txt"' in src
    assert src.index("elif want_contextualize and defer:") < src.index("elif want_contextualize:")
    assert src.index("if want_raptor and defer:") < src.index("elif want_raptor:")
    assert "await enqueue_enrich(" in src  # queues enrich_segments (enrich_sweep.py)


def test_worker_registers_the_job():
    from src.agentrag.worker.settings import WorkerSettings

    assert "enrich_segments" in {f.__name__ for f in WorkerSettings.functions}
