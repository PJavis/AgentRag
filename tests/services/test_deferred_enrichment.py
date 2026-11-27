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
        self.indexed.extend(chunks)


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


async def test_segments_are_contextualized_in_place(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.settings, "CONTEXTUAL_CACHE_DIR", str(tmp_path / "ctx"))
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
    # RAPTOR replaces any earlier tree instead of stacking a second one.
    assert store.deleted_trees == 1
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


async def test_raptor_only_job_uses_indexed_embeddings(tmp_path):
    store, gw = _Store(_leaves()), _Gateway()
    out = await process_enrich_job(_job(tmp_path, contextualize=False), es_store=store,
                                   embedder=_Embedder(), gateway=gw, is_current=_current)
    assert out["status"] == "done" and not store.updates and store.deleted_trees == 1


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
    assert 'defer = defer_enrichment and mode == "async"' in src
    assert src.index("elif want_contextualize and defer:") < src.index("elif want_contextualize:")
    assert src.index("if want_raptor and defer:") < src.index("elif want_raptor:")
    assert '"enrich_segments"' in src


def test_worker_registers_the_job():
    from src.agentrag.worker.settings import WorkerSettings

    assert "enrich_segments" in {f.__name__ for f in WorkerSettings.functions}
