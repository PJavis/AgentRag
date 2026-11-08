"""A big upload must not cost a corpus pass of LLM calls or look stuck.

A 5.9k-row workbook became 4,618 search chunks; contextual retrieval made one
DeepSeek call per chunk, 4 at a time (~1 h), while the upload sat on
"pending" with no progress, and RAPTOR would have summarised the table rows.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

from src.agentrag.ingestion import pipeline
from src.agentrag.ingestion.contextualizer import Contextualizer


def test_spreadsheets_skip_contextualize(monkeypatch):
    monkeypatch.setattr(pipeline.settings, "CONTEXTUAL_MAX_CHUNKS", 1500)
    assert pipeline._contextualize_skip_reason("excel", 10) == "spreadsheet"
    assert pipeline._contextualize_skip_reason("csv", 10) == "spreadsheet"


def test_chunk_cap_skips_only_above_the_limit(monkeypatch):
    monkeypatch.setattr(pipeline.settings, "CONTEXTUAL_MAX_CHUNKS", 1500)
    assert pipeline._contextualize_skip_reason("pdf", 1500) is None
    assert "1501 chunks" in pipeline._contextualize_skip_reason("pdf", 1501)
    monkeypatch.setattr(pipeline.settings, "CONTEXTUAL_MAX_CHUNKS", 0)  # 0 = no cap
    assert pipeline._contextualize_skip_reason("pdf", 10**6) is None


def test_raptor_is_not_built_for_spreadsheets():
    src = inspect.getsource(pipeline.ingest_folder)
    assert "settings.RAPTOR_ENABLED and source_type not in _EXCEL_SOURCE_TYPES" in src


class _SlowGateway:
    def __init__(self):
        self.active = 0
        self.peak = 0

    async def text_response(self, system_prompt, user_prompt, task="general"):  # noqa: ANN001
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return "ctx"


def test_concurrency_comes_from_the_setting(tmp_path, monkeypatch):
    from src.agentrag.ingestion import contextualizer as mod

    monkeypatch.setattr(mod.settings, "CONTEXTUAL_CONCURRENCY", 16)
    gw = _SlowGateway()
    chunks = [{"content": f"c{i}", "content_hash": f"h{i}"} for i in range(40)]
    asyncio.run(Contextualizer(gw, cache_dir=str(tmp_path)).contextualize_chunks("doc", chunks, "T"))
    assert gw.peak == 16  # was hard-wired to max(EMBEDDING_BATCH_SIZE // 4, 4)


def test_progress_is_reported_and_ends_at_total(tmp_path, monkeypatch):
    from src.agentrag.ingestion import contextualizer as mod

    monkeypatch.setattr(mod.settings, "CONTEXTUAL_CONCURRENCY", 8)
    seen: list[tuple[int, int]] = []

    async def on_progress(done, total):
        seen.append((done, total))

    chunks = [{"content": f"c{i}", "content_hash": f"h{i}"} for i in range(30)]
    asyncio.run(
        Contextualizer(_SlowGateway(), cache_dir=str(tmp_path)).contextualize_chunks(
            "doc", chunks, "T", on_progress=on_progress
        )
    )
    assert seen and seen[-1] == (30, 30)
    assert all(t == 30 for _, t in seen)


def test_a_failing_progress_callback_does_not_break_ingest(tmp_path):
    async def boom(done, total):
        raise RuntimeError("db down")

    chunks = [{"content": "c", "content_hash": "h"}]
    out = asyncio.run(
        Contextualizer(_SlowGateway(), cache_dir=str(tmp_path)).contextualize_chunks(
            "doc", chunks, "T", on_progress=boom
        )
    )
    assert out[0]["context_text"] == "ctx"


def _doc(**kw):
    base = dict(id="d", title="t", source_type="excel", graph_status="pending",
                graph_total_chunks=0, graph_processed_chunks=0, graph_failed_chunks=0,
                parse_total_pages=0, parse_done_pages=0, graph_last_error=None,
                created_at=None, updated_at=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_activity_panel_shows_real_pipeline_stages():
    # It filtered on "processing"/"queued", which nothing sets: a document being
    # parsed or enriched vanished from the panel while it was working.
    from src.agentrag.adapter.routers import activity

    row = activity._doc_progress_row(_doc(graph_status="parsing", parse_total_pages=200, parse_done_pages=50))
    assert row["graph_status"] == "processing" and row["stage"] == "parsing"
    assert (row["chunks_done"], row["chunks_total"], row["progress_pct"]) == (50, 200, 25.0)
    row = activity._doc_progress_row(_doc(graph_status="enriching", graph_total_chunks=10, graph_processed_chunks=5))
    assert row["graph_status"] == "processing" and row["progress_pct"] == 50.0
    assert activity._doc_progress_row(_doc(graph_status="done_partial"))["graph_status"] == "done"
    for stage in ("pending", "parsing", "searchable", "enriching", "failed"):
        assert stage in activity._ACTIVE_STAGES


def test_spreadsheets_skip_structmem_extraction():
    # 1,163 extract calls for one workbook; the doc goes straight to done.
    src = inspect.getsource(pipeline.ingest_folder)
    assert "settings.STRUCTMEM_SKIP_SPREADSHEETS and source_type in _EXCEL_SOURCE_TYPES" in src
    assert 'if mode == "async" and not skip_graph:' in src
    assert src.index("if skip_graph:") < src.index('elif mode == "sync":')


async def test_a_skeleton_showing_progress_is_still_populated():
    # Progress flips the upload's skeleton to "parsing". The save step only
    # recognised "pending" skeletons, so it answered "skipped" and nothing was
    # indexed (caught live: job done in 29 s, 0 segments).
    from src.agentrag.ingestion.stores.postgres_store import PostgresStore

    skeleton = SimpleNamespace(id="d1", content_hash="h", graph_status="parsing",
                               title="t", source_type="excel")

    class Result:
        def __init__(self, docs=None):
            self.docs = docs

        def scalars(self):
            return self

        def all(self):
            return self.docs

        def scalar_one_or_none(self):
            return None  # no segments yet

    class Session:
        def __init__(self):
            self.n = 0
            self.added = []

        async def execute(self, stmt):
            self.n += 1
            return Result([skeleton] if self.n == 1 else None)

        async def flush(self):
            pass

        async def commit(self):
            pass

        def add(self, obj):
            self.added.append(obj)

    session = Session()
    doc_id, status = await PostgresStore().save_document_and_segments(
        session,
        {"source_id": "d1/x.xlsx", "content_hash": "h", "title": "t", "source_type": "excel"},
        [{"content": "c", "content_hash": "c1", "segment_type": "text",
          "section_path": "", "position": 0, "metadata": {}}],
        project_id="p",
    )
    assert (doc_id, status) == ("d1", "ingested")
    assert len(session.added) == 1


async def test_an_interrupted_run_is_repopulated_on_retry():
    # Cut off after the save committed (row pending/parsing WITH segments): the
    # ARQ retry must re-index, not answer "skipped" and leave the row stuck.
    from src.agentrag.ingestion.stores.postgres_store import PostgresStore

    row = SimpleNamespace(id="d1", content_hash="h", graph_status="parsing",
                          title="t", source_type="excel")

    class Result:
        def scalars(self):
            return self

        def all(self):
            return [row]

        def scalar_one_or_none(self):
            return object()  # segments already exist

    class Session:
        added: list = []

        async def execute(self, stmt):
            return Result()

        async def flush(self):
            pass

        async def commit(self):
            pass

        def add(self, obj):
            self.added.append(obj)

    _, status = await PostgresStore().save_document_and_segments(
        Session(),
        {"source_id": "d1/x.xlsx", "content_hash": "h", "title": "t", "source_type": "excel"},
        [{"content": "c", "content_hash": "c1", "segment_type": "text",
          "section_path": "", "position": 0, "metadata": {}}],
        project_id="p",
    )
    assert status == "ingested"


async def test_a_finished_document_is_still_skipped():
    from src.agentrag.ingestion.stores.postgres_store import PostgresStore

    row = SimpleNamespace(id="d1", content_hash="h", graph_status="done", title="t", source_type="pdf")

    class Result:
        def scalars(self):
            return self

        def all(self):
            return [row]

    class Session:
        async def execute(self, stmt):
            return Result()

    _, status = await PostgresStore().save_document_and_segments(
        Session(), {"source_id": "a.pdf", "content_hash": "h"}, [], project_id="p"
    )
    assert status == "skipped"
