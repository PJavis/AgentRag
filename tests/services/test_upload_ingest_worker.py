"""UI upload ingest runs as an ARQ job, not inside the api process.

As a FastAPI BackgroundTask, sync parsers blocked the api event loop and an api
restart mid-ingest dropped the job, leaving the document "pending" forever.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from src.agentrag.adapter.routers import sources
from src.agentrag.config import settings
from src.agentrag.ingestion import upload_jobs


class FakePool:
    def __init__(self, fail=False, worker_alive=True):
        self.fail = fail
        self.jobs = []
        self.worker_alive = worker_alive

    async def exists(self, key):
        assert key == "arq:queue:health-check"
        return 1 if self.worker_alive else 0

    async def enqueue_job(self, name, **kw):
        if self.fail:
            raise ConnectionError("redis down")
        self.jobs.append((name, kw))
        return SimpleNamespace(job_id="j")


@pytest.fixture
def pool(monkeypatch):
    p = FakePool()
    import src.agentrag.worker.pool as pool_mod

    monkeypatch.setattr(pool_mod, "get_pool", lambda: p)
    monkeypatch.setattr(settings, "UPLOAD_INGEST_VIA_WORKER", True)
    return p


async def test_enqueue_hands_absolute_path_to_worker(pool, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ok = await sources._enqueue_upload_ingest("stage/x/a.pdf", "u1", ["nb"], "h", "sk")
    assert ok is True
    [(name, kw)] = pool.jobs
    assert name == "upload_ingest"
    assert kw == {
        "file_path": str(tmp_path / "stage/x/a.pdf"),
        "user_id": "u1", "notebook_ids": ["nb"], "content_hash": "h", "skeleton_id": "sk",
    }


async def test_disabled_flag_means_in_process(pool, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_INGEST_VIA_WORKER", False)
    assert await sources._enqueue_upload_ingest("a", "u", [], "h", "s") is False
    assert pool.jobs == []


async def test_enqueue_failure_falls_back_in_process(monkeypatch):
    import src.agentrag.worker.pool as pool_mod

    monkeypatch.setattr(pool_mod, "get_pool", lambda: FakePool(fail=True))
    monkeypatch.setattr(settings, "UPLOAD_INGEST_VIA_WORKER", True)
    assert await sources._enqueue_upload_ingest("a", "u", [], "h", "s") is False


async def test_enqueue_returning_none_falls_back(monkeypatch):
    # ARQ returns None instead of a Job when it refuses the enqueue.
    class NonePool:
        async def exists(self, key):
            return 1

        async def enqueue_job(self, name, **kw):
            return None

    import src.agentrag.worker.pool as pool_mod

    monkeypatch.setattr(pool_mod, "get_pool", lambda: NonePool())
    monkeypatch.setattr(settings, "UPLOAD_INGEST_VIA_WORKER", True)
    assert await sources._enqueue_upload_ingest("a", "u", [], "h", "s") is False


async def test_missing_staged_file_marks_skeleton_failed(monkeypatch, tmp_path):
    # The classic shared-volume bug: api staged the file somewhere the worker
    # cannot see. The doc must go "failed", not sit "pending" forever.
    marked = {}
    events = []

    async def fake_mark(doc_id, error):
        marked[doc_id] = error

    async def fake_event(**kw):
        events.append(kw)

    import src.agentrag.observability.activity as activity

    monkeypatch.setattr(upload_jobs, "_mark_failed", fake_mark)
    monkeypatch.setattr(activity, "record_event", fake_event)

    async def exists(_sid):
        return True

    monkeypatch.setattr(upload_jobs, "_skeleton_exists", exists)
    result = await upload_jobs.run_upload_ingest(
        str(tmp_path / "gone" / "a.pdf"), "u1", [], None, skeleton_id="sk-1"
    )
    assert result is None
    assert "staged upload missing" in marked["sk-1"]
    assert events[0]["event_type"] == "ingest_failed"
    assert events[0]["target_id"] == "sk-1"


async def test_success_links_notebooks_and_cleans_staging(monkeypatch, tmp_path):
    stage = tmp_path / "upload_1"
    stage.mkdir()
    (stage / "a.pdf").write_bytes(b"%PDF")
    doc_id = uuid.uuid4()
    doc = SimpleNamespace(id=doc_id, content_hash=None)
    linked = {}
    events = []

    async def fake_ingest(folder, user_id):
        assert folder == str(stage)
        return {"documents": [{"document_id": str(doc_id), "segments_indexed": 4}]}

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, _m, key):
            return doc if key == doc_id else None

        async def commit(self):
            linked["committed"] = True

    async def fake_link(session, did, nbs):
        linked["args"] = (did, nbs)

    async def fake_event(**kw):
        events.append(kw)

    import src.agentrag.ingestion.pipeline as pipeline
    import src.agentrag.observability.activity as activity

    monkeypatch.setattr(pipeline, "ingest_folder", fake_ingest)
    monkeypatch.setattr(upload_jobs, "AsyncSessionLocal", Session)
    monkeypatch.setattr(upload_jobs, "link_to_notebooks", fake_link)
    monkeypatch.setattr(activity, "record_event", fake_event)

    out = await upload_jobs.run_upload_ingest(str(stage / "a.pdf"), "u", ["nb1"], "hash")
    assert out == str(doc_id)
    assert linked == {"args": (doc_id, ["nb1"]), "committed": True}
    assert doc.content_hash == "hash"
    assert not stage.exists()
    assert events[0]["event_type"] == "ingest_done"
    assert events[0]["payload"]["segment_count"] == 4


def test_worker_registers_upload_ingest_and_installs_pool():
    import inspect

    from src.agentrag.worker import settings as ws
    from src.agentrag.worker.functions import upload_ingest

    assert upload_ingest in ws.WorkerSettings.functions
    # ingest_folder enqueues graph_ingest via get_pool(); without this the
    # worker raises "ARQ pool not initialized" on every async-mode upload.
    assert "set_pool(ctx[\"redis\"])" in inspect.getsource(ws._on_startup)


def test_staging_dir_is_on_the_shared_cache_volume():
    # docker-compose mounts structmem_cache at /app/.cache on api AND worker.
    assert settings.UPLOAD_STAGING_DIR.startswith(".cache/")


async def test_no_live_worker_means_in_process(monkeypatch):
    # Redis up, no worker consuming: the job would sit queued and the doc
    # "pending" forever.
    import src.agentrag.worker.pool as pool_mod

    p = FakePool(worker_alive=False)
    monkeypatch.setattr(pool_mod, "get_pool", lambda: p)
    monkeypatch.setattr(settings, "UPLOAD_INGEST_VIA_WORKER", True)
    assert await sources._enqueue_upload_ingest("a", "u", [], "h", "s") is False
    assert p.jobs == []


async def test_timeout_cancellation_marks_failed_and_keeps_staging(monkeypatch, tmp_path):
    import asyncio

    stage = tmp_path / "upload_9"
    stage.mkdir()
    (stage / "a.pdf").write_bytes(b"%PDF")
    marked, events = {}, []

    async def slow_ingest(folder, user_id):
        raise asyncio.CancelledError()

    async def fake_mark(doc_id, error):
        marked[doc_id] = error

    async def fake_event(**kw):
        events.append(kw)

    async def exists(_sid):
        return True

    import src.agentrag.ingestion.pipeline as pipeline
    import src.agentrag.observability.activity as activity

    monkeypatch.setattr(pipeline, "ingest_folder", slow_ingest)
    monkeypatch.setattr(upload_jobs, "_mark_failed", fake_mark)
    monkeypatch.setattr(upload_jobs, "_skeleton_exists", exists)
    monkeypatch.setattr(activity, "record_event", fake_event)

    with pytest.raises(asyncio.CancelledError):
        await upload_jobs.run_upload_ingest(str(stage / "a.pdf"), "u", [], None, skeleton_id="sk-9")
    assert "cancelled" in marked["sk-9"]
    assert events[0]["event_type"] == "ingest_failed"  # was "ingest_done"
    assert stage.exists()  # ARQ retries the job; the file must still be there


async def test_skeleton_deleted_while_queued_is_not_resurrected(monkeypatch, tmp_path):
    stage = tmp_path / "upload_8"
    stage.mkdir()
    (stage / "a.pdf").write_bytes(b"%PDF")
    called = []

    async def fake_ingest(folder, user_id):
        called.append(folder)
        return {"documents": []}

    async def gone(_sid):
        return False

    async def fake_event(**kw):
        pass

    import src.agentrag.ingestion.pipeline as pipeline
    import src.agentrag.observability.activity as activity

    monkeypatch.setattr(pipeline, "ingest_folder", fake_ingest)
    monkeypatch.setattr(upload_jobs, "_skeleton_exists", gone)
    monkeypatch.setattr(activity, "record_event", fake_event)

    assert await upload_jobs.run_upload_ingest(str(stage / "a.pdf"), "u", [], None, skeleton_id="sk-8") is None
    assert called == []
    assert not stage.exists()


def test_worker_refreshes_liveness_key_often():
    from src.agentrag.worker import settings as ws

    assert ws.WorkerSettings.health_check_interval <= 120


async def test_new_layout_ingests_the_staging_root(monkeypatch, tmp_path):
    # <staging>/<doc id>/<filename>: ingesting the root makes source_id
    # "<doc id>/<filename>", the key the skeleton was created with. A bare
    # filename source_id was shared by every user's "a.pdf".
    doc_id = uuid.uuid4()
    stage = tmp_path / "upload_2"
    (stage / str(doc_id)).mkdir(parents=True)
    (stage / str(doc_id) / "a.pdf").write_bytes(b"%PDF")
    seen = {}

    async def fake_ingest(folder, user_id):
        seen["folder"] = folder
        from src.agentrag.ingestion.connectors.folder import FolderConnector

        seen["source_ids"] = [d["source_id"] for d in FolderConnector(folder).list_documents()]
        seen["titles"] = [d["title"] for d in FolderConnector(folder).list_documents()]
        return {"documents": []}

    async def exists(_sid):
        return True

    async def fake_mark(*a):
        return None

    async def fake_event(**kw):
        return None

    import src.agentrag.ingestion.pipeline as pipeline
    import src.agentrag.observability.activity as activity

    monkeypatch.setattr(pipeline, "ingest_folder", fake_ingest)
    monkeypatch.setattr(upload_jobs, "_skeleton_exists", exists)
    monkeypatch.setattr(upload_jobs, "_mark_failed", fake_mark)
    monkeypatch.setattr(activity, "record_event", fake_event)
    await upload_jobs.run_upload_ingest(
        str(stage / str(doc_id) / "a.pdf"), "u", [], None, skeleton_id=str(doc_id)
    )
    assert seen["folder"] == str(stage)
    assert seen["source_ids"] == [f"{doc_id}/a.pdf"]
    assert seen["titles"] == ["a"]  # the title shown to the user is unchanged
    assert not stage.exists()  # the whole staging dir is cleaned, not just the doc dir
