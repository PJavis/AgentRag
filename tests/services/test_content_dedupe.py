"""One content key for upload and ingest, and folder ingest dedupes by content.

Found 2026-11-28: 122 documents, 35 distinct contents.
- Upload dedupe hashed the raw bytes; ingest stores sha256(bytes + parse
  fingerprint) for PDFs (since 2026-09-06), so the lookup never matched a PDF.
  The same mismatch on the upload skeleton row made the ingest delete the
  skeleton and mint a new document id.
- Folder ingest deduped by source_id only; uploads live in data/originals as
  <uuid>.pdf, so re-ingesting that folder made one document per copy.
"""
from __future__ import annotations

import inspect

import pytest

from src.agentrag.ingestion.connectors import folder
from src.agentrag.ingestion import pipeline, upload_jobs


@pytest.mark.parametrize("non_default_parse", [False, True])
def test_upload_key_equals_ingest_key(tmp_path, monkeypatch, non_default_parse):
    from src.agentrag.config import Settings

    default = Settings.model_fields["PDF_PRESERVE_TABLES"].default
    monkeypatch.setattr(folder.settings, "PDF_PRESERVE_TABLES", (not default) if non_default_parse else default)
    data = b"%PDF-1.4 fake bytes"
    f = tmp_path / "x.pdf"
    f.write_bytes(data)
    key = folder.content_cache_key(data, ".pdf")
    assert key == folder._document_cache_key(f, ".pdf")
    assert folder.content_cache_key(data, ".PDF") == key
    listed = folder.FolderConnector(str(tmp_path)).list_documents()
    assert listed[0]["content_hash"] == key


def test_non_pdf_key_is_plain_sha(monkeypatch):
    import hashlib

    monkeypatch.setattr(folder.settings, "PDF_PRESERVE_TABLES", True)
    assert folder.content_cache_key(b"# note", ".md") == hashlib.sha256(b"# note").hexdigest()


def test_upload_route_uses_the_ingest_key():
    from src.agentrag.adapter.routers import sources

    src = inspect.getsource(sources)
    assert "content_hash = content_cache_key(raw, os.path.splitext(filename)[1])" in src
    # dedupe looks up the raw-bytes hash, owner-scoped
    assert "find_existing_document(bytes_sha, owner_uuid)" in src
    assert "source_bytes_sha=bytes_sha" in src


def test_folder_ingest_skips_content_duplicates_before_parsing():
    src = inspect.getsource(pipeline.ingest_folder)
    assert inspect.signature(pipeline.ingest_folder).parameters["skip_content_duplicates"].default is True
    skip = src.index('report["status"] = "duplicate"')
    # before any parse / progress write: a duplicate costs nothing
    assert skip < src.index("await _mark_ingest_progress(doc[\"source_id\"], 0, 0)")
    assert skip < src.index("pdf_parser.parse(file_path)")


def test_uploads_opt_out_of_folder_content_dedupe():
    # the skeleton row holds the same content under the same source_id; the
    # upload's own UPLOAD_DEDUPE_BY_HASH already decided to ingest
    assert "skip_content_duplicates=False" in inspect.getsource(upload_jobs.run_upload_ingest)


# ── SQL actually issued (captured from a fake session) ────────────────────

class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _Session:
    def __init__(self, results):
        self.results = list(results)
        self.sql: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt):
        from sqlalchemy.dialects import postgresql

        self.sql.append(str(stmt.compile(dialect=postgresql.dialect(),
                                         compile_kwargs={"literal_binds": True})))
        return _Result(self.results.pop(0))


def _patch_session(monkeypatch, module, results):
    sess = _Session(results)
    monkeypatch.setattr(module, "AsyncSessionLocal", lambda: sess)
    return sess


def test_upload_dedupe_is_owner_scoped_and_needs_segments(monkeypatch):
    import asyncio
    import uuid as _uuid

    from src.agentrag.adapter import upload_dedupe

    owner = _uuid.uuid4()
    sess = _patch_session(monkeypatch, upload_dedupe, [None, None])
    asyncio.run(upload_dedupe.find_existing_document("abc", owner))
    asyncio.run(upload_dedupe.find_existing_document("abc", None))
    mine, anon = sess.sql
    assert "documents.source_bytes_sha = 'abc'" in mine
    assert f"documents.user_id = '{owner}'" in mine  # never another user's file
    assert "documents.user_id IS NULL" in anon
    assert "EXISTS" in mine and "segments.document_id = documents.id" in mine
    assert "documents.content_hash =" not in mine  # parse-settings dependent: never matched PDFs


def test_folder_dedupe_skips_only_new_source_ids_with_a_shared_copy(monkeypatch):
    import asyncio
    import uuid as _uuid

    existing = _uuid.uuid4()
    # source_id already known → replace path, never "duplicate" (an edited
    # file must not keep serving its old content)
    sess = _patch_session(monkeypatch, pipeline, [existing])
    assert asyncio.run(pipeline._indexed_duplicate_of("abc", "a.md")) is None
    assert len(sess.sql) == 1
    # new source_id, shared copy with segments → duplicate of it
    sess = _patch_session(monkeypatch, pipeline, [None, existing])
    assert asyncio.run(pipeline._indexed_duplicate_of("abc", "copy.pdf")) == str(existing)
    q = sess.sql[1]
    assert "documents.source_bytes_sha = 'abc'" in q and "documents.user_id IS NULL" in q
    assert "segments.document_id = documents.id" in q
    assert asyncio.run(pipeline._indexed_duplicate_of(None, "x.pdf")) is None


# ── corpus cleanup plan ─────────────────────────────────────────────────────

def _d(i, sha="s1", user=None, status="done"):
    return {"id": i, "title": i, "source_id": f"{i}.pdf", "user_id": user,
            "source_bytes_sha": sha, "graph_status": status}


def test_plan_keeps_the_most_enriched_then_oldest():
    from src.agentrag.ingestion.dedupe_corpus import plan

    docs = [_d("old"), _d("rich"), _d("empty"), _d("solo", sha="s2")]
    counts = {"old": {"leaves": 9, "contextualized": 0, "summaries": 0},
              "rich": {"leaves": 9, "contextualized": 9, "summaries": 3},
              "solo": {"leaves": 4}}
    (g,) = plan(docs, counts)
    assert g.keep["id"] == "rich" and {d["id"] for d in g.drop} == {"old", "empty"}
    # tie → oldest
    (g,) = plan([_d("a"), _d("b")], {"a": {"leaves": 1}, "b": {"leaves": 1}})
    assert g.keep["id"] == "a"


def test_plan_never_merges_owners_or_touches_in_flight_or_unhashed():
    from src.agentrag.ingestion.dedupe_corpus import plan

    assert plan([_d("a", user="u1"), _d("b", user="u2")], {}) == []
    assert plan([_d("a"), _d("b", status="parsing")], {}) == []
    assert plan([_d("a", sha=None), _d("b", sha=None)], {}) == []


def test_backfill_finds_folder_and_upload_originals(tmp_path):
    from src.agentrag.ingestion.dedupe_corpus import _candidate_paths

    paths = _candidate_paths(tmp_path, "D", "abc.pdf")
    assert tmp_path / "abc.pdf" in paths and tmp_path / "D.pdf" in paths
    assert _candidate_paths(tmp_path, "D", None) == []


def test_backfill_trusts_only_an_unchanged_file(tmp_path, monkeypatch):
    # Review: hashing an edited file would group a stale document with a
    # current copy, and the cleanup could keep the stale one.
    import datetime as dt
    import os

    from src.agentrag.ingestion.dedupe_corpus import _file_is_what_was_ingested

    f = tmp_path / "a.md"
    f.write_bytes(b"v1")
    key = folder.content_cache_key(b"v1", ".md")
    now = dt.datetime.now(dt.timezone.utc)
    assert _file_is_what_was_ingested(f, key, None)  # ingest key still matches
    old = now - dt.timedelta(days=1)
    os.utime(f, (old.timestamp() - 60, old.timestamp() - 60))
    assert _file_is_what_was_ingested(f, "stale-settings-key", old)  # untouched since ingest
    f.write_bytes(b"v2")  # edited after ingest (mtime = now)
    assert not _file_is_what_was_ingested(f, "stale-settings-key", old)
    assert not _file_is_what_was_ingested(f, None, None)


def test_dedupe_lookups_require_a_searchable_copy(monkeypatch):
    import asyncio

    from src.agentrag.adapter import upload_dedupe

    sess = _patch_session(monkeypatch, upload_dedupe, [None])
    asyncio.run(upload_dedupe.find_existing_document("abc", None))
    sess2 = _patch_session(monkeypatch, pipeline, [None, None])
    asyncio.run(pipeline._indexed_duplicate_of("abc", "new.pdf"))
    for q in (sess.sql[0], sess2.sql[1]):
        assert "documents.graph_status IS DISTINCT FROM 'pending'" in q
        assert "documents.graph_status IS DISTINCT FROM 'parsing'" in q


class _Doc:
    def __init__(self, status, content_hash, id_="d"):
        self.graph_status = status
        self.content_hash = content_hash
        self.id = id_


def test_skeleton_populated_despite_hash_drift_but_not_an_inflight_run(monkeypatch):
    """Upload skeleton (pending, no segments) under the same source_id is
    populated even when the api computed a different hash; a pending row that
    already has segments (another run) is not hijacked on a mismatch."""
    import asyncio

    from src.agentrag.ingestion.stores.postgres_store import PostgresStore

    async def run(existing, has_segments):
        store = PostgresStore()

        async def _has(_s, _id):
            return has_segments

        monkeypatch.setattr(PostgresStore, "_has_segments", staticmethod(_has))
        decided = {}

        class _R:
            def scalars(self):
                return self

            def all(self):
                return existing

        class _S:
            def add(self, obj):
                decided.setdefault("added", []).append(obj)

            async def execute(self, stmt):
                return _R()

            async def flush(self):
                pass

            async def commit(self):
                pass

            async def rollback(self):
                pass

            async def delete(self, obj):
                decided["deleted"] = obj

        doc_id, status = await store.save_document_and_segments(
            _S(), {"source_id": "s", "content_hash": "worker-key", "title": "t",
                   "source_type": "pdf", "source_bytes_sha": "b"}, [], project_id="p")
        return doc_id, status, decided

    skel = _Doc("pending", "api-key", "SKEL")
    doc_id, status, decided = asyncio.run(run([skel], has_segments=False))
    assert (doc_id, status) == ("SKEL", "ingested") and "deleted" not in decided
    assert skel.content_hash == "worker-key" and skel.source_bytes_sha == "b"

    inflight = _Doc("pending", "older-key", "RUN")
    _doc_id, _status, decided = asyncio.run(run([inflight], has_segments=True))
    assert decided.get("deleted") is inflight  # old replace behaviour, not hijacked

    # a "done" twin under the same source_id must not short-circuit the skeleton
    done = _Doc("done", "worker-key", "DONE")
    skel2 = _Doc("pending", "api-key", "SKEL2")
    doc_id, status, _ = asyncio.run(run([done, skel2], has_segments=False))
    assert (doc_id, status) == ("SKEL2", "ingested")
