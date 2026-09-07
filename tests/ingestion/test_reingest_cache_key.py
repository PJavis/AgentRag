"""A parser flag flip must actually re-ingest, and must not duplicate in ES.

Two defects this pins, both of which would silently corrupt an arm-B rollout:

  * `save_document_and_segments` skips a document whose `content_hash` matches,
    and the hash was the file bytes alone. Flipping a PARSER flag leaves the
    bytes identical, so every document reports "skipped" and the flag never
    takes effect — a flip that looks successful and changes nothing.
  * `index_segments` assigns a fresh uuid4 `_id` per chunk, so re-indexing
    APPENDS. Without a purge, retrieval serves both generations at once.
"""
from src.agentrag.config import settings
from src.agentrag.ingestion.connectors.folder import FolderConnector


def _pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 identical bytes")
    return path


def test_the_cache_key_changes_when_the_parser_arm_changes(tmp_path, monkeypatch):
    _pdf(tmp_path)
    connector = FolderConnector(str(tmp_path))

    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", False)
    off = connector.list_documents()[0]["content_hash"]
    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", True)
    on = connector.list_documents()[0]["content_hash"]

    assert off != on, "same hash → every document is skipped → the flip is a no-op"


def test_turning_the_arm_off_again_restores_the_previous_key(tmp_path, monkeypatch):
    """Flipping back must not strand the corpus on a third key.

    (The stronger claim — that an all-default deployment keeps the plain file
    hash, so it is never re-ingested merely because parse config joined the key —
    is asserted by test_returning_every_setting_to_default_restores_the_plain_hash.)
    """
    _pdf(tmp_path)
    connector = FolderConnector(str(tmp_path))

    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", False)
    before = connector.list_documents()[0]["content_hash"]
    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", True)
    flipped = connector.list_documents()[0]["content_hash"]
    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", False)
    restored = connector.list_documents()[0]["content_hash"]

    assert flipped != before
    assert restored == before


def test_a_non_pdf_is_unaffected_by_the_pdf_parser_flag(tmp_path, monkeypatch):
    (tmp_path / "notes.md").write_text("# hello", encoding="utf-8")
    connector = FolderConnector(str(tmp_path))

    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", False)
    off = connector.list_documents()[0]["content_hash"]
    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", True)
    assert connector.list_documents()[0]["content_hash"] == off


def test_the_pipeline_purges_a_document_before_re_indexing_it():
    """Pins the call, and its ORDER, in the pipeline source.

    A source-level assertion is crude, but the alternative — driving the real
    ingest — needs Postgres, Elasticsearch and an embedder, and the thing worth
    protecting is exactly one line: without the purge, re-ingest appends a second
    generation of chunks and retrieval serves both. Reversed, the purge would
    delete the chunks just written.
    """
    import inspect

    from src.agentrag.ingestion import pipeline

    source = inspect.getsource(pipeline)
    purge = source.index('es_store.delete_document(doc["title"])')
    index = source.index('es_store.index_segments(chunks_search, doc["title"])')
    assert purge < index


def test_an_ocr_setting_change_also_forces_a_reingest(tmp_path, monkeypatch):
    """PDF_PRESERVE_TABLES was never the only parse-affecting setting.

    Pointing VISION_BASE_URL at a host without the configured vision model
    silently degraded scanned pages to ~40 characters. A re-ingest to repair
    that must not report "skipped" just because the file bytes are unchanged.
    """
    _pdf(tmp_path)
    connector = FolderConnector(str(tmp_path))
    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", False)

    monkeypatch.setattr(settings, "VISION_BASE_URL", "http://127.0.0.1:11434/v1/")
    before = connector.list_documents()[0]["content_hash"]
    monkeypatch.setattr(settings, "VISION_BASE_URL",
                        "https://generativelanguage.googleapis.com/v1beta/openai/")
    assert connector.list_documents()[0]["content_hash"] != before

    monkeypatch.setattr(settings, "PDF_OCR_VISION_THRESHOLD", 250)
    assert connector.list_documents()[0]["content_hash"] != before


def test_returning_every_setting_to_default_restores_the_plain_hash(tmp_path, monkeypatch):
    """So a deployment that never touched these is never re-ingested for nothing."""
    import hashlib

    from src.agentrag.config import Settings

    path = _pdf(tmp_path)
    for name in ("PDF_PRESERVE_TABLES", "PDF_OCR_VISION_THRESHOLD", "VISION_BASE_URL",
                 "VISION_PROVIDER", "VISION_MODEL", "PDF_OCR_LANG", "PDF_OCR_DPI",
                 "PDF_OCR_FALLBACK_ENABLED", "PDF_OCR_MIN_TEXT_CHARS",
                 "PDF_OCR_VISION_FALLBACK"):
        monkeypatch.setattr(settings, name, Settings.model_fields[name].default)
    got = FolderConnector(str(tmp_path)).list_documents()[0]["content_hash"]
    assert got == hashlib.sha256(path.read_bytes()).hexdigest()
