"""A parser-config change must not throw away every context sentence.

The context blurb answers "what is this passage about", which depends on the
chunk and the document's identity — not on whether the parser rendered tables.
Keying on the PARSED text made the 2026-09-06 arm-B flip regenerate ~4.6k
sentences that would all have been reusable.

Note the deliberate asymmetry with the document cache key in
`connectors/folder.py`, which DOES include the parser arm: a re-ingest must redo
the parse, while a derived per-chunk blurb need not. Same word "cache",
different dependency, different key.
"""
import asyncio
import hashlib

from src.agentrag.config import settings
from src.agentrag.ingestion.connectors.folder import FolderConnector
from src.agentrag.ingestion.contextualizer import Contextualizer


class _StubGateway:
    def __init__(self):
        self.calls = 0

    async def text_response(self, system_prompt, user_prompt, task):
        self.calls += 1
        return f"context {self.calls}"


def test_documents_expose_the_raw_source_hash_separately(tmp_path, monkeypatch):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 bytes")
    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", True)
    doc = FolderConnector(str(tmp_path)).list_documents()[0]
    assert doc["source_bytes_sha"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # content_hash stays parse-aware so a flag flip still forces a re-ingest
    assert doc["content_hash"] != doc["source_bytes_sha"]


def test_a_parse_change_reuses_cached_context(tmp_path):
    gateway = _StubGateway()
    ctx = Contextualizer(gateway, cache_dir=str(tmp_path))
    chunks = [{"content": "Paracetamol 500 mg", "content_hash": "chunk-1"}]

    asyncio.run(ctx.contextualize_chunks(
        doc_text="ORIGINAL PARSE", chunks=chunks, document_title="d", doc_key="src-sha"))
    assert gateway.calls == 1

    # Same document, different parse output (arm B appended table markdown).
    chunks2 = [{"content": "Paracetamol 500 mg", "content_hash": "chunk-1"}]
    asyncio.run(ctx.contextualize_chunks(
        doc_text="PARSE WITH | TABLES |", chunks=chunks2,
        document_title="d", doc_key="src-sha"))
    assert gateway.calls == 1, "a parse change must not invalidate the blurb"
    assert chunks2[0]["context_text"] == "context 1"


def test_a_genuine_source_change_does_invalidate(tmp_path):
    gateway = _StubGateway()
    ctx = Contextualizer(gateway, cache_dir=str(tmp_path))
    chunks = [{"content": "Paracetamol 500 mg", "content_hash": "chunk-1"}]

    asyncio.run(ctx.contextualize_chunks(
        doc_text="A", chunks=chunks, document_title="d", doc_key="sha-v1"))
    chunks2 = [{"content": "Paracetamol 500 mg", "content_hash": "chunk-1"}]
    asyncio.run(ctx.contextualize_chunks(
        doc_text="A", chunks=chunks2, document_title="d", doc_key="sha-v2"))
    assert gateway.calls == 2


def test_without_a_doc_key_it_falls_back_to_the_old_behaviour(tmp_path):
    """Callers that pass no key keep the previous parsed-text keying."""
    gateway = _StubGateway()
    ctx = Contextualizer(gateway, cache_dir=str(tmp_path))
    for text in ("PARSE A", "PARSE B"):
        chunks = [{"content": "x", "content_hash": "c1"}]
        asyncio.run(ctx.contextualize_chunks(
            doc_text=text, chunks=chunks, document_title="d"))
    assert gateway.calls == 2
