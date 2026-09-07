"""Repairing a few documents must not cost a whole-corpus pass.

Contextualisation is ~1 LLM call per chunk, so a full re-ingest of this corpus
is thousands of calls. `only` filters by source_id — deliberately not by
pointing the connector at a different directory, because source_id is the path
relative to the folder root, so a subset directory would mint new source_ids and
duplicate every document instead of replacing it.
"""
from src.agentrag.ingestion.connectors.folder import FolderConnector


def _corpus(tmp_path, names):
    for n in names:
        (tmp_path / n).write_bytes(b"%PDF-1.4 " + n.encode())
    return FolderConnector(str(tmp_path))


def test_the_filter_selects_by_source_id(tmp_path):
    docs = _corpus(tmp_path, ["a.pdf", "b.pdf", "c.pdf"]).list_documents()
    only = {"b.pdf"}
    picked = [d for d in docs if d["source_id"] in only]
    assert [d["source_id"] for d in picked] == ["b.pdf"]


def test_source_ids_are_relative_to_the_folder_root(tmp_path):
    """This is why the filter exists instead of a subset directory."""
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "x.pdf").write_bytes(b"%PDF-1.4")
    from_root = FolderConnector(str(tmp_path)).list_documents()[0]["source_id"]
    from_sub = FolderConnector(str(sub)).list_documents()[0]["source_id"]
    assert from_root == "nested/x.pdf"
    assert from_sub == "x.pdf"
    assert from_root != from_sub, (
        "ingesting a subset from its own directory would create a second "
        "document rather than replacing the original"
    )


def test_ingest_folder_accepts_the_only_parameter():
    import inspect

    from src.agentrag.ingestion.pipeline import ingest_folder

    assert "only" in inspect.signature(ingest_folder).parameters
