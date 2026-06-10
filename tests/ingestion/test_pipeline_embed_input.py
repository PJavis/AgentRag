from src.agentrag.ingestion.pipeline import _embed_input_for_chunk


def test_embed_input_prepends_context_when_present():
    c = {"content": "leaf body", "context_text": "From cardiology chapter."}
    assert _embed_input_for_chunk(c) == "From cardiology chapter.\n\nleaf body"


def test_embed_input_is_content_only_when_no_context():
    c = {"content": "leaf body", "context_text": None}
    assert _embed_input_for_chunk(c) == "leaf body"
