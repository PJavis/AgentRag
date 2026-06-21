"""Smoke test: with every enhancement flag ON, the embed-input helper,
summary cap, and grounding marker compose without error.
Pure-unit level (no live ES/LLM) — guards against signature drift."""
from src.agentrag.ingestion.pipeline import _embed_input_for_chunk
from src.agentrag.retrieval import elasticsearch_retriever as er
from src.agentrag.agent.service import _has_uncertainty


def test_pipeline_helper_and_cap_and_classify(monkeypatch):
    monkeypatch.setattr(er.settings, "RAPTOR_SUMMARY_MAX_RATIO", 0.4)

    # WS1 embed-input
    assert _embed_input_for_chunk({"content": "x", "context_text": "ctx"}) == "ctx\n\nx"

    # WS2 cap (leaves backfill the demoted summaries when truncated to size)
    r = er.ElasticsearchRetriever.__new__(er.ElasticsearchRetriever)
    capped = r._cap_summary_nodes(
        [{"node_level": 1}, {"node_level": 1},
         {"node_level": 0}, {"node_level": 0}], size=3)
    assert sum(1 for h in capped if h.get("node_level", 0) >= 1) <= 1

    # WS3 grounding marker
    assert _has_uncertainty("Tôi không tìm thấy thông tin.")
