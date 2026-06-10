from src.agentrag.retrieval import elasticsearch_retriever as er


def test_summary_nodes_capped_by_ratio(monkeypatch):
    monkeypatch.setattr(er.settings, "RAPTOR_SUMMARY_MAX_RATIO", 0.4)
    r = er.ElasticsearchRetriever.__new__(er.ElasticsearchRetriever)
    hits = [
        {"node_level": 1, "content": "s1"}, {"node_level": 1, "content": "s2"},
        {"node_level": 1, "content": "s3"}, {"node_level": 0, "content": "l1"},
        {"node_level": 0, "content": "l2"}, {"node_level": 0, "content": "l3"},
    ]
    out = r._cap_summary_nodes(hits, size=5)
    n_summary = sum(1 for h in out if h.get("node_level", 0) >= 1)
    assert n_summary <= 2  # floor(0.4 * 5)
    assert len(out) == 5   # leaves backfill the dropped summaries
