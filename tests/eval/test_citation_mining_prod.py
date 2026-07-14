from src.agentrag.eval.citation_mining import feedback_to_row, mine_triplets


def _citations():
    # Shape of ChatMessage.citations — _build_packed_citations output (relevance order).
    return [
        {"source": 1, "excerpt": "metformin 500mg hai lần mỗi ngày", "document_title": "d.pdf"},
        {"source": 2, "excerpt": "insulin nền liều khởi đầu", "document_title": "d.pdf"},
        {"source": 3, "excerpt": "paracetamol hạ sốt", "document_title": "d.pdf"},
    ]


def test_feedback_to_row_thumbs_up_passes_mining_filter():
    row = feedback_to_row(
        question="liều metformin?", answer="500mg hai lần mỗi ngày [1].",
        citations=_citations(), rating=1,
    )
    assert row["system_mean"] == 1.0
    assert row["cited_sources"] == [1]
    assert [p["content"] for p in row["packed"]] == [
        "metformin 500mg hai lần mỗi ngày", "insulin nền liều khởi đầu", "paracetamol hạ sốt",
    ]
    trips = mine_triplets([row])
    assert len(trips) == 1
    assert trips[0]["positive"] == "metformin 500mg hai lần mỗi ngày"
    # rerank_score absent in prod citations → stable sort keeps relevance order:
    # hardest negative = first uncited item.
    assert trips[0]["negative"] == "insulin nền liều khởi đầu"
    assert trips[0]["source"] == "citation"


def test_feedback_to_row_downvote_filtered_by_mining():
    row = feedback_to_row(question="q?", answer="sai [1].", citations=_citations(), rating=-1)
    assert row["system_mean"] == 0.0
    assert mine_triplets([row]) == []


def test_feedback_to_row_none_on_empty_inputs():
    assert feedback_to_row(question="", answer="a [1]", citations=_citations(), rating=1) is None
    assert feedback_to_row(question="q", answer="", citations=_citations(), rating=1) is None
    assert feedback_to_row(question="q", answer="a [1]", citations=[], rating=1) is None


def test_feedback_to_row_sorts_by_source_number():
    cits = [
        {"source": 2, "excerpt": "second"},
        {"source": 1, "excerpt": "first"},
    ]
    row = feedback_to_row(question="q?", answer="a [1].", citations=cits, rating=1)
    assert [p["content"] for p in row["packed"]] == ["first", "second"]
    assert row["cited_sources"] == [1]
