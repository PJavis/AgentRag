from src.agentrag.eval.citation_mining import mine_triplets


def _row(**over):
    row = {
        "qid": "c2-1", "question": "liều metformin?",
        "system_answer": "500mg [1].",
        "system_mean": 0.9,
        "refusal_class": "hallucinated",
        "cited_sources": [1],
        "packed": [
            {"content": "metformin 500mg", "rerank_score": 0.72},
            {"content": "insulin liều", "rerank_score": 0.66},
            {"content": "paracetamol", "rerank_score": 0.58},
        ],
    }
    row.update(over)
    return row


def test_mines_cited_vs_hardest_uncited():
    trips = mine_triplets([_row()])
    assert trips == [{
        "query": "liều metformin?",
        "positive": "metformin 500mg",
        "negative": "insulin liều",   # hardest uncited (0.66 > 0.58)
        "source": "citation",
    }]


def test_skips_low_score_rows():
    assert mine_triplets([_row(system_mean=0.4)]) == []


def test_skips_rows_without_negatives():
    row = _row(cited_sources=[1, 2, 3])  # everything cited → no negative
    assert mine_triplets([row]) == []


def test_multiple_positives_cycle_negatives():
    row = _row(cited_sources=[1, 2])
    trips = mine_triplets([row])
    assert len(trips) == 2
    assert {t["positive"] for t in trips} == {"metformin 500mg", "insulin liều"}
    assert all(t["negative"] == "paracetamol" for t in trips)


def test_ignores_out_of_range_citation():
    row = _row(cited_sources=[1, 9])
    trips = mine_triplets([row])
    assert len(trips) == 1
    assert trips[0]["positive"] == "metformin 500mg"
