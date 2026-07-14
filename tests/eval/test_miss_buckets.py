from src.agentrag.eval.miss_buckets import (
    bucket_row, gold_overlap, render_report, summarize_buckets,
)


def _row(**over):
    row = {
        "qid": "c2-1", "question": "q?",
        "system_answer": "Trả lời chắc chắn [1].",
        "system_mean": 0.0, "oracle_mean": 1.0, "judge2_mean": 0.0,
        "refusal_class": "hallucinated",
        "cited_sources": [1],
        "packed": [{"content": "hoàn toàn khác biệt nội dung", "rerank_score": 0.6,
                    "document_title": "d", "section_path": "s"}],
        "gold_contexts": ["thuốc metformin liều 500mg ngày hai lần"],
        "tool_queries": ["q"], "citations_count": 1,
    }
    row.update(over)
    return row


def test_gold_overlap_high_when_gold_packed():
    packed = [{"content": "thuốc metformin liều 500mg ngày hai lần cho bệnh nhân"}]
    assert gold_overlap(packed, ["thuốc metformin liều 500mg ngày hai lần"]) > 0.5


def test_gold_overlap_zero_when_disjoint():
    assert gold_overlap([{"content": "abc def"}], ["xyz uvw"]) == 0.0


def test_gold_overlap_empty_packed():
    assert gold_overlap([], ["gold"]) == 0.0


def test_bucket_not_a_miss():
    assert bucket_row(_row(system_mean=0.9)) is None


def test_bucket_false_abstention():
    row = _row(refusal_class="abstained", cited_sources=[], citations_count=0)
    assert bucket_row(row) == "false_abstention"


def test_bucket_retrieval_miss():
    assert bucket_row(_row()) == "retrieval_miss"  # packed disjoint from gold


def test_bucket_generation_miss():
    row = _row(packed=[{"content": "thuốc metformin liều 500mg ngày hai lần"}])
    assert bucket_row(row) == "generation_miss"


def test_summarize_counts_and_judge_gap():
    rows = [
        _row(),                                                     # retrieval_miss
        _row(qid="c2-2", refusal_class="abstained", cited_sources=[]),  # false_abstention
        _row(qid="c2-3", system_mean=0.9, judge2_mean=0.85),        # not a miss
        _row(qid="c2-4", system_mean=0.3, judge2_mean=0.8),         # judge_gap flag
    ]
    s = summarize_buckets(rows)
    assert s["n"] == 4
    assert s["misses"] == 3
    assert s["buckets"]["retrieval_miss"] == 2
    assert s["buckets"]["false_abstention"] == 1
    assert s["judge_gap_rows"] == ["c2-4"]


def test_render_report_contains_buckets_and_rows():
    rows = [_row()]
    md = render_report(rows, summarize_buckets(rows), label="test-set")
    assert "retrieval_miss" in md
    assert "c2-1" in md
    assert md.startswith("# Miss buckets — test-set")
