from src.agentrag.eval.probe_rows import build_probe_row, parse_inline_citations


def test_parse_inline_citations_dedup_sorted():
    ans = "Liều dùng là 5mg [2]. Chống chỉ định suy thận [1][2]."
    assert parse_inline_citations(ans) == [1, 2]


def test_parse_inline_citations_ignores_links_and_empty():
    assert parse_inline_citations("xem [tài liệu](http://x) nhé") == []
    assert parse_inline_citations("") == []
    assert parse_inline_citations(None) == []


def _chat_out():
    return {
        "answer": "Đáp án đúng [1].",
        "context": [
            {"content": "gold text here", "rerank_score": 0.71,
             "document_title": "doc.pdf", "section_path": "1>2", "extra": "dropped"},
            {"content": "distractor", "rerank_score": 0.58,
             "document_title": "doc.pdf", "section_path": "3"},
        ],
        "citations": [{"source": 1}, {"source": 2}],
        "tool_trace": [
            {"tool_name": "search_hybrid_kg", "tool_input": {"query": "q-hop-1"}},
            {"tool_name": "search_hybrid_kg", "tool_input": {"query": "q-hop-2"}},
        ],
    }


def test_build_probe_row_shape():
    row = build_probe_row(
        qid="c2-1", question="q?", chat_out=_chat_out(), oracle_answer="oracle",
        system_mean=0.9, oracle_mean=1.0, judge2_mean=0.85,
        gold_contexts=["gold text here"],
    )
    assert row["qid"] == "c2-1"
    assert row["cited_sources"] == [1]
    assert row["refusal_class"] == "hallucinated"  # confident answer, per classify_refusal
    assert row["tool_queries"] == ["q-hop-1", "q-hop-2"]
    assert row["citations_count"] == 2
    assert row["packed"] == [
        {"content": "gold text here", "rerank_score": 0.71,
         "document_title": "doc.pdf", "section_path": "1>2"},
        {"content": "distractor", "rerank_score": 0.58,
         "document_title": "doc.pdf", "section_path": "3"},
    ]


def test_build_probe_row_abstention():
    out = _chat_out()
    out["answer"] = "Tài liệu hiện có không có thông tin để trả lời câu hỏi này."
    out["citations"] = []
    row = build_probe_row(
        qid="c2-2", question="q?", chat_out=out, oracle_answer="o",
        system_mean=0.0, oracle_mean=1.0, judge2_mean=0.0,
        gold_contexts=["g"],
    )
    assert row["refusal_class"] == "abstained"
    assert row["cited_sources"] == []
