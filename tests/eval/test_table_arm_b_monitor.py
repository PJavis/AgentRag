"""Production monitor for the arm-B rollout. Tests pin the metrics, not a result."""
from scripts.eval.table_arm_b_monitor import (
    arm_of_answer,
    is_abstention,
    is_lookup_overrun,
    rows_touched,
    summarize_by_arm,
)

TABLE = [
    ["Thuốc", "Liều", "Đường dùng"],
    ["Paracetamol", "500 mg", "uống"],
    ["Ibuprofen", "400 mg", "tiêm"],
]


def test_abstention_uses_the_shipped_uncertainty_markers():
    assert is_abstention("Tôi không tìm thấy thông tin trong tài liệu.")
    assert is_abstention("Thông tin này không có trong tài liệu.")
    assert not is_abstention("Liều dùng là 500 mg.")


def test_rows_touched_counts_distinct_data_rows_the_answer_draws_from():
    assert rows_touched("500 mg", TABLE) == 1
    assert rows_touched("500 mg và Ibuprofen 400 mg", TABLE) == 2


def test_rows_touched_ignores_tokens_shared_by_several_rows():
    """A word appearing in two rows identifies neither. Counting it would report
    overrun on every answer that happens to use a common word."""
    shared = [["Thuốc", "Liều"], ["Paracetamol", "500 mg"], ["Paracetamol", "400 mg"]]
    assert rows_touched("Paracetamol", shared) == 0


def test_rows_touched_ignores_the_header_row():
    assert rows_touched("Thuốc Liều Đường dùng", TABLE) == 0


def test_lookup_overrun_flags_a_short_answer_spanning_two_rows():
    assert is_lookup_overrun("500 mg Ibuprofen 400 mg", TABLE, max_tokens=60)


def test_lookup_overrun_does_not_flag_a_single_row_answer():
    assert not is_lookup_overrun("500 mg", TABLE, max_tokens=60)


def test_lookup_overrun_does_not_flag_a_long_answer():
    """A long answer may legitimately summarise several rows. Only short
    lookup-shaped answers are evidence of a lost cell boundary."""
    long_answer = "500 mg Ibuprofen 400 mg " + "chi tiết " * 60
    assert not is_lookup_overrun(long_answer, TABLE, max_tokens=60)


def test_arm_of_answer_reads_the_segment_stamp_behind_the_citations():
    segs = {"h1": {"pdf_preserve_tables": True}, "h2": {"pdf_preserve_tables": True}}
    assert arm_of_answer([{"content_hash": "h1"}, {"content_hash": "h2"}], segs) == "B"

    segs_off = {"h1": {"pdf_preserve_tables": False}}
    assert arm_of_answer([{"content_hash": "h1"}], segs_off) == "A"


def test_an_answer_citing_both_arms_is_mixed_and_excluded_not_guessed():
    """Mid-rollout, one answer can cite a doc ingested under each arm. Assigning
    it to either would put the rollout's transition into the comparison."""
    segs = {"h1": {"pdf_preserve_tables": True}, "h2": {"pdf_preserve_tables": False}}
    assert arm_of_answer([{"content_hash": "h1"}, {"content_hash": "h2"}], segs) == "mixed"


def test_an_answer_with_no_stamped_citation_is_unknown():
    assert arm_of_answer([{"content_hash": "nope"}], {}) == "unknown"
    assert arm_of_answer([], {}) == "unknown"


def test_summarize_reports_rates_per_arm_and_never_divides_by_zero():
    recs = [
        {"arm": "A", "abstained": True, "overrun": False, "latency_ms": 100},
        {"arm": "A", "abstained": False, "overrun": True, "latency_ms": 300},
        {"arm": "B", "abstained": False, "overrun": False, "latency_ms": 200},
        {"arm": "mixed", "abstained": True, "overrun": True, "latency_ms": 999},
    ]
    out = summarize_by_arm(recs)
    assert out["A"]["answers"] == 2
    assert out["A"]["abstention_rate"] == 0.5
    assert out["A"]["overrun_rate"] == 0.5
    assert out["B"]["overrun_rate"] == 0.0
    # mixed is reported, never folded into A or B
    assert "mixed" in out and out["mixed"]["answers"] == 1
    assert summarize_by_arm([]) == {}


def test_resolve_pdf_maps_a_cited_title_to_the_corpus_file(tmp_path):
    from scripts.eval.table_arm_b_monitor import resolve_pdf

    (tmp_path / "abc-123.pdf").write_bytes(b"%PDF-1.4")
    assert resolve_pdf(str(tmp_path), "abc-123") == tmp_path / "abc-123.pdf"


def test_resolve_pdf_returns_none_rather_than_guessing(tmp_path):
    from scripts.eval.table_arm_b_monitor import resolve_pdf

    assert resolve_pdf(str(tmp_path), "missing") is None
    assert resolve_pdf(str(tmp_path), "") is None


def test_citation_pages_handles_the_range_strings_production_actually_stores():
    """Real rows carry `page` values like "22-24" for a chunk spanning pages.
    A bare int() on that raised on live data."""
    from scripts.eval.table_arm_b_monitor import citation_pages

    assert citation_pages({"page": 7}) == [7]
    assert citation_pages({"page": "7"}) == [7]
    assert citation_pages({"page": "22-24"}) == [22, 23, 24]
    assert citation_pages({"page": "p.9"}) == [9]
    assert citation_pages({"page_start": 4, "page_end": 6}) == [4, 5, 6]
    assert citation_pages({"page_start": 4}) == [4]


def test_citation_pages_returns_nothing_rather_than_raising_on_junk():
    from scripts.eval.table_arm_b_monitor import citation_pages

    assert citation_pages({}) == []
    assert citation_pages({"page": "unknown"}) == []
    assert citation_pages({"page": "a-b"}) == []
