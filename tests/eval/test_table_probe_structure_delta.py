"""0a — structure-delta: does flattening actually destroy row/column adjacency?

The probe's premise is that `get_text("text", sort=True)` scrambles tables. That
premise has never been measured on this corpus. These tests pin the measurement,
not the outcome: the metric must be able to report **no damage**, otherwise it is
rigged toward the answer the probe wants.
"""
from scripts.eval.table_probe_structure_delta import (
    cells_on_one_line,
    format_report,
    norm_cell,
    row_adjacency,
    summarize,
    table_scores,
    usable_candidate,
)

ROWS = [
    ["Thuốc", "Liều", "Đường dùng"],
    ["Paracetamol", "500 mg", "uống"],
    ["Ibuprofen", "400 mg", "uống"],
    ["Morphin", "10 mg", "tiêm"],
]

INTACT_PAGE = (
    "Bảng 1. Liều dùng\n"
    "Thuốc Liều Đường dùng\n"
    "Paracetamol 500 mg uống\n"
    "Ibuprofen 400 mg uống\n"
    "Morphin 10 mg tiêm\n"
)

# Column-major flattening — every cell survives, every row is destroyed.
SCRAMBLED_PAGE = (
    "Thuốc\nParacetamol\nIbuprofen\nMorphin\n"
    "Liều\n500 mg\n400 mg\n10 mg\n"
    "Đường dùng\nuống\nuống\ntiêm\n"
)


def test_norm_cell_collapses_internal_newlines_and_runs_of_space():
    assert norm_cell("Đường\ndùng") == "Đường dùng"
    assert norm_cell("  500   mg \n") == "500 mg"
    assert norm_cell(None) == ""


def test_cells_on_one_line_true_when_the_row_survived_flattening():
    assert cells_on_one_line(INTACT_PAGE, ["Paracetamol", "500 mg", "uống"])


def test_cells_on_one_line_false_when_the_row_was_split_across_lines():
    assert not cells_on_one_line(SCRAMBLED_PAGE, ["Paracetamol", "500 mg", "uống"])


def test_cells_on_one_line_requires_document_order_not_mere_presence():
    """A line holding the cells in the wrong order has lost the binding too."""
    text = "500 mg Paracetamol uống\n"
    assert not cells_on_one_line(text, ["Paracetamol", "500 mg", "uống"])


def test_cells_on_one_line_ignores_empty_cells():
    assert cells_on_one_line("Paracetamol uống\n", ["Paracetamol", "", None, "uống"])


def test_cells_on_one_line_is_false_when_fewer_than_two_cells_are_populated():
    """One cell alone carries no adjacency to preserve — never count it as a win."""
    assert not cells_on_one_line("Paracetamol\n", ["Paracetamol", "", ""])


def test_row_adjacency_counts_only_rows_with_two_populated_cells():
    rows = [["a", "b"], ["c", ""], ["d", "e"]]
    text = "a b\nc\nd e\n"
    preserved, total = row_adjacency(text, rows)
    assert (preserved, total) == (2, 2)  # the ["c", ""] row is not counted at all


def test_pool_membership_counts_the_header_row_but_scoring_does_not():
    """`usable_candidate` must select the same 27 tables the survey counted.

    The survey's `structured_rows` includes the header row; the adjacency
    denominator must not, or arm B scores its own repeated header as a win.
    """
    s = table_scores(ROWS, INTACT_PAGE)
    assert s["structured_rows"] == 4   # header + 3 data rows, survey-compatible
    assert s["scored_rows"] == 3       # data rows only


def test_arm_a_scores_one_when_flattening_did_no_damage():
    """The metric MUST be able to say 'nothing was destroyed' — else it is rigged."""
    s = table_scores(ROWS, INTACT_PAGE)
    assert s["arm_a"] == 1.0
    assert s["delta"] == 0.0


def test_arm_a_scores_zero_on_column_major_flattening():
    s = table_scores(ROWS, SCRAMBLED_PAGE)
    assert s["arm_a"] == 0.0
    assert s["arm_b"] == 1.0
    assert s["delta"] == 1.0


def test_arm_b_is_reported_as_tautological_not_as_evidence():
    """render_markdown emits one row per line, so arm B is 1.0 by construction."""
    s = table_scores(ROWS, SCRAMBLED_PAGE)
    assert s["arm_b_tautological"] is True


def test_header_adjacency_is_scored_separately_from_data_rows():
    s_intact = table_scores(ROWS, INTACT_PAGE)
    s_scrambled = table_scores(ROWS, SCRAMBLED_PAGE)
    assert s_intact["header_a"] is True
    assert s_scrambled["header_a"] is False


def test_an_unsafe_table_is_marked_not_scored():
    """Arm B never rewrites a gate-failing table, so a delta would be fiction."""
    prose = [["Một đoạn văn dài " * 20], ["Một đoạn khác " * 20]]
    s = table_scores(prose, "irrelevant")
    assert s["gate_passed"] is False
    assert s["delta"] is None


def test_usable_candidate_matches_the_power_analysis_pool():
    assert usable_candidate({"structured_rows": 3, "cols": 3})  # survey-compatible: header counts
    assert not usable_candidate({"structured_rows": 2, "cols": 5})
    assert not usable_candidate({"structured_rows": 9, "cols": 2})


def test_summarize_reports_per_document_as_well_as_overall():
    recs = [
        {"doc": "a.pdf", "page": 1, "arm_a": 0.0, "delta": 1.0, "gate_passed": True},
        {"doc": "a.pdf", "page": 2, "arm_a": 0.0, "delta": 1.0, "gate_passed": True},
        {"doc": "b.pdf", "page": 1, "arm_a": 1.0, "delta": 0.0, "gate_passed": True},
    ]
    out = summarize(recs)
    assert out["tables"] == 3
    assert out["docs"] == 2
    assert out["median_arm_a"] == 0.0
    # Per-document means, so one 14-table document cannot carry the headline.
    assert out["per_doc"]["a.pdf"]["mean_arm_a"] == 0.0
    assert out["per_doc"]["b.pdf"]["mean_arm_a"] == 1.0
    assert out["mean_of_doc_means_arm_a"] == 0.5


def test_summarize_ignores_gate_failing_tables():
    recs = [
        {"doc": "a.pdf", "page": 1, "arm_a": 0.0, "delta": 1.0, "gate_passed": True},
        {"doc": "a.pdf", "page": 9, "arm_a": None, "delta": None, "gate_passed": False},
    ]
    assert summarize(recs)["tables"] == 1


def test_summarize_of_nothing_does_not_divide_by_zero():
    out = summarize([])
    assert out["tables"] == 0 and out["median_arm_a"] is None


def test_report_states_the_kill_condition_and_the_arm_b_caveat():
    md = format_report(summarize([
        {"doc": "a.pdf", "page": 1, "arm_a": 0.9, "delta": 0.1, "gate_passed": True},
    ]))
    assert "tautolog" in md.lower()          # arm B = 1.0 by construction
    assert "NO-GO" in md                     # the kill condition must be named
    assert "n_eff" in md or "cluster" in md.lower()  # clustering caveat carried over


def test_failure_attribution_separates_wrapped_cells_from_scrambling():
    """A wrapped cell and a scrambled column are different defects. Say which."""
    from scripts.eval.table_probe_structure_delta import classify_row_failure

    wrapped = "Mào tinh hoàn\nTắc mào tinh\nhoàn nguyên phát\n"
    assert classify_row_failure(wrapped, ["Mào tinh hoàn", "Tắc mào tinh hoàn nguyên phát"]) == (
        "ordered_but_split_across_lines"
    )

    scrambled = "500 mg\nParacetamol\n"
    assert classify_row_failure(scrambled, ["Paracetamol", "500 mg"]) == "out_of_document_order"

    absent = "something else entirely\n"
    assert classify_row_failure(absent, ["Paracetamol", "500 mg"]) == "cell_text_absent"

    # The common real case: a wrapped cell whose halves are separated by the
    # neighbouring column's text. The words are all there; the cell is not.
    interleaved = "Tắc mào tinh\nSau nhiễm khuẩn\nhoàn nguyên phát\n500 mg\n"
    assert classify_row_failure(
        interleaved, ["Tắc mào tinh hoàn nguyên phát", "500 mg"]
    ) == "cell_fragmented_across_columns"


def test_failure_attribution_is_carried_into_table_scores():
    s = table_scores(ROWS, SCRAMBLED_PAGE)
    assert sum(s["arm_a_failures"].values()) == s["scored_rows"]
