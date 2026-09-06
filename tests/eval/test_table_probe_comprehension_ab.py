"""0c — comprehension probe. Tests pin the scoring, not the outcome."""
from scripts.eval.table_probe_comprehension_ab import (
    build_prompt,
    normalize_answer,
    pair_outcomes,
    score_answer,
    token_f1,
)


def test_normalize_answer_strips_case_punctuation_and_padding():
    assert normalize_answer("  500 MG.  ") == "500 mg"
    assert normalize_answer("«uống»") == "uống"
    assert normalize_answer(None) == ""


def test_score_answer_accepts_an_exact_cell_and_a_cell_inside_a_sentence():
    assert score_answer("500 mg", "500 mg") == 1.0
    assert score_answer("The dose is 500 mg.", "500 mg") == 1.0


def test_score_answer_rejects_a_different_cell():
    assert score_answer("400 mg", "500 mg") == 0.0


def test_score_answer_rejects_the_unknown_sentinel_even_if_truth_is_unknown():
    """UNKNOWN is an abstention, never a hit — otherwise a model that always
    abstains scores on any row whose cell happens to say 'unknown'."""
    assert score_answer("UNKNOWN", "unknown") == 0.0


def test_score_answer_does_not_credit_a_one_word_overlap_with_a_long_cell():
    long_cell = "Khởi phát chủ yếu tháng thứ 2, sẩn mụn nước, vị trí má và trán"
    assert score_answer("má", long_cell) == 0.0


def test_token_f1_is_continuous_between_zero_and_one():
    assert token_f1("500 mg", "500 mg") == 1.0
    assert token_f1("", "500 mg") == 0.0
    partial = token_f1("500 milligram", "500 mg")
    assert 0.0 < partial < 1.0


def test_prompt_carries_the_context_and_names_row_and_column():
    p = build_prompt("PAGE TEXT HERE", row_label="Paracetamol", column="Liều")
    assert "PAGE TEXT HERE" in p["user"]
    assert "Paracetamol" in p["user"] and "Liều" in p["user"]
    # Grounding is not optional: the model must be told to abstain, not guess.
    assert "UNKNOWN" in p["system"]


def test_pair_outcomes_counts_wins_losses_and_ties():
    a = [{"correct": 0.0}, {"correct": 1.0}, {"correct": 1.0}]
    b = [{"correct": 1.0}, {"correct": 0.0}, {"correct": 1.0}]
    out = pair_outcomes(a, b)
    assert (out["n_wins"], out["n_losses"], out["n_ties"]) == (1, 1, 1)


def test_pair_outcomes_excludes_a_question_that_errored_in_either_arm():
    """An API failure is not a loss. Scoring it as one would measure uptime."""
    a = [{"correct": 1.0}, {"error": "timeout"}]
    b = [{"correct": 0.0}, {"correct": 1.0}]
    out = pair_outcomes(a, b)
    assert out["n_wins"] + out["n_losses"] + out["n_ties"] == 1
    assert out["n_excluded"] == 1


def test_cell_bleed_separates_verbosity_from_lost_cell_boundaries():
    """The mechanism test: surplus words FROM OTHER CELLS mean the model could not
    see where the cell stopped — a wrong answer that containment scores as right."""
    from scripts.eval.table_probe_comprehension_ab import cell_bleed

    table = [["Thuốc", "Liều"], ["Paracetamol", "500 mg"], ["Ibuprofen", "400 mg"]]

    # Surplus is measured against the gold cell's own tokens, so "500" and "mg"
    # are not surplus — "ibuprofen" and "400" are, and both are another row's.
    borrowed, extra = cell_bleed("500 mg Ibuprofen 400 mg", "500 mg", table)
    assert extra == 2 and borrowed == 2

    borrowed, extra = cell_bleed("the dose is 500 mg", "500 mg", table)
    assert extra == 3 and borrowed == 0      # verbose, but no row was merged in

    assert cell_bleed("500 mg", "500 mg", table) == (0, 0)


def test_aggregate_samples_reports_fractional_correctness_and_flags_disagreement():
    from scripts.eval.table_probe_comprehension_ab import aggregate_samples

    out = aggregate_samples([
        {"answer": "500 mg", "correct": 1.0, "f1": 1.0},
        {"answer": "UNKNOWN", "correct": 0.0, "f1": 0.0},
        {"answer": "500 mg", "correct": 1.0, "f1": 1.0},
    ])
    assert abs(out["correct"] - 2 / 3) < 1e-9
    assert out["unstable"] is True
    assert out["samples"] == 3


def test_aggregate_samples_is_stable_when_every_call_agrees():
    from scripts.eval.table_probe_comprehension_ab import aggregate_samples

    out = aggregate_samples([{"answer": "x", "correct": 1.0, "f1": 1.0}] * 3)
    assert out["correct"] == 1.0 and out["unstable"] is False


def test_aggregate_samples_keeps_an_all_error_question_excludable():
    from scripts.eval.table_probe_comprehension_ab import aggregate_samples

    assert "error" in aggregate_samples([{"error": "timeout"}, {"error": "timeout"}])


def test_duplication_control_adds_a_second_copy_and_no_structure():
    from scripts.eval.table_probe_comprehension_ab import duplicated_context

    out = duplicated_context("PAGE")
    assert out.count("PAGE") == 2
    assert "|" not in out  # no markdown structure smuggled into the control
