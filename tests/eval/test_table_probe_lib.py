"""Tests for the table-probe pure helpers, with emphasis on the decision rule."""

import pytest

from scripts.eval.table_probe_lib import (
    arm_index_name,
    corpus_matches,
    decide_paired,
    decide_track,
    mean_delta,
    paired_outcomes,
    rank_targets,
    sign_test_p,
    validate_probe_row,
)


def test_decide_track_full_when_all_up():
    track, _ = decide_track(True, True, "sk-key")
    assert track == "full"


@pytest.mark.parametrize(
    "es,tei,key",
    [(False, True, "k"), (True, False, "k"), (True, True, None), (True, True, "")],
)
def test_decide_track_offline_when_any_missing(es, tei, key):
    track, reason = decide_track(es, tei, key)
    assert track == "offline"
    assert reason


def test_arm_index_name_isolates_arms():
    assert arm_index_name("probe", "a") == "probe_arm_a"
    assert arm_index_name("probe", "b") == "probe_arm_b"


def test_arm_index_name_rejects_bad_arm():
    with pytest.raises(ValueError):
        arm_index_name("probe", "c")


# --- target selection -------------------------------------------------------

SURVEY = {
    "per_doc": [
        {"doc": "big.pdf", "data_grids": [
            {"page": 8, "rows": 14, "cols": 6},    # 84 cells
            {"page": 16, "rows": 17, "cols": 8},   # 136 cells
        ]},
        {"doc": "small.pdf", "data_grids": [{"page": 3, "rows": 4, "cols": 2}]},
    ]
}


def test_rank_targets_returns_tables_not_documents():
    targets = rank_targets(SURVEY)
    assert len(targets) == 3
    assert targets[0] == {"doc": "big.pdf", "page": 16, "rows": 17, "cols": 8, "cells": 136}


def test_rank_targets_truncates():
    assert len(rank_targets(SURVEY, top_n=2)) == 2


def test_rank_targets_handles_empty_survey():
    assert rank_targets({"per_doc": []}) == []


# --- row schema -------------------------------------------------------------

GOOD_ROW = {
    "id": "tbl-01",
    "question": "Q?",
    "reference_answer": "0",
    "gold_contexts": ["| a | b |"],
    "source_doc": "2f499de4.pdf",
    "source_page": 8,
}


def test_validate_probe_row_accepts_good_row():
    assert validate_probe_row(GOOD_ROW) == []


def test_validate_probe_row_requires_doc_page_provenance():
    row = {k: v for k, v in GOOD_ROW.items() if k not in ("source_doc", "source_page")}
    errs = validate_probe_row(row)
    assert any("source_doc" in e for e in errs)
    assert any("source_page" in e for e in errs)


def test_validate_probe_row_rejects_non_int_page():
    errs = validate_probe_row({**GOOD_ROW, "source_page": "8"})
    assert any("source_page must be an int" in e for e in errs)


def test_validate_probe_row_flags_missing_and_empty_gold():
    errs = validate_probe_row({"id": "q1", "question": "Q?", "gold_contexts": []})
    assert any("reference_answer" in e for e in errs)
    assert any("gold_contexts" in e for e in errs)


# --- paired outcomes --------------------------------------------------------

def test_paired_outcomes_classifies_wins_losses_ties():
    a = {"q1": 0.0, "q2": 1.0, "q3": 0.5}
    b = {"q1": 1.0, "q2": 0.0, "q3": 0.5}
    out = paired_outcomes(a, b)
    assert (out["n_wins"], out["n_losses"], out["n_ties"]) == (1, 1, 1)
    assert out["wins"] == ["q1"] and out["losses"] == ["q2"]


def test_paired_outcomes_ignores_questions_missing_from_an_arm():
    out = paired_outcomes({"q1": 0.0, "q2": 0.0}, {"q1": 1.0})
    assert out["n_compared"] == 1


def test_paired_outcomes_restricts_to_eligible():
    a = {"q1": 0.0, "q2": 0.0}
    b = {"q1": 1.0, "q2": 1.0}
    out = paired_outcomes(a, b, eligible={"q1"})
    assert out["n_compared"] == 1 and out["n_wins"] == 1


# --- sign test --------------------------------------------------------------

def test_sign_test_symmetric_split_is_not_significant():
    assert sign_test_p(5, 5) == pytest.approx(1.0)


def test_sign_test_matches_known_exact_values():
    # 8 discordant pairs, all one way: 2 * 0.5^8 = 0.0078125
    assert sign_test_p(8, 0) == pytest.approx(0.0078125)
    # 7 of 8 one way: 2 * (1+8)/256
    assert sign_test_p(7, 1) == pytest.approx(2 * 9 / 256)


def test_sign_test_no_discordant_pairs_is_p_one():
    assert sign_test_p(0, 0) == 1.0


# --- decision rule ----------------------------------------------------------

def _outcomes(wins, losses, ties=0):
    return {"n_wins": wins, "n_losses": losses, "n_ties": ties}


def test_decide_go_requires_margin_and_significance():
    d = decide_paired(_outcomes(8, 0))
    assert d["decision"] == "GO"
    assert d["p_value"] < 0.05


def test_decide_inconclusive_when_margin_met_but_underpowered():
    """3W/0L is a 3:1 margin but p=0.25 — the small-n trap the old gate fell into."""
    d = decide_paired(_outcomes(3, 0))
    assert d["decision"] == "INCONCLUSIVE"
    assert "more questions" in d["reason"]


def test_decide_inconclusive_when_lead_is_too_narrow():
    d = decide_paired(_outcomes(9, 6))
    assert d["decision"] == "INCONCLUSIVE"
    assert "margin" in d["reason"]


def test_decide_no_go_when_b_does_not_lead():
    assert decide_paired(_outcomes(4, 6))["decision"] == "NO-GO"
    assert decide_paired(_outcomes(5, 5))["decision"] == "NO-GO"


def test_decide_no_go_when_everything_ties():
    d = decide_paired(_outcomes(0, 0, ties=12))
    assert d["decision"] == "NO-GO"
    assert "indistinguishable" in d["reason"]


def test_decide_inconclusive_when_nothing_eligible():
    d = decide_paired(_outcomes(0, 0, ties=0))
    assert d["decision"] == "INCONCLUSIVE"
    assert "no eligible questions" in d["reason"]


def test_old_ten_question_gate_would_have_fired_on_one_flip():
    """Regression guard for the rev-1 flaw: n=10, a single question flipping
    produced a +0.10 mean delta and therefore a GO. The paired rule must not."""
    a = {f"q{i}": 0.0 for i in range(10)}
    b = {**a, "q0": 1.0}
    assert mean_delta(a, b) == pytest.approx(0.10)  # the old gate's exact GO threshold
    assert decide_paired(paired_outcomes(a, b))["decision"] == "INCONCLUSIVE"


def test_mean_delta_is_none_without_shared_questions():
    assert mean_delta({}, {}) is None


# --- Regressions found 2026-09-06 by review of the shipped helpers ----------

CONCENTRATED = {
    "per_doc": [
        {"doc": "big.pdf", "data_grids": [
            {"page": p, "rows": 10, "cols": 10,
             "est_tokens": 900, "max_block_tokens": 480} for p in range(1, 9)
        ]},
        {"doc": "mid.pdf", "data_grids": [
            {"page": 1, "rows": 6, "cols": 6,
             "est_tokens": 300, "max_block_tokens": 300},
            {"page": 2, "rows": 5, "cols": 5,
             "est_tokens": 250, "max_block_tokens": 250},
        ]},
        {"doc": "small.pdf", "data_grids": [
            {"page": 1, "rows": 4, "cols": 4,
             "est_tokens": 900, "max_block_tokens": 900},
        ]},
    ]
}


def test_rank_targets_does_not_concentrate_on_one_document():
    """A cell-count sort put 8 of the top 10 in one file — worse than doc-level.

    The sign test treats wins and losses as independent draws over the corpus;
    they are not if most targets come from the same document.
    """
    top = rank_targets(CONCENTRATED, top_n=6)
    from_big = sum(1 for t in top if t["doc"] == "big.pdf")
    assert from_big <= 3
    assert {t["doc"] for t in top} == {"big.pdf", "mid.pdf", "small.pdf"}


def test_rank_targets_still_leads_with_the_densest_grid():
    assert rank_targets(CONCENTRATED)[0]["doc"] == "big.pdf"


def test_rank_targets_filters_on_the_largest_block_not_the_table_total():
    """A big table is emitted as several under-budget blocks, each kept whole.

    Filtering on the whole-table total discarded 10 of the corpus's 22 grids --
    every large multi-row table, i.e. exactly the case where the arms differ most
    -- leaving only small single-block grids that tie.
    """
    kept = rank_targets(CONCENTRATED, max_tokens=512)
    assert kept and all(t["max_block_tokens"] <= 512 for t in kept)
    # big.pdf totals 900 tokens but packs into 480-token blocks -> eligible.
    assert "big.pdf" in {t["doc"] for t in kept}
    # small.pdf is one 900-token block -> genuinely cannot fit.
    assert "small.pdf" not in {t["doc"] for t in kept}


def test_rank_targets_keeps_grids_the_survey_did_not_measure():
    assert len(rank_targets(SURVEY, max_tokens=512)) == 3


def test_rank_targets_falls_back_to_est_tokens_on_an_older_survey():
    old = {"per_doc": [{"doc": "a.pdf", "data_grids": [
        {"page": 1, "rows": 4, "cols": 4, "est_tokens": 900},
        {"page": 2, "rows": 3, "cols": 3, "est_tokens": 100},
    ]}]}
    kept = rank_targets(old, max_tokens=512)
    assert [t["page"] for t in kept] == [2]


def test_validate_probe_row_rejects_bool_and_out_of_range_pages():
    assert any("int" in e for e in validate_probe_row({**GOOD_ROW, "source_page": True}))
    assert any(">= 1" in e for e in validate_probe_row({**GOOD_ROW, "source_page": 0}))
    assert any(">= 1" in e for e in validate_probe_row({**GOOD_ROW, "source_page": -3}))


def test_paired_outcomes_reports_questions_only_one_arm_scored():
    """Silently intersecting the arms is survivorship bias favouring GO."""
    a = {f"q{i}": 0.0 for i in range(10)}
    b = {f"q{i}": 1.0 for i in range(6)}  # B timed out on the 4 hardest
    out = paired_outcomes(a, b)
    assert out["n_compared"] == 6
    assert out["n_missing"] == 4
    assert out["scored_a_only"] == ["q6", "q7", "q8", "q9"]
    assert out["scored_b_only"] == []


def test_paired_outcomes_reports_questions_dropped_by_the_eligible_gate():
    a = {"q1": 0.0, "q2": 0.0}
    b = {"q1": 1.0, "q2": 1.0}
    out = paired_outcomes(a, b, eligible={"q1"})
    assert out["n_ineligible"] == 1 and out["ineligible"] == ["q2"]


def test_mean_delta_averages_the_population_the_decision_used():
    a = {"q1": 0.0, "q2": 1.0}
    b = {"q1": 1.0, "q2": 0.0}
    assert mean_delta(a, b) == 0.0
    assert mean_delta(a, b, eligible={"q1"}) == 1.0
    assert mean_delta(a, b, eligible=set()) is None


def test_corpus_matches_blocks_a_changed_corpus():
    """`eval/corpus_fingerprint.py` hashes segment counts, which arm B changes by
    design — it would flag a correct run. This hashes document content instead."""
    ok, why = corpus_matches("abc123", "abc123")
    assert ok and "matches" in why

    ok, why = corpus_matches("abc123", "def456")
    assert not ok and "added or removed" in why


def test_corpus_matches_warns_but_allows_an_unstamped_evalset():
    ok, why = corpus_matches(None, "abc123")
    assert ok and "cannot verify" in why
    ok, why = corpus_matches("abc123", None)
    assert ok
