from scripts.mine_preference import (
    MAX_PAIRS_PER_GROUP,
    _normalize_q,
    build_dpo_pairs,
    build_kto_records,
)

_LONG = "x" * 40  # ≥ MIN_ANSWER_CHARS (30)


def _row(q, a, r):
    return {"question": q, "answer": a, "rating": r}


def test_kto_labels_and_skips():
    rows = [
        _row("Q1?", "good " + _LONG, 1),
        _row("Q2?", "bad " + _LONG, -1),
        _row("", "no question " + _LONG, 1),   # skip: empty question
        _row("Q3?", "short", 1),                # skip: answer < 30 chars
        _row("Q4?", "zero rating " + _LONG, 0), # skip: rating not in (1,-1)
    ]
    recs = build_kto_records(rows)
    assert len(recs) == 2
    assert recs[0] == {"prompt": "Q1?", "completion": "good " + _LONG, "label": True}
    assert recs[1]["label"] is False


def test_kto_empty():
    assert build_kto_records([]) == []


def test_dpo_pairs_same_question_normalized():
    rows = [
        _row("What is X?", "chosen " + _LONG, 1),
        _row(" what is x? ", "rejected " + _LONG, -1),  # normalizes into same group
    ]
    pairs = build_dpo_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "chosen " + _LONG
    assert pairs[0]["rejected"] == "rejected " + _LONG
    assert pairs[0]["prompt"] == "What is X?"


def test_dpo_no_pair_single_polarity():
    rows = [_row("Q?", "up1 " + _LONG, 1), _row("Q?", "up2 " + _LONG, 1)]
    assert build_dpo_pairs(rows) == []


def test_dpo_identical_answer_excluded():
    same = "same " + _LONG
    rows = [_row("Q?", same, 1), _row("Q?", same, -1)]
    assert build_dpo_pairs(rows) == []


def test_dpo_cap_per_group():
    rows = [_row("Q?", f"up{i} " + _LONG, 1) for i in range(5)] + \
           [_row("Q?", f"down{i} " + _LONG, -1) for i in range(5)]  # 25 possible
    assert len(build_dpo_pairs(rows)) == MAX_PAIRS_PER_GROUP


def test_normalize_q():
    assert _normalize_q("  Hello   World? ") == "hello world?"
    assert _normalize_q(None) == ""
