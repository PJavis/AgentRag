import json

import pytest

from scripts.finetune_dpo import load_preference_records

_LONG = "x" * 40


def _write(tmp_path, rows):
    p = tmp_path / "pref.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def test_kto_keeps_valid_and_skips_bad(tmp_path):
    path = _write(tmp_path, [
        {"prompt": "Q1", "completion": "good " + _LONG, "label": True},
        {"prompt": "Q2", "completion": "bad " + _LONG, "label": False},
        {"prompt": "Q3", "completion": "no label " + _LONG},        # skip: no label
        {"prompt": "", "completion": "empty prompt " + _LONG, "label": True},  # skip
        {"prompt": "Q4", "completion": "x", "label": "yes"},         # skip: label not bool
    ])
    rows = load_preference_records(path, "kto")
    assert len(rows) == 2
    assert rows[0] == {"prompt": "Q1", "completion": "good " + _LONG, "label": True}


def test_orpo_keeps_valid_and_skips_identical(tmp_path):
    path = _write(tmp_path, [
        {"prompt": "Q1", "chosen": "c " + _LONG, "rejected": "r " + _LONG},
        {"prompt": "Q2", "chosen": "same", "rejected": "same"},      # skip: identical
        {"prompt": "Q3", "chosen": "only chosen " + _LONG},          # skip: no rejected
    ])
    rows = load_preference_records(path, "orpo")
    assert len(rows) == 1
    assert rows[0]["chosen"] == "c " + _LONG and rows[0]["rejected"] == "r " + _LONG


def test_raises_when_no_valid_rows(tmp_path):
    path = _write(tmp_path, [{"prompt": "", "completion": "", "label": True}])
    with pytest.raises(SystemExit):
        load_preference_records(path, "kto")
