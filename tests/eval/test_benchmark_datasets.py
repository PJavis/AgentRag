"""Pure dataset normalization — no HF download / deepeval needed."""
from __future__ import annotations

from src.agentrag.eval.benchmark_datasets import (
    DATASETS,
    SUITES,
    EvalExample,
    _as_context_list,
    normalize_row,
)


def test_as_context_list_handles_str_list_dict_none():
    assert _as_context_list("hello") == ["hello"]
    assert _as_context_list(["a", "", "b"]) == ["a", "b"]
    assert _as_context_list([{"text": "t1"}, {"content": "c2"}, {"x": 1}]) == ["t1", "c2"]
    assert _as_context_list(None) == []
    assert _as_context_list("   ") == []


def test_normalize_qac_row_vn():
    row = {"question": "Q?", "answer": "A.", "context": "ctx passage"}
    ex = normalize_row(row, kind="qac", lang="vi", source="vn_bkai", idx=3)
    assert isinstance(ex, EvalExample)
    assert ex.question == "Q?" and ex.reference_answer == "A."
    assert ex.gold_contexts == ["ctx passage"]
    assert ex.lang == "vi" and ex.source == "vn_bkai" and ex.id == "vn_bkai-3"


def test_normalize_ragbench_row_en():
    row = {"id": "x9", "question": "What?", "response": "Because.", "documents": ["d1", "d2"]}
    ex = normalize_row(row, kind="ragbench", lang="en", source="en_covidqa", idx=0)
    assert ex.id == "x9"
    assert ex.reference_answer == "Because."
    assert ex.gold_contexts == ["d1", "d2"]
    assert ex.lang == "en"


def test_registry_and_suites_consistent():
    for name in SUITES["both"]:
        assert name in DATASETS
    assert set(SUITES["both"]) == set(SUITES["vn"]) | set(SUITES["en"])
