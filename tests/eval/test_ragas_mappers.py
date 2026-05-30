"""Pure RAGAS sample mappers — no ragas/langchain deps required."""
from __future__ import annotations

from src.agentrag.eval.ragas_eval import build_ragas_row, extract_context_texts


def test_extract_context_prefers_content_then_excerpt_then_text():
    items = [
        {"content": "full body", "excerpt": "short"},
        {"excerpt": "only excerpt"},
        {"text": "only text"},
    ]
    assert extract_context_texts(items) == ["full body", "only excerpt", "only text"]


def test_extract_context_accepts_bare_strings_and_drops_empties():
    items = ["passage one", "", {"content": "  "}, {"content": "kept"}]
    assert extract_context_texts(items) == ["passage one", "kept"]


def test_extract_context_handles_none():
    assert extract_context_texts(None) == []


def test_build_ragas_row_uses_canonical_field_names():
    row = build_ragas_row(
        question="Thuốc nào điều trị tăng huyết áp?",
        answer="Amlodipine.",
        context_items=[{"content": "Amlodipine là thuốc chẹn kênh canxi."}],
        ground_truth="Amlodipine",
    )
    assert row == {
        "user_input": "Thuốc nào điều trị tăng huyết áp?",
        "response": "Amlodipine.",
        "retrieved_contexts": ["Amlodipine là thuốc chẹn kênh canxi."],
        "reference": "Amlodipine",
    }


def test_build_ragas_row_defaults_blank_answer_and_reference():
    row = build_ragas_row(question="q", answer="", context_items=None)
    assert row["response"] == ""
    assert row["reference"] == ""
    assert row["retrieved_contexts"] == []
