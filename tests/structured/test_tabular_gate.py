import asyncio

from src.agentrag.structured.pipeline import (
    StructuredReasoningPipeline,
    has_tabular_evidence,
)


def test_has_tabular_evidence_detects_excel_sheet_marker():
    assert has_tabular_evidence([{"content": "### Sheet: Doses\nrow data"}])


def test_has_tabular_evidence_detects_csv_fence():
    assert has_tabular_evidence([{"content": "```csv\ndrug,dose\naspirin,100\n```"}])


def test_has_tabular_evidence_detects_markdown_table():
    md = "| Drug | Dose |\n| ---- | ---- |\n| Aspirin | 100mg |"
    assert has_tabular_evidence([{"content": md}])


def test_has_tabular_evidence_detects_html_table():
    assert has_tabular_evidence([{"content": "<table><tr><td>x</td></tr></table>"}])


def test_has_tabular_evidence_detects_segment_type_table():
    assert has_tabular_evidence([{"segment_type": "table", "content": "anything"}])


def test_has_tabular_evidence_false_for_prose():
    assert not has_tabular_evidence(
        [{"content": "Nhồi máu cơ tim là tình trạng tắc nghẽn động mạch vành."}]
    )


def test_has_tabular_evidence_false_for_empty():
    assert not has_tabular_evidence([])
    assert not has_tabular_evidence(None)


def test_pipeline_falls_back_to_semantic_on_prose_corpus():
    """Corpus-aware gate: a structured-intent question over a prose-only corpus
    must short-circuit to the semantic path BEFORE schema discovery."""
    p = StructuredReasoningPipeline.__new__(StructuredReasoningPipeline)

    class _Knowledge:
        async def bootstrap_search(self, query, document_title):  # noqa: ANN001
            return {"query": query}, {"results": [
                {"content": "Aspirin ức chế kết tập tiểu cầu, dùng trong nhồi máu cơ tim."},
                {"content": "Đột quỵ là tổn thương não do thiếu máu cục bộ."},
            ]}

    class _Security:
        def filter_tool_results(self, out, document_title):  # noqa: ANN001
            return out

    p._knowledge = _Knowledge()
    p._security = _Security()

    out = asyncio.run(p.run(
        question="So sánh nhồi máu cơ tim và đột quỵ",
        document_title=None,
        chat_history=None,
        query_type="comparison",
        classifier_confidence=0.95,
    ))
    assert out["_structured_fallback"] is True
    assert out["_fallback_reason"] == "no_tabular_evidence"
