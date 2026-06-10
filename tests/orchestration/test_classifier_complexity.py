import asyncio
from src.agentrag.structured.query_classifier import QueryIntentClassifier, ClassifierOutput


def test_short_factual_question_is_simple():
    c = QueryIntentClassifier()
    out = asyncio.run(c.classify("Triệu chứng của nhồi máu cơ tim là gì?"))
    assert isinstance(out, ClassifierOutput)
    assert out.complexity == "simple"
    assert out.single_domain is True


def test_comparison_question_is_complex():
    c = QueryIntentClassifier()
    out = asyncio.run(c.classify("So sánh nhồi máu cơ tim và đột quỵ về cơ chế và điều trị"))
    # structured comparison → complex, and spans >1 domain
    assert out.complexity == "complex"
