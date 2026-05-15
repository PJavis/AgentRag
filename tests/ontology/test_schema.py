"""Smoke test for ResolvedTerm Pydantic schema."""
from src.agentrag.ontology.schema import ResolvedTerm


def test_resolved_term_roundtrip():
    t = ResolvedTerm(
        canonical="Đau ngực",
        synonyms=["chest pain"],
        system_tag=None,
        specialty_tags=["noi", "cap_cuu"],
        icd10_code="R07.4",
        confidence=1.0,
        source="custom",
    )
    d = t.model_dump()
    assert d["canonical"] == "Đau ngực"
    assert d["specialty_tags"] == ["noi", "cap_cuu"]
    assert d["confidence"] == 1.0
    assert d["icd10_code"] == "R07.4"


def test_resolved_term_defaults():
    t = ResolvedTerm(canonical="X")
    assert t.synonyms == []
    assert t.specialty_tags == []
    assert t.system_tag is None
    assert t.icd10_code is None
    assert t.confidence == 1.0
    assert t.source == "custom"
