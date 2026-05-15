"""Smoke test for OntologyTerm table — persist + roundtrip."""
from __future__ import annotations

import pytest
from sqlalchemy import select

from src.agentrag.database import AsyncSessionLocal
from src.agentrag.ontology.models import OntologyTerm


@pytest.mark.asyncio
async def test_ontology_term_persists():
    async with AsyncSessionLocal() as s:
        t = OntologyTerm(
            canonical="TestSmokePersist",
            canonical_norm="testsmokepersist",
            synonyms=["alias-a", "alias-b"],
            system_tag=None,
            specialty_tags=["noi", "cap_cuu"],
            icd10_code="X00.0",
            source="custom",
        )
        s.add(t)
        await s.commit()
        result = await s.execute(
            select(OntologyTerm).where(
                OntologyTerm.canonical_norm == "testsmokepersist"
            )
        )
        row = result.scalar_one()
        assert row.canonical == "TestSmokePersist"
        assert "alias-a" in row.synonyms
        assert "cap_cuu" in row.specialty_tags
        assert row.icd10_code == "X00.0"
        await s.delete(row)
        await s.commit()
