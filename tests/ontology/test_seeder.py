"""Smoke test for ontology seeder idempotency."""
from __future__ import annotations

import pytest
from sqlalchemy import select

from scripts.seed_ontology import _norm, seed_from_yaml
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.ontology.models import OntologyTerm


@pytest.mark.asyncio
async def test_seed_from_yaml_idempotent(tmp_path):
    yaml_path = tmp_path / "terms.yaml"
    yaml_path.write_text(
        "- canonical: TestSeedTermZZ\n"
        "  synonyms: [tst-zz]\n"
        "  system_tag: tim_mach\n"
        "  specialty_tags: [noi]\n"
        "  icd10: I00.0\n"
    )
    n1 = await seed_from_yaml(str(yaml_path))
    n2 = await seed_from_yaml(str(yaml_path))
    assert n1 == 1
    assert n2 == 0
    async with AsyncSessionLocal() as s:
        rows = (
            await s.execute(
                select(OntologyTerm).where(
                    OntologyTerm.canonical == "TestSeedTermZZ"
                )
            )
        ).scalars().all()
        assert len(rows) == 1
        assert rows[0].icd10_code == "I00.0"
        await s.delete(rows[0])
        await s.commit()


def test_norm_strips_diacritics():
    assert _norm("Đau ngực") == "dau nguc"
    assert _norm("KHÓ thở") == "kho tho"
    assert _norm("") == ""
