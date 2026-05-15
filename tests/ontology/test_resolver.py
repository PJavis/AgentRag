"""Tests for TermResolver. Assumes custom_terms.yaml has been seeded.

To prime DB:
    PYTHONPATH=. uv run python scripts/seed_ontology.py --yaml data/ontology/custom_terms.yaml
"""
from __future__ import annotations

import pytest

from src.agentrag.ontology.resolver import TermResolver, _norm


@pytest.mark.asyncio
async def test_resolver_exact():
    r = TermResolver()
    out = await r.resolve("Đau ngực")
    assert out is not None
    assert out.canonical == "Đau ngực"
    assert "cap_cuu" in out.specialty_tags
    assert out.confidence == 1.0


@pytest.mark.asyncio
async def test_resolver_synonym():
    r = TermResolver()
    out = await r.resolve("chest pain")
    assert out is not None
    assert out.canonical == "Đau ngực"
    assert out.confidence == 1.0


@pytest.mark.asyncio
async def test_resolver_norm_diacritic_insensitive():
    r = TermResolver()
    out = await r.resolve("dau nguc")
    assert out is not None
    assert out.canonical == "Đau ngực"


@pytest.mark.asyncio
async def test_resolver_miss_returns_none():
    r = TermResolver()
    out = await r.resolve("xyzzy_no_such_term_anywhere")
    assert out is None


def test_norm_collapse_whitespace():
    assert _norm("  Đau   NGỰC  ") == "dau nguc"


@pytest.mark.asyncio
async def test_resolver_fuzzy_typo():
    r = TermResolver()
    out = await r.resolve("dauu ngucc")  # one-letter typos
    assert out is not None
    assert out.canonical == "Đau ngực"
    assert 0.45 < out.confidence < 1.0


@pytest.mark.asyncio
async def test_expand_query_adds_synonyms():
    r = TermResolver()
    expanded = await r.expand_query("chest pain trong cấp cứu")
    # Canonical should be appended when a synonym match is found.
    assert "Đau ngực" in expanded


@pytest.mark.asyncio
async def test_find_in_text_returns_terms():
    r = TermResolver()
    hits = await r.find_in_text(
        "Bệnh nhân vào viện vì đau ngực và khó thở."
    )
    canonicals = {t.canonical for t in hits}
    assert "Đau ngực" in canonicals
    assert "Khó thở" in canonicals
