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
