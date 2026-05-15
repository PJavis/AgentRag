"""Resolve free-form medical terms → canonical + tags.

Resolution order:
    1. Exact canonical_norm match
    2. Synonym JSONB substring match (case-insensitive)
    3. Trigram fuzzy match (added in T6)
    4. None — caller decides whether to fall back to SLM
"""
from __future__ import annotations

import unicodedata
from typing import Any

from sqlalchemy import String, cast, func, select
from sqlalchemy.dialects.postgresql import JSONB

from src.agentrag.database import AsyncSessionLocal
from src.agentrag.ontology.models import OntologyTerm
from src.agentrag.ontology.schema import ResolvedTerm


def _norm(text: str) -> str:
    """Same normalisation used by the seeder — keep in sync."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    ascii_only = "".join(
        c for c in decomposed if unicodedata.category(c) != "Mn"
    )
    ascii_only = ascii_only.replace("đ", "d").replace("Đ", "d")
    return " ".join(ascii_only.lower().split())


def _to_resolved(row: OntologyTerm, *, confidence: float) -> ResolvedTerm:
    return ResolvedTerm(
        canonical=row.canonical,
        synonyms=list(row.synonyms or []),
        system_tag=row.system_tag,
        specialty_tags=list(row.specialty_tags or []),
        icd10_code=row.icd10_code,
        confidence=confidence,
        source=row.source,
    )


class TermResolver:
    async def resolve(self, term: str) -> ResolvedTerm | None:
        if not term or not term.strip():
            return None
        norm = _norm(term)
        async with AsyncSessionLocal() as s:
            # 1. exact canonical_norm
            row = (
                await s.execute(
                    select(OntologyTerm).where(
                        OntologyTerm.canonical_norm == norm
                    )
                )
            ).scalar_one_or_none()
            if row is not None:
                return _to_resolved(row, confidence=1.0)

            # 2. synonym match — case-insensitive substring in JSONB text.
            # Wrap term in quotes so we match array element, not partial.
            lower_term = term.lower()
            row = (
                await s.execute(
                    select(OntologyTerm).where(
                        func.lower(cast(OntologyTerm.synonyms, String)).ilike(
                            f'%"{lower_term}"%'
                        )
                    )
                )
            ).scalar_one_or_none()
            if row is not None:
                return _to_resolved(row, confidence=1.0)

        return None
