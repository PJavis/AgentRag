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


_FUZZY_THRESHOLD = 0.45


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

            # 2. synonym match — case-insensitive JSONB substring (quoted).
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

            # 3. Trigram fuzzy (pg_trgm extension).
            sim = func.similarity(OntologyTerm.canonical_norm, norm)
            fuzzy = (
                await s.execute(
                    select(OntologyTerm, sim.label("sim"))
                    .where(sim > _FUZZY_THRESHOLD)
                    .order_by(sim.desc())
                    .limit(1)
                )
            ).first()
            if fuzzy is not None:
                row, similarity = fuzzy
                return _to_resolved(row, confidence=float(similarity))

        return None

    async def expand_query(self, query: str) -> str:
        """Append canonical + synonym hints to a query string for retrieval."""
        hits = await self.find_in_text(query, max_terms=5)
        extras: list[str] = []
        for h in hits:
            extras.append(h.canonical)
            extras.extend(h.synonyms[:3])
        if not extras:
            return query
        unique = []
        seen = set()
        for e in extras:
            low = e.lower()
            if low not in seen and low not in query.lower():
                seen.add(low)
                unique.append(e)
        if not unique:
            return query
        return f"{query} {' '.join(unique)}"

    async def find_in_text(
        self, text: str, max_terms: int = 10
    ) -> list[ResolvedTerm]:
        """Scan text for known ontology terms via case-insensitive substring.

        Cheap for ≤ a few thousand rows. Loads all terms once per call.
        """
        if not text or not text.strip():
            return []
        text_lower = text.lower()
        async with AsyncSessionLocal() as s:
            all_rows = (
                await s.execute(select(OntologyTerm))
            ).scalars().all()
        hits: list[ResolvedTerm] = []
        seen_ids: set[Any] = set()
        for row in all_rows:
            needles = [row.canonical.lower()] + [
                str(syn).lower() for syn in (row.synonyms or [])
            ]
            if any(n and n in text_lower for n in needles):
                if row.id in seen_ids:
                    continue
                seen_ids.add(row.id)
                hits.append(_to_resolved(row, confidence=1.0))
                if len(hits) >= max_terms:
                    break
        return hits
