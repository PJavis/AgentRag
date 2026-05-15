"""Wire-format schema for resolved ontology terms.

Returned by `TermResolver.resolve()` and consumed by SectionTagger,
DomainRouter, and adapter API endpoints.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ResolvedTerm(BaseModel):
    canonical: str
    synonyms: list[str] = Field(default_factory=list)
    system_tag: str | None = None
    specialty_tags: list[str] = Field(default_factory=list)
    icd10_code: str | None = None
    confidence: float = 1.0   # 1.0 = exact/synonym; < 1.0 = fuzzy
    source: str = "custom"
