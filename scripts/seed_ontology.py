"""Idempotent ontology seeder. Inputs: YAML (custom) + CSV (ICD-10 VN).

Usage:
    uv run python scripts/seed_ontology.py \\
        --yaml data/ontology/custom_terms.yaml \\
        --icd10 data/ontology/icd10_vn.csv

YAML entry → OntologyTerm row with source='custom'.
CSV row    → OntologyTerm row with source='icd10_vn'.
Upsert key: (canonical_norm, source). Re-running is a no-op.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import unicodedata
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select

from src.agentrag.database import AsyncSessionLocal
from src.agentrag.ontology.models import OntologyTerm


def _norm(text: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace. Keep in sync with resolver._norm."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    ascii_only = "".join(
        c for c in decomposed if unicodedata.category(c) != "Mn"
    )
    ascii_only = ascii_only.replace("đ", "d").replace("Đ", "d")
    return " ".join(ascii_only.lower().split())


async def _upsert(
    session,
    *,
    canonical: str,
    synonyms: list[str],
    system_tag: str | None,
    specialty_tags: list[str],
    icd10_code: str | None,
    source: str,
    parent_canonical: str | None = None,
) -> bool:
    """Insert OntologyTerm if (canonical_norm, source) is new. Returns True if inserted."""
    norm = _norm(canonical)
    existing = (
        await session.execute(
            select(OntologyTerm).where(
                OntologyTerm.canonical_norm == norm,
                OntologyTerm.source == source,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        return False
    parent_id = None
    if parent_canonical:
        parent = (
            await session.execute(
                select(OntologyTerm).where(
                    OntologyTerm.canonical_norm == _norm(parent_canonical)
                )
            )
        ).scalar_one_or_none()
        if parent:
            parent_id = parent.id
    session.add(
        OntologyTerm(
            canonical=canonical,
            canonical_norm=norm,
            synonyms=synonyms,
            system_tag=system_tag,
            specialty_tags=specialty_tags,
            icd10_code=icd10_code,
            source=source,
            parent_id=parent_id,
        )
    )
    return True


async def seed_from_yaml(path: str) -> int:
    """Read curated YAML, upsert into DB. Returns count of newly inserted rows."""
    entries: list[dict[str, Any]] = (
        yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    )
    inserted = 0
    async with AsyncSessionLocal() as session:
        for e in entries:
            if await _upsert(
                session,
                canonical=e["canonical"],
                synonyms=list(e.get("synonyms") or []),
                system_tag=e.get("system_tag"),
                specialty_tags=list(e.get("specialty_tags") or []),
                icd10_code=e.get("icd10"),
                source="custom",
                parent_canonical=e.get("parent"),
            ):
                inserted += 1
        await session.commit()
    return inserted


async def seed_from_icd10(path: str) -> int:
    """Read ICD-10 VN CSV. Expected columns: code,name_vi,name_en."""
    if not Path(path).exists():
        return 0
    inserted = 0
    async with AsyncSessionLocal() as session:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = (row.get("code") or "").strip()
                name_vi = (row.get("name_vi") or "").strip()
                name_en = (row.get("name_en") or "").strip()
                if not code or not name_vi:
                    continue
                synonyms = (
                    [name_en] if name_en and name_en != name_vi else []
                )
                if await _upsert(
                    session,
                    canonical=name_vi,
                    synonyms=synonyms,
                    system_tag=None,
                    specialty_tags=[],
                    icd10_code=code,
                    source="icd10_vn",
                ):
                    inserted += 1
        await session.commit()
    return inserted


async def _main(args) -> None:
    yc = await seed_from_yaml(args.yaml) if args.yaml else 0
    ic = await seed_from_icd10(args.icd10) if args.icd10 else 0
    print(f"Seeded: {yc} custom, {ic} icd10_vn")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", default="data/ontology/custom_terms.yaml")
    p.add_argument("--icd10", default="data/ontology/icd10_vn.csv")
    args = p.parse_args()
    asyncio.run(_main(args))
