# Medical KB Domain Partition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Partition the medical KB by `system_tag` (hệ cơ quan) × `specialty_tag` (chuyên khoa lâm sàng), add a Postgres ontology table with canonical/synonym mapping (custom YAML + ICD-10 VN seed), tag chunks at section granularity with LLM override, and route queries through a confidence-based federated retriever that respects a UI domain filter.

**Architecture:** 3 KB layers. Layer 1 = `ontology_terms` (Postgres) for canonical Vietnamese medical terms + synonyms. Layer 2 = `agentrag_segments` (ES) with new `system_tag` + `specialty_tag` keyword fields. Layer 3 = `FederatedRetriever` wraps `ElasticsearchRetriever` and consults `DomainRouter` (SLM) to pick domains, with explicit UI override.

**Tech Stack:** Python 3.11, SQLAlchemy 2.0 (async), Alembic, Elasticsearch 8.x, Pydantic v2, FastAPI, Next.js 16, llama3.2:3b for routing/tagging.

**Spec:** `docs/superpowers/specs/2026-05-15-medical-kb-domain-partition-design.md`

---

## Phase 1 — Ontology storage

### Task 1: Add `ontology_terms` table

**Files:**
- Create: `src/agentrag/ontology/__init__.py`
- Create: `src/agentrag/ontology/models.py`
- Create: `migrations/versions/2026_05_15_add_ontology_terms.py`
- Test: `tests/ontology/test_models.py`

- [ ] **Step 1: Write failing test for `OntologyTerm` model fields**

```python
# tests/ontology/test_models.py
import pytest
from sqlalchemy import select
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.ontology.models import OntologyTerm


@pytest.mark.asyncio
async def test_ontology_term_persists():
    async with AsyncSessionLocal() as s:
        t = OntologyTerm(
            canonical="Đau ngực",
            canonical_norm="dau nguc",
            synonyms=["chest pain", "thống tâm"],
            system_tag=None,
            specialty_tags=["noi", "cap_cuu"],
            icd10_code="R07.4",
            source="custom",
        )
        s.add(t)
        await s.commit()
        result = await s.execute(
            select(OntologyTerm).where(OntologyTerm.canonical_norm == "dau nguc")
        )
        row = result.scalar_one()
        assert row.canonical == "Đau ngực"
        assert "chest pain" in row.synonyms
        assert "cap_cuu" in row.specialty_tags
        # cleanup
        await s.delete(row)
        await s.commit()
```

- [ ] **Step 2: Run test, verify it fails**

```bash
uv run pytest tests/ontology/test_models.py -v
```

Expected: ImportError or `relation "ontology_terms" does not exist`.

- [ ] **Step 3: Create the SQLAlchemy model**

```python
# src/agentrag/ontology/__init__.py
"""Medical ontology layer — canonical terms, synonyms, hierarchy."""
```

```python
# src/agentrag/ontology/models.py
"""SQLAlchemy model for the ontology_terms table."""
from __future__ import annotations

import uuid

from sqlalchemy import Column, ForeignKey, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.sql import func

from src.agentrag.database.base import Base


class OntologyTerm(Base):
    __tablename__ = "ontology_terms"
    __table_args__ = {"extend_existing": True}

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    canonical = Column(String(255), nullable=False)
    canonical_norm = Column(String(255), nullable=False, index=True)
    synonyms = Column(JSONB, nullable=False, default=list)
    system_tag = Column(String(32), nullable=True)
    specialty_tags = Column(JSONB, nullable=False, default=list)
    parent_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("ontology_terms.id", ondelete="SET NULL"),
        nullable=True,
    )
    icd10_code = Column(String(16), nullable=True, index=True)
    source = Column(String(16), nullable=False)  # custom | icd10_vn | mesh
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
```

- [ ] **Step 4: Generate Alembic migration**

```bash
cd /home/nguyenquocdung/work/AgentRag
uv run alembic revision -m "add_ontology_terms" --rev-id 2026051501
```

Then edit the generated file to:

```python
# migrations/versions/2026051501_add_ontology_terms.py
"""add ontology_terms table"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "2026051501"
down_revision: Union[str, Sequence[str], None] = "d7e2a4b9c1f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ontology_terms",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("canonical", sa.String(255), nullable=False),
        sa.Column("canonical_norm", sa.String(255), nullable=False),
        sa.Column("synonyms", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("system_tag", sa.String(32), nullable=True),
        sa.Column("specialty_tags", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("parent_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("ontology_terms.id", ondelete="SET NULL"), nullable=True),
        sa.Column("icd10_code", sa.String(16), nullable=True),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_ontology_canonical_norm", "ontology_terms", ["canonical_norm"])
    op.create_index("ix_ontology_icd10", "ontology_terms", ["icd10_code"])
    op.execute("CREATE INDEX ix_ontology_synonyms_gin ON ontology_terms USING GIN (synonyms)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_ontology_synonyms_gin")
    op.drop_index("ix_ontology_icd10", table_name="ontology_terms")
    op.drop_index("ix_ontology_canonical_norm", table_name="ontology_terms")
    op.drop_table("ontology_terms")
```

Replace `down_revision` with current head: `uv run alembic heads` — paste the printed revision id.

- [ ] **Step 5: Apply migration + run test**

```bash
uv run alembic upgrade head
uv run pytest tests/ontology/test_models.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agentrag/ontology/ migrations/versions/2026051501_*.py tests/ontology/test_models.py
git commit -m "feat(ontology): add ontology_terms table + OntologyTerm model"
```

---

### Task 2: Pydantic `ResolvedTerm` schema

**Files:**
- Create: `src/agentrag/ontology/schema.py`
- Test: `tests/ontology/test_schema.py`

- [ ] **Step 1: Write failing test**

```python
# tests/ontology/test_schema.py
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
```

- [ ] **Step 2: Implement schema**

```python
# src/agentrag/ontology/schema.py
"""Wire-format schema for resolved ontology terms."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ResolvedTerm(BaseModel):
    canonical: str
    synonyms: list[str] = Field(default_factory=list)
    system_tag: str | None = None
    specialty_tags: list[str] = Field(default_factory=list)
    icd10_code: str | None = None
    confidence: float = 1.0          # 1.0 exact, < 1.0 fuzzy
    source: str = "custom"
```

- [ ] **Step 3: Run test + commit**

```bash
uv run pytest tests/ontology/test_schema.py -v
git add src/agentrag/ontology/schema.py tests/ontology/test_schema.py
git commit -m "feat(ontology): ResolvedTerm pydantic schema"
```

---

### Task 3: Initial custom ontology YAML

**Files:**
- Create: `data/ontology/custom_terms.yaml`

- [ ] **Step 1: Seed 50 high-frequency Vietnamese medical terms**

```yaml
# data/ontology/custom_terms.yaml — curated canonical terms
# Each entry produces one OntologyTerm row with source='custom'.

- canonical: Đau ngực
  synonyms: [đau tức ngực, đau vùng ngực, thống tâm, chest pain]
  system_tag: null
  specialty_tags: [noi, cap_cuu, tim_mach]
  icd10: R07.4

- canonical: Khó thở
  synonyms: [khó thở khi gắng sức, thở ngắn, dyspnea, shortness of breath]
  system_tag: ho_hap
  specialty_tags: [noi, cap_cuu]
  icd10: R06.0

- canonical: Ho ra máu
  synonyms: [khái huyết, hemoptysis]
  system_tag: ho_hap
  specialty_tags: [noi]
  icd10: R04.2

- canonical: Sốt cao
  synonyms: [sốt, sốt cao liên tục, hyperthermia, fever]
  system_tag: da_he
  specialty_tags: [noi, nhi, truyen_nhiem]
  icd10: R50.9

- canonical: Đau bụng
  synonyms: [đau vùng bụng, abdominal pain]
  system_tag: tieu_hoa
  specialty_tags: [noi, ngoai, cap_cuu]
  icd10: R10.4

- canonical: Tăng huyết áp
  synonyms: [HTA, cao huyết áp, hypertension]
  system_tag: tim_mach
  specialty_tags: [noi]
  icd10: I10

- canonical: Suy tim
  synonyms: [suy tim sung huyết, heart failure, CHF]
  system_tag: tim_mach
  specialty_tags: [noi]
  icd10: I50.9

- canonical: Nhồi máu cơ tim
  synonyms: [NMCT, myocardial infarction, MI, heart attack]
  system_tag: tim_mach
  specialty_tags: [noi, cap_cuu]
  icd10: I21.9

- canonical: Đột quỵ
  synonyms: [tai biến mạch máu não, stroke, CVA]
  system_tag: than_kinh
  specialty_tags: [noi, cap_cuu]
  icd10: I64

- canonical: Viêm phổi
  synonyms: [pneumonia, viêm phổi cộng đồng, CAP]
  system_tag: ho_hap
  specialty_tags: [noi, truyen_nhiem, nhi]
  icd10: J18.9

- canonical: Hen phế quản
  synonyms: [hen, asthma, bronchial asthma]
  system_tag: ho_hap
  specialty_tags: [noi, nhi]
  icd10: J45.9

- canonical: COPD
  synonyms: [bệnh phổi tắc nghẽn mạn tính, chronic obstructive pulmonary disease]
  system_tag: ho_hap
  specialty_tags: [noi]
  icd10: J44.9

- canonical: Đái tháo đường
  synonyms: [tiểu đường, diabetes mellitus, DM, ĐTĐ]
  system_tag: noi_tiet
  specialty_tags: [noi]
  icd10: E11.9

- canonical: Loét dạ dày
  synonyms: [loét dạ dày tá tràng, peptic ulcer, gastric ulcer]
  system_tag: tieu_hoa
  specialty_tags: [noi, ngoai]
  icd10: K25.9

- canonical: Viêm gan B
  synonyms: [HBV, hepatitis B, viêm gan virus B]
  system_tag: tieu_hoa
  specialty_tags: [noi, truyen_nhiem]
  icd10: B18.1

- canonical: Gãy xương
  synonyms: [fracture, gãy xương hở, gãy xương kín]
  system_tag: co_xuong_khop
  specialty_tags: [ngoai, cap_cuu]
  icd10: T14.20

- canonical: Gãy xương dài
  synonyms: [gãy thân xương dài, long bone fracture]
  system_tag: co_xuong_khop
  specialty_tags: [ngoai]
  parent: Gãy xương

- canonical: Trật khớp
  synonyms: [trật khớp vai, dislocation, joint dislocation]
  system_tag: co_xuong_khop
  specialty_tags: [ngoai, cap_cuu]
  icd10: T14.30

- canonical: Viêm khớp
  synonyms: [arthritis, viêm đa khớp]
  system_tag: co_xuong_khop
  specialty_tags: [noi]
  icd10: M13.9

- canonical: Van hai lá
  synonyms: [mitral valve, van hai lá tim]
  system_tag: tim_mach
  specialty_tags: [tim_mach, giai_phau]
  parent: Tim

- canonical: Tim
  synonyms: [heart, cardiac]
  system_tag: tim_mach
  specialty_tags: [giai_phau]
  icd10: null

- canonical: Phổi
  synonyms: [lung, pulmonary]
  system_tag: ho_hap
  specialty_tags: [giai_phau]

- canonical: Gan
  synonyms: [liver, hepatic]
  system_tag: tieu_hoa
  specialty_tags: [giai_phau]

- canonical: Thận
  synonyms: [kidney, renal]
  system_tag: tiet_nieu
  specialty_tags: [giai_phau]

- canonical: Não
  synonyms: [brain, cerebrum, encephalon]
  system_tag: than_kinh
  specialty_tags: [giai_phau]

- canonical: Tủy sống
  synonyms: [spinal cord, medulla spinalis]
  system_tag: than_kinh
  specialty_tags: [giai_phau]

- canonical: Hệ thần kinh chi dưới
  synonyms: [thần kinh chi dưới, lower limb nerves]
  system_tag: than_kinh
  specialty_tags: [giai_phau]
  parent: Tủy sống

- canonical: Dạ dày
  synonyms: [stomach, gastric]
  system_tag: tieu_hoa
  specialty_tags: [giai_phau]

- canonical: Sonde tiểu
  synonyms: [đặt sonde tiểu, urinary catheter, catheter]
  system_tag: tiet_nieu
  specialty_tags: [ngoai, cap_cuu]

- canonical: Thuốc giảm đau
  synonyms: [analgesic, painkiller, NSAID, paracetamol]
  system_tag: null
  specialty_tags: [duoc_ly, noi]

- canonical: Opioid
  synonyms: [morphin, opiate, narcotic]
  system_tag: null
  specialty_tags: [duoc_ly, cap_cuu]
  parent: Thuốc giảm đau

- canonical: Paracetamol
  synonyms: [acetaminophen, tylenol]
  system_tag: null
  specialty_tags: [duoc_ly]
  parent: Thuốc giảm đau

- canonical: Kháng sinh
  synonyms: [antibiotic, kháng khuẩn]
  system_tag: null
  specialty_tags: [duoc_ly, truyen_nhiem]

- canonical: Sốc
  synonyms: [shock, sốc nhiễm khuẩn, sốc giảm thể tích]
  system_tag: da_he
  specialty_tags: [cap_cuu, hoi_suc]
  icd10: R57.9

- canonical: Sốc phản vệ
  synonyms: [anaphylactic shock, anaphylaxis]
  system_tag: mien_dich
  specialty_tags: [cap_cuu]
  icd10: T78.2
  parent: Sốc

- canonical: Ngộ độc
  synonyms: [poisoning, intoxication]
  system_tag: da_he
  specialty_tags: [cap_cuu, noi]
  icd10: T65.9

- canonical: Chấn thương sọ não
  synonyms: [TBI, head injury, traumatic brain injury]
  system_tag: than_kinh
  specialty_tags: [ngoai, cap_cuu]
  icd10: S06.9

- canonical: Khám lâm sàng
  synonyms: [physical examination, thăm khám]
  system_tag: null
  specialty_tags: [general]

- canonical: Tiền sử bệnh
  synonyms: [bệnh sử, medical history, past medical history]
  system_tag: null
  specialty_tags: [general]

- canonical: Xét nghiệm máu
  synonyms: [blood test, CBC, công thức máu]
  system_tag: huyet_hoc
  specialty_tags: [xet_nghiem]

- canonical: X-quang
  synonyms: [X-ray, chụp X quang, radiograph]
  system_tag: null
  specialty_tags: [chan_doan_hinh_anh]

- canonical: CT scan
  synonyms: [chụp CT, computed tomography]
  system_tag: null
  specialty_tags: [chan_doan_hinh_anh]

- canonical: MRI
  synonyms: [chụp cộng hưởng từ, magnetic resonance imaging]
  system_tag: null
  specialty_tags: [chan_doan_hinh_anh]

- canonical: Siêu âm
  synonyms: [ultrasound, sonography]
  system_tag: null
  specialty_tags: [chan_doan_hinh_anh]

- canonical: Điện tâm đồ
  synonyms: [ECG, EKG, electrocardiogram]
  system_tag: tim_mach
  specialty_tags: [xet_nghiem, chan_doan_hinh_anh]

- canonical: Hồi sức tim phổi
  synonyms: [CPR, cardiopulmonary resuscitation, hồi sinh tim phổi]
  system_tag: da_he
  specialty_tags: [cap_cuu, hoi_suc]

- canonical: Đặt nội khí quản
  synonyms: [intubation, đặt ống nội khí quản, NKQ]
  system_tag: ho_hap
  specialty_tags: [cap_cuu, hoi_suc]

- canonical: Truyền dịch
  synonyms: [IV fluids, dịch truyền, intravenous fluids]
  system_tag: null
  specialty_tags: [cap_cuu, noi, hoi_suc]

- canonical: Thai kỳ
  synonyms: [pregnancy, có thai, mang thai]
  system_tag: sinh_duc
  specialty_tags: [san]

- canonical: Cấp cứu sản khoa
  synonyms: [obstetric emergency, băng huyết, tiền sản giật]
  system_tag: sinh_duc
  specialty_tags: [san, cap_cuu]
```

- [ ] **Step 2: Commit**

```bash
mkdir -p data/ontology
git add data/ontology/custom_terms.yaml
git commit -m "data(ontology): seed 50 Vietnamese medical canonical terms"
```

---

### Task 4: ICD-10 seeder script

**Files:**
- Create: `scripts/seed_ontology.py`
- Test: `tests/ontology/test_seeder.py`

- [ ] **Step 1: Write failing test for seeder**

```python
# tests/ontology/test_seeder.py
import pytest
from sqlalchemy import select
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.ontology.models import OntologyTerm
from scripts.seed_ontology import seed_from_yaml


@pytest.mark.asyncio
async def test_seed_from_yaml_idempotent(tmp_path):
    yaml_path = tmp_path / "terms.yaml"
    yaml_path.write_text(
        "- canonical: TestSeedTerm\n  synonyms: [tst]\n  system_tag: tim_mach\n"
        "  specialty_tags: [noi]\n  icd10: I00.0\n"
    )
    # First run inserts
    n1 = await seed_from_yaml(str(yaml_path))
    # Second run is a no-op (upsert by canonical_norm + source)
    n2 = await seed_from_yaml(str(yaml_path))
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(
            select(OntologyTerm).where(OntologyTerm.canonical == "TestSeedTerm")
        )).scalars().all()
        assert len(rows) == 1
        assert rows[0].icd10_code == "I00.0"
        # Cleanup
        await s.delete(rows[0])
        await s.commit()
    assert n1 == 1
    assert n2 == 0
```

- [ ] **Step 2: Run test, verify it fails**

```bash
uv run pytest tests/ontology/test_seeder.py -v
```

Expected: ImportError on `seed_from_yaml`.

- [ ] **Step 3: Implement seeder**

```python
# scripts/seed_ontology.py
"""Idempotent ontology seeder. Inputs: YAML (custom) + CSV (ICD-10 VN).

Usage:
    uv run python scripts/seed_ontology.py \\
        --yaml data/ontology/custom_terms.yaml \\
        --icd10 data/ontology/icd10_vn.csv

Each YAML entry → one OntologyTerm row with source='custom'.
Each CSV row → one OntologyTerm row with source='icd10_vn'.
Upsert by (canonical_norm, source).
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
    """Lowercase, strip diacritics, collapse whitespace for fuzzy index."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    ascii_only = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    # Vietnamese đ/Đ → d
    ascii_only = ascii_only.replace("đ", "d").replace("Đ", "d")
    return " ".join(ascii_only.lower().split())


async def _upsert(
    session, *, canonical: str, synonyms: list[str], system_tag: str | None,
    specialty_tags: list[str], icd10_code: str | None, source: str,
    parent_canonical: str | None = None,
) -> bool:
    """Returns True if newly inserted, False if already existed."""
    norm = _norm(canonical)
    existing = (await session.execute(
        select(OntologyTerm).where(
            OntologyTerm.canonical_norm == norm,
            OntologyTerm.source == source,
        )
    )).scalar_one_or_none()
    if existing is not None:
        return False
    parent_id = None
    if parent_canonical:
        parent = (await session.execute(
            select(OntologyTerm).where(OntologyTerm.canonical_norm == _norm(parent_canonical))
        )).scalar_one_or_none()
        if parent:
            parent_id = parent.id
    session.add(OntologyTerm(
        canonical=canonical,
        canonical_norm=norm,
        synonyms=synonyms,
        system_tag=system_tag,
        specialty_tags=specialty_tags,
        icd10_code=icd10_code,
        source=source,
        parent_id=parent_id,
    ))
    return True


async def seed_from_yaml(path: str) -> int:
    """Read curated YAML, return count of newly inserted rows."""
    entries: list[dict[str, Any]] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
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
    """Read ICD-10 VN CSV. Expected columns: code,name_vi,name_en (header row)."""
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
                synonyms = [name_en] if name_en and name_en != name_vi else []
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
    asyncio.run(_main(p.parse_args()))
```

- [ ] **Step 4: Run test, verify it passes**

```bash
uv run pytest tests/ontology/test_seeder.py -v
```

- [ ] **Step 5: Seed custom YAML against live DB**

```bash
uv run python scripts/seed_ontology.py --yaml data/ontology/custom_terms.yaml
```

Expected: prints `Seeded: 50 custom, 0 icd10_vn` (or close — count depends on YAML entries).

- [ ] **Step 6: Commit**

```bash
git add scripts/seed_ontology.py tests/ontology/test_seeder.py
git commit -m "feat(ontology): idempotent seeder (YAML + ICD-10 CSV)"
```

---

### Task 5: `TermResolver` — exact + synonym lookup

**Files:**
- Create: `src/agentrag/ontology/resolver.py`
- Test: `tests/ontology/test_resolver.py`

- [ ] **Step 1: Write failing tests for exact + synonym resolution**

```python
# tests/ontology/test_resolver.py
import pytest
from src.agentrag.ontology.resolver import TermResolver


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
    assert await r.resolve("xyzzy_no_such_term") is None
```

- [ ] **Step 2: Run test, verify failure**

```bash
uv run pytest tests/ontology/test_resolver.py -v
```

- [ ] **Step 3: Implement resolver — exact + synonym only**

```python
# src/agentrag/ontology/resolver.py
"""Resolve free-form medical terms → canonical + tags.

Resolution strategy (in order):
  1. Exact canonical_norm match
  2. Synonym JSONB contains match (case-insensitive)
  3. (Task 6) trigram fuzzy match
  4. (Task 6) optional SLM fallback for unmatched terms
"""
from __future__ import annotations

import unicodedata
from typing import Any

from sqlalchemy import select, func, or_, cast
from sqlalchemy.dialects.postgresql import JSONB

from src.agentrag.database import AsyncSessionLocal
from src.agentrag.ontology.models import OntologyTerm
from src.agentrag.ontology.schema import ResolvedTerm


def _norm(text: str) -> str:
    """Same normalisation used by the seeder — keep in sync."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    ascii_only = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    ascii_only = ascii_only.replace("đ", "d").replace("Đ", "d")
    return " ".join(ascii_only.lower().split())


class TermResolver:
    async def resolve(self, term: str) -> ResolvedTerm | None:
        if not term or not term.strip():
            return None
        norm = _norm(term)
        async with AsyncSessionLocal() as s:
            # 1. exact canonical_norm
            row = (await s.execute(
                select(OntologyTerm).where(OntologyTerm.canonical_norm == norm)
            )).scalar_one_or_none()
            if row is None:
                # 2. synonym match — JSONB contains [term] case-insensitive
                # Use raw SQL because JSONB containment is exact-cased; lower it.
                row = (await s.execute(
                    select(OntologyTerm).where(
                        func.lower(cast(OntologyTerm.synonyms, JSONB)).cast(str).ilike(f'%"{term.lower()}"%')
                    )
                )).scalar_one_or_none()
            if row is None:
                return None
        return ResolvedTerm(
            canonical=row.canonical,
            synonyms=row.synonyms or [],
            system_tag=row.system_tag,
            specialty_tags=row.specialty_tags or [],
            icd10_code=row.icd10_code,
            confidence=1.0,
            source=row.source,
        )
```

- [ ] **Step 4: Run test + commit**

```bash
uv run pytest tests/ontology/test_resolver.py -v
git add src/agentrag/ontology/resolver.py tests/ontology/test_resolver.py
git commit -m "feat(ontology): TermResolver exact + synonym lookup"
```

---

### Task 6: `TermResolver` — fuzzy + `expand_query` + `find_in_text`

**Files:**
- Modify: `src/agentrag/ontology/resolver.py`
- Modify: `tests/ontology/test_resolver.py`

- [ ] **Step 1: Enable pg_trgm extension via migration**

```bash
cd /home/nguyenquocdung/work/AgentRag
uv run alembic revision -m "enable_pg_trgm" --rev-id 2026051502
```

Edit generated file:

```python
# migrations/versions/2026051502_enable_pg_trgm.py
from alembic import op

revision = "2026051502"
down_revision = "2026051501"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE INDEX IF NOT EXISTS ix_ontology_canonical_trgm "
               "ON ontology_terms USING GIN (canonical_norm gin_trgm_ops)")


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_ontology_canonical_trgm")
    # leave pg_trgm in place — other code may rely on it
```

```bash
uv run alembic upgrade head
```

- [ ] **Step 2: Write failing tests for fuzzy + expand + find**

```python
# Append to tests/ontology/test_resolver.py
@pytest.mark.asyncio
async def test_resolver_fuzzy_typo():
    r = TermResolver()
    out = await r.resolve("dauu ngucc")  # one-letter typos
    assert out is not None
    assert out.canonical == "Đau ngực"
    assert 0.5 < out.confidence < 1.0


@pytest.mark.asyncio
async def test_expand_query_adds_synonyms():
    r = TermResolver()
    expanded = await r.expand_query("chest pain trong cấp cứu")
    assert "Đau ngực" in expanded


@pytest.mark.asyncio
async def test_find_in_text_returns_terms():
    r = TermResolver()
    hits = await r.find_in_text("Bệnh nhân vào viện vì đau ngực và khó thở.")
    canonicals = {t.canonical for t in hits}
    assert "Đau ngực" in canonicals
    assert "Khó thở" in canonicals
```

- [ ] **Step 3: Extend resolver**

```python
# Modify src/agentrag/ontology/resolver.py — add to TermResolver class

    async def resolve(self, term: str) -> ResolvedTerm | None:
        if not term or not term.strip():
            return None
        norm = _norm(term)
        async with AsyncSessionLocal() as s:
            row = (await s.execute(
                select(OntologyTerm).where(OntologyTerm.canonical_norm == norm)
            )).scalar_one_or_none()
            if row is not None:
                return _to_resolved(row, confidence=1.0)

            row = (await s.execute(
                select(OntologyTerm).where(
                    func.lower(cast(OntologyTerm.synonyms, JSONB)).cast(str).ilike(f'%"{term.lower()}"%')
                )
            )).scalar_one_or_none()
            if row is not None:
                return _to_resolved(row, confidence=1.0)

            # 3. trigram fuzzy (requires pg_trgm extension)
            rows = (await s.execute(
                select(
                    OntologyTerm,
                    func.similarity(OntologyTerm.canonical_norm, norm).label("sim"),
                ).where(
                    func.similarity(OntologyTerm.canonical_norm, norm) > 0.55
                ).order_by(func.similarity(OntologyTerm.canonical_norm, norm).desc()).limit(1)
            )).first()
            if rows is None:
                return None
            row, sim = rows
            return _to_resolved(row, confidence=float(sim))

    async def expand_query(self, query: str) -> str:
        """Append canonical + synonym hints to a query string for retrieval."""
        hits = await self.find_in_text(query, max_terms=5)
        extras: list[str] = []
        for h in hits:
            extras.append(h.canonical)
            extras.extend(h.synonyms[:3])
        if not extras:
            return query
        return f"{query} {' '.join(set(extras))}"

    async def find_in_text(self, text: str, max_terms: int = 10) -> list[ResolvedTerm]:
        """Detect known ontology terms in a chunk of text.

        Heuristic: scan all canonical terms and synonyms, look for case-
        insensitive substring presence. Cheap for ≤ a few thousand terms.
        """
        text_lower = text.lower()
        async with AsyncSessionLocal() as s:
            all_rows = (await s.execute(select(OntologyTerm))).scalars().all()
        hits: list[ResolvedTerm] = []
        seen_ids = set()
        for row in all_rows:
            needles = [row.canonical.lower()] + [str(syn).lower() for syn in (row.synonyms or [])]
            if any(n and n in text_lower for n in needles):
                if row.id in seen_ids:
                    continue
                seen_ids.add(row.id)
                hits.append(_to_resolved(row, confidence=1.0))
                if len(hits) >= max_terms:
                    break
        return hits


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
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/ontology/test_resolver.py -v
git add migrations/versions/2026051502_*.py src/agentrag/ontology/resolver.py tests/ontology/test_resolver.py
git commit -m "feat(ontology): fuzzy match + expand_query + find_in_text"
```

---

## Phase 2 — Ingestion tagging

### Task 7: ES mapping — add domain tag fields

**Files:**
- Modify: `src/agentrag/ingestion/stores/elasticsearch_store.py`

- [ ] **Step 1: Find the existing `index_segments` / mapping block**

```bash
grep -n "system_tag\|properties\|create.*index" src/agentrag/ingestion/stores/elasticsearch_store.py | head -20
```

- [ ] **Step 2: Add `system_tag`, `specialty_tag`, `canonical_terms` to mapping**

Inside the `ensure_index` (or equivalent) function, add to the `properties` dict:

```python
"system_tag":      {"type": "keyword"},
"specialty_tag":   {"type": "keyword"},
"canonical_terms": {"type": "keyword"},
```

And in `index_segments` (or wherever segment doc is built), pass through:

```python
doc = {
    # ... existing fields ...
    "system_tag":      chunk.get("system_tag"),
    "specialty_tag":   chunk.get("specialty_tag", []),
    "canonical_terms": chunk.get("canonical_terms", []),
}
```

- [ ] **Step 3: For existing indices, run a `_mapping` PUT manually**

```bash
curl -X PUT "http://localhost:9200/agentrag_segments/_mapping" -H 'Content-Type: application/json' -d '{
  "properties": {
    "system_tag":      {"type": "keyword"},
    "specialty_tag":   {"type": "keyword"},
    "canonical_terms": {"type": "keyword"}
  }
}'
```

Expected: `{"acknowledged":true}`.

- [ ] **Step 4: Commit**

```bash
git add src/agentrag/ingestion/stores/elasticsearch_store.py
git commit -m "feat(ingestion): add system_tag/specialty_tag fields to ES mapping"
```

---

### Task 8: `SectionTagger` — section_path → tags

**Files:**
- Create: `src/agentrag/ingestion/section_tagger.py`
- Test: `tests/ingestion/test_section_tagger.py`

- [ ] **Step 1: Write failing test for section parsing**

```python
# tests/ingestion/test_section_tagger.py
import pytest
from src.agentrag.ingestion.section_tagger import SectionTagger


@pytest.mark.asyncio
async def test_tagger_uses_section_path():
    tagger = SectionTagger()
    chunk = {
        "section_path": "Chương 3 / Hệ tim mạch / Tim",
        "content": "Tim gồm bốn buồng…",
    }
    out = await tagger.tag_chunk(chunk)
    assert out["system_tag"] == "tim_mach"
    assert "Tim" in out["canonical_terms"]


@pytest.mark.asyncio
async def test_tagger_generic_heading_falls_back_to_content():
    tagger = SectionTagger()
    chunk = {
        "section_path": "Tổng quan",
        "content": "Bệnh nhân được chẩn đoán nhồi máu cơ tim cấp.",
    }
    out = await tagger.tag_chunk(chunk)
    assert out["system_tag"] == "tim_mach"
```

- [ ] **Step 2: Implement**

```python
# src/agentrag/ingestion/section_tagger.py
"""Assign system_tag + specialty_tag + canonical_terms to each chunk.

Strategy:
  1. Parse section_path into segments split by '/'.
  2. Resolve each segment via TermResolver. Most specific child wins.
  3. If section_path is generic (e.g. "Tổng quan", "Mở đầu", "Phần 1") —
     ResolvedTerm comes back empty → fall back to scanning chunk content
     for any known term and aggregate.
  4. canonical_terms = unique canonical strings detected anywhere.
"""
from __future__ import annotations

import re
from typing import Any

from src.agentrag.ontology.resolver import TermResolver


_GENERIC_HEADINGS = {
    "tong quan", "mo dau", "gioi thieu", "phan 1", "phan 2", "phan 3",
    "muc 1", "muc 2", "chuong 1", "chuong 2", "chuong 3",
}


def _norm(s: str) -> str:
    import unicodedata
    d = unicodedata.normalize("NFD", s or "")
    a = "".join(c for c in d if unicodedata.category(c) != "Mn")
    a = a.replace("đ", "d").replace("Đ", "d")
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", a.lower()).split())


class SectionTagger:
    def __init__(self, resolver: TermResolver | None = None) -> None:
        self._resolver = resolver or TermResolver()

    async def tag_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        section_path: str = chunk.get("section_path") or ""
        content: str = chunk.get("content") or ""

        system_tag: str | None = None
        specialty_set: set[str] = set()
        canonical_set: set[str] = set()

        # 1. Walk section_path; later segments are more specific → overwrite.
        for seg in [s.strip() for s in section_path.split("/") if s.strip()]:
            if _norm(seg) in _GENERIC_HEADINGS:
                continue
            r = await self._resolver.resolve(seg)
            if r is None:
                continue
            if r.system_tag:
                system_tag = r.system_tag
            specialty_set.update(r.specialty_tags)
            canonical_set.add(r.canonical)

        # 2. If section_path didn't yield a system_tag, scan content body.
        if system_tag is None:
            content_hits = await self._resolver.find_in_text(content, max_terms=5)
            for h in content_hits:
                if system_tag is None and h.system_tag:
                    system_tag = h.system_tag
                specialty_set.update(h.specialty_tags)
                canonical_set.add(h.canonical)

        return {
            **chunk,
            "system_tag": system_tag,
            "specialty_tag": sorted(specialty_set),
            "canonical_terms": sorted(canonical_set),
        }
```

- [ ] **Step 3: Run test + commit**

```bash
uv run pytest tests/ingestion/test_section_tagger.py -v
git add src/agentrag/ingestion/section_tagger.py tests/ingestion/test_section_tagger.py
git commit -m "feat(ingestion): SectionTagger maps section_path + content → tags"
```

---

### Task 9: Wire `SectionTagger` into ingest pipeline (feature-flagged)

**Files:**
- Modify: `src/agentrag/config.py` (add `TAGGING_ENABLED`)
- Modify: `src/agentrag/ingestion/pipeline.py`
- Modify: `.env` and `.env.example`

- [ ] **Step 1: Add config flag**

In `src/agentrag/config.py`, after existing `STRUCTMEM_ENABLED` (or any ingest flag), add:

```python
    TAGGING_ENABLED: bool = True         # SectionTagger inserts domain tags during ingest
```

- [ ] **Step 2: Modify pipeline to call SectionTagger after chunking**

In `src/agentrag/ingestion/pipeline.py`, find the block that builds `chunks_search` then embeds. Just before the embed call, insert:

```python
if settings.TAGGING_ENABLED:
    from src.agentrag.ingestion.section_tagger import SectionTagger
    _tagger = SectionTagger()
    chunks_search = [await _tagger.tag_chunk(c) for c in chunks_search]
```

- [ ] **Step 3: Add to .env / .env.example**

Add under "STRUCTMEM" or a new "TAGGING" section:

```env
# Domain tagging (S5) — populate system_tag/specialty_tag on every chunk
TAGGING_ENABLED=true
```

Both files.

- [ ] **Step 4: Spot-check with a small ingest**

```bash
# Stop dev, restart so config flag is picked up
make stop || true
# Re-ingest one PDF from data/docs
curl -X POST http://localhost:8000/ingest/upload -F "file=@data/docs/<some_small.pdf>"
```

Then verify:

```bash
curl -s "http://localhost:9200/agentrag_segments/_search?size=3&q=*" | jq '.hits.hits[]._source | {section_path, system_tag, specialty_tag, canonical_terms}'
```

Expected: at least some segments have non-null `system_tag`.

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/config.py src/agentrag/ingestion/pipeline.py .env.example
git commit -m "feat(ingestion): wire SectionTagger into pipeline (TAGGING_ENABLED)"
```

---

### Task 10: Backfill tags for existing segments

**Files:**
- Create: `scripts/backfill_tags.py`
- Test: `tests/scripts/test_backfill_tags.py`

- [ ] **Step 1: Write a smoke test that exercises a single chunk**

```python
# tests/scripts/test_backfill_tags.py
import pytest
from scripts.backfill_tags import _tag_one_segment


@pytest.mark.asyncio
async def test_tag_one_segment_returns_tagged_doc():
    seg = {
        "id": "test123",
        "section_path": "Chương 3 / Hệ tim mạch / Van hai lá",
        "content": "Van hai lá đóng vai trò...",
    }
    out = await _tag_one_segment(seg)
    assert out["system_tag"] == "tim_mach"
    assert "Van hai lá" in out["canonical_terms"]
```

- [ ] **Step 2: Implement script**

```python
# scripts/backfill_tags.py
"""Re-tag every existing segment in Elasticsearch with system_tag /
specialty_tag / canonical_terms derived from section_path + content.

Idempotent: re-running just refreshes tags.

Usage:
    uv run python scripts/backfill_tags.py
    uv run python scripts/backfill_tags.py --batch 200 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Any

from elasticsearch import AsyncElasticsearch

from src.agentrag.config import settings
from src.agentrag.ingestion.section_tagger import SectionTagger


async def _tag_one_segment(seg: dict[str, Any]) -> dict[str, Any]:
    tagger = SectionTagger()
    return await tagger.tag_chunk(seg)


async def _main(batch: int, dry_run: bool) -> None:
    es = AsyncElasticsearch(hosts=[settings.ELASTICSEARCH_URL])
    tagger = SectionTagger()
    try:
        scroll = await es.search(
            index=settings.ELASTICSEARCH_INDEX_NAME,
            body={"query": {"match_all": {}}, "size": batch},
            scroll="2m",
        )
        sid = scroll["_scroll_id"]
        total = scroll["hits"]["total"]["value"]
        print(f"Backfilling {total} segments (batch={batch}, dry_run={dry_run})")
        updated = 0
        while True:
            hits = scroll["hits"]["hits"]
            if not hits:
                break
            actions: list[dict[str, Any]] = []
            for hit in hits:
                src = hit["_source"]
                tagged = await tagger.tag_chunk(src)
                if dry_run:
                    continue
                actions.append({"update": {"_index": hit["_index"], "_id": hit["_id"]}})
                actions.append({"doc": {
                    "system_tag": tagged.get("system_tag"),
                    "specialty_tag": tagged.get("specialty_tag", []),
                    "canonical_terms": tagged.get("canonical_terms", []),
                }})
                updated += 1
            if actions:
                await es.bulk(operations=actions, refresh=False)
            scroll = await es.scroll(scroll_id=sid, scroll="2m")
        print(f"Done. {updated} segments updated.")
    finally:
        await es.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=200)
    p.add_argument("--dry-run", action="store_true")
    asyncio.run(_main(p.parse_args().batch, p.parse_args().dry_run))
```

- [ ] **Step 3: Run dry-run, then real run**

```bash
uv run pytest tests/scripts/test_backfill_tags.py -v
uv run python scripts/backfill_tags.py --dry-run
uv run python scripts/backfill_tags.py
```

- [ ] **Step 4: Verify a tagged segment**

```bash
curl -s "http://localhost:9200/agentrag_segments/_search?size=1&q=system_tag:tim_mach" | jq '.hits.hits[0]._source | {title:.document_title, section_path, system_tag}'
```

Expected: at least one hit (depends on corpus).

- [ ] **Step 5: Commit**

```bash
git add scripts/backfill_tags.py tests/scripts/test_backfill_tags.py
git commit -m "feat(scripts): backfill_tags batch updates existing segments"
```

---

## Phase 3 — Retrieval federation

### Task 11: Extend `ElasticsearchRetriever.search` with `filters`

**Files:**
- Modify: `src/agentrag/retrieval/elasticsearch_retriever.py`
- Test: `tests/retrieval/test_filters.py`

- [ ] **Step 1: Write failing test**

```python
# tests/retrieval/test_filters.py
import pytest
from src.agentrag.retrieval.elasticsearch_retriever import ElasticsearchRetriever


@pytest.mark.asyncio
async def test_search_with_system_filter_only_returns_matching_system():
    r = ElasticsearchRetriever()
    out = await r.search(
        query="đau ngực",
        mode="hybrid",
        top_k=5,
        filters={"systems": ["tim_mach"]},
    )
    # All hits should have system_tag == tim_mach (or null pre-backfill segments)
    for hit in out.get("results", []):
        st = hit.get("system_tag")
        assert st is None or st == "tim_mach", f"expected tim_mach got {st}"
```

- [ ] **Step 2: Read current signature**

```bash
grep -n "async def search" src/agentrag/retrieval/elasticsearch_retriever.py | head -3
```

- [ ] **Step 3: Add `filters` kwarg and inject `terms` query clauses**

In `search()`, accept `filters: dict | None = None`. Inside the ES query body (wherever `bool: {must: [...]}` lives) append filter clauses:

```python
def _filter_clauses(filters: dict | None) -> list[dict]:
    if not filters:
        return []
    clauses = []
    systems = filters.get("systems") or []
    if systems:
        clauses.append({"terms": {"system_tag": systems}})
    specs = filters.get("specialties") or []
    if specs:
        clauses.append({"terms": {"specialty_tag": specs}})
    return clauses

# inside search:
body["query"]["bool"]["filter"] = body["query"]["bool"].get("filter", []) + _filter_clauses(filters)
```

(Adjust to the actual query-building code path. Apply same to sparse + dense + hybrid branches.)

- [ ] **Step 4: Run test + commit**

```bash
uv run pytest tests/retrieval/test_filters.py -v
git add src/agentrag/retrieval/elasticsearch_retriever.py tests/retrieval/test_filters.py
git commit -m "feat(retrieval): ElasticsearchRetriever.search accepts filters"
```

---

### Task 12: `DomainRouter` — SLM classify query → systems + specialties

**Files:**
- Create: `src/agentrag/orchestration/__init__.py`
- Create: `src/agentrag/orchestration/domain_router.py`
- Create: `src/agentrag/orchestration/__init__.py` (empty)
- Test: `tests/orchestration/test_domain_router.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/orchestration/test_domain_router.py
import pytest
from src.agentrag.orchestration.domain_router import DomainRouter, DomainRoute


@pytest.mark.asyncio
async def test_classify_returns_route():
    router = DomainRouter()
    r = await router.classify("Đau ngực kèm khó thở ở bệnh nhân 60 tuổi")
    assert isinstance(r, DomainRoute)
    assert "tim_mach" in r.systems or "ho_hap" in r.systems
    assert 0.0 <= r.confidence <= 1.0


@pytest.mark.asyncio
async def test_low_confidence_expands_to_top_k():
    router = DomainRouter()
    # ambiguous query — confidence should be lower, return multiple
    r = await router.classify("triệu chứng chung của người lớn tuổi")
    if r.confidence < 0.7:
        assert len(r.systems) <= 3  # top-3 cap
```

- [ ] **Step 2: Implement**

```python
# src/agentrag/orchestration/__init__.py
"""Orchestration: routers + planners that decide where to send a query."""
```

```python
# src/agentrag/orchestration/domain_router.py
"""SLM-driven classifier: free-text query → set of medical domains.

Uses LLMGateway.json_response with task="domain_router" so users can route
this call to a cheap model via LLM_TASK_MODEL_MAP.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.agentrag.config import settings
from src.agentrag.services.llm_gateway import LLMGateway

SYSTEM_PROMPT = """\
You are a medical domain router for Vietnamese medical content.
Given a query, identify which body system(s) and clinical specialty(s)
are most relevant. Vietnamese taxonomy:

systems: tim_mach, ho_hap, tieu_hoa, than_kinh, noi_tiet, co_xuong_khop,
huyet_hoc, tiet_nieu, sinh_duc, da_lieu, mat_tmh, tam_than, mien_dich,
nhi_khoa, da_he

specialties: noi, ngoai, san, nhi, cap_cuu, hoi_suc, truyen_nhiem,
ung_buou, chan_doan_hinh_anh, xet_nghiem, duoc_ly, giai_phau,
sinh_ly_benh, general

Return JSON exactly:
{
  "systems":    ["...up to 3 most relevant..."],
  "specialties":["...up to 3..."],
  "confidence": 0.0
}
Confidence ~1.0 = single clear domain; ~0.5 = multi-system / ambiguous.
"""


@dataclass
class DomainRoute:
    systems: list[str]
    specialties: list[str]
    confidence: float
    raw: dict = field(default_factory=dict)


class DomainRouter:
    def __init__(self) -> None:
        self._gateway = LLMGateway()

    async def classify(self, query: str) -> DomainRoute:
        try:
            payload, _ = await self._gateway.json_response(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=json.dumps({"query": query}, ensure_ascii=False),
                task="domain_router",
            )
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        systems = [s for s in payload.get("systems") or [] if isinstance(s, str)]
        specs = [s for s in payload.get("specialties") or [] if isinstance(s, str)]
        try:
            conf = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        # Apply confidence threshold
        threshold = settings.DOMAIN_ROUTER_CONFIDENCE_THRESHOLD
        top_k = settings.DOMAIN_ROUTER_TOP_K
        if conf >= threshold and systems:
            systems = systems[:1]
        else:
            systems = systems[:top_k]
        return DomainRoute(systems=systems, specialties=specs, confidence=conf, raw=payload)
```

- [ ] **Step 3: Add config flags**

In `src/agentrag/config.py`:

```python
    DOMAIN_FILTER_ENABLED: bool = True
    DOMAIN_ROUTER_CONFIDENCE_THRESHOLD: float = 0.7
    DOMAIN_ROUTER_TOP_K: int = 3
```

Add to `.env` / `.env.example` under LLM ROUTING section:

```env
DOMAIN_FILTER_ENABLED=true
DOMAIN_ROUTER_CONFIDENCE_THRESHOLD=0.7
DOMAIN_ROUTER_TOP_K=3
```

- [ ] **Step 4: Run test + commit**

```bash
uv run pytest tests/orchestration/test_domain_router.py -v
git add src/agentrag/orchestration/ src/agentrag/config.py .env.example tests/orchestration/
git commit -m "feat(orchestration): DomainRouter SLM classifier + config flags"
```

---

### Task 13: `FederatedRetriever` wrapper

**Files:**
- Create: `src/agentrag/retrieval/federated.py`
- Test: `tests/retrieval/test_federated.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/retrieval/test_federated.py
import pytest
from src.agentrag.retrieval.federated import FederatedRetriever


@pytest.mark.asyncio
async def test_explicit_system_override_skips_router():
    fr = FederatedRetriever()
    out = await fr.search(
        query="anything", top_k=3, system_override="tim_mach"
    )
    for hit in out.get("results", []):
        st = hit.get("system_tag")
        assert st is None or st == "tim_mach"


@pytest.mark.asyncio
async def test_no_override_uses_router_to_pick_systems():
    fr = FederatedRetriever()
    out = await fr.search(query="đau ngực", top_k=3)
    # Just ensure the call succeeds and returns the route used.
    assert "results" in out
    assert out.get("domain_route") is not None
```

- [ ] **Step 2: Implement**

```python
# src/agentrag/retrieval/federated.py
"""Wraps ElasticsearchRetriever with domain-aware filtering.

If `system_override` is provided → use it directly (UI dropdown path).
Else → consult DomainRouter and pass its picks as filter clauses.
"""
from __future__ import annotations

from typing import Any

from src.agentrag.config import settings
from src.agentrag.orchestration.domain_router import DomainRouter, DomainRoute
from src.agentrag.retrieval.elasticsearch_retriever import ElasticsearchRetriever


class FederatedRetriever:
    def __init__(
        self,
        base: ElasticsearchRetriever | None = None,
        router: DomainRouter | None = None,
    ) -> None:
        self._base = base or ElasticsearchRetriever()
        self._router = router or DomainRouter()

    async def search(
        self,
        query: str,
        *,
        document_title: str | None = None,
        system_override: str | None = None,
        specialty_override: list[str] | None = None,
        top_k: int | None = None,
        mode: str = "hybrid_kg",
        rerank: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not settings.DOMAIN_FILTER_ENABLED:
            return await self._base.search(
                query=query, document_title=document_title, top_k=top_k,
                mode=mode, rerank=rerank, **kwargs,
            )

        route: DomainRoute | None = None
        filters: dict[str, list[str]] = {}
        if system_override:
            filters["systems"] = [system_override]
        if specialty_override:
            filters["specialties"] = list(specialty_override)
        if not filters:
            route = await self._router.classify(query)
            if route.systems:
                filters["systems"] = route.systems
            if route.specialties:
                filters["specialties"] = route.specialties

        out = await self._base.search(
            query=query, document_title=document_title, top_k=top_k,
            mode=mode, rerank=rerank, filters=filters, **kwargs,
        )
        if route:
            out["domain_route"] = {
                "systems": route.systems,
                "specialties": route.specialties,
                "confidence": route.confidence,
            }
        return out
```

- [ ] **Step 3: Run test + commit**

```bash
uv run pytest tests/retrieval/test_federated.py -v
git add src/agentrag/retrieval/federated.py tests/retrieval/test_federated.py
git commit -m "feat(retrieval): FederatedRetriever with router + override"
```

---

### Task 14: Wire `FederatedRetriever` into agent + adapter chat

**Files:**
- Modify: `src/agentrag/services/knowledge_service.py` (if it instantiates retriever) OR `src/agentrag/agent/service.py`
- Modify: `src/agentrag/adapter/routers/chat.py`
- Modify: `src/agentrag/adapter/models.py`

- [ ] **Step 1: Add `domain_filter` to `ExecuteChatRequest`**

```python
# src/agentrag/adapter/models.py — find ExecuteChatRequest; add:
class ExecuteChatRequest(BaseModel):
    session_id: str
    message: str
    context: Any = None
    model_override: str | None = None
    domain_filter: dict[str, Any] | None = None   # NEW: {"system": "...", "specialties": [...]}
```

- [ ] **Step 2: Plumb filter through chat handler**

In `src/agentrag/adapter/routers/chat.py`, find `execute_chat` and forward `domain_filter` to the agent. Two options depending on architecture:

Option A — pass via `AgentService.chat(..., domain_filter=...)` kwarg.
Option B — set a contextvar `current_domain_filter` before calling.

Use Option A for clarity. Update `AgentService.chat` signature:

```python
async def chat(
    self,
    question: str,
    document_title: str | None = None,
    chat_history: list[dict[str, Any]] | None = None,
    conversation_id: str | None = None,
    domain_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
```

Inside, when the retriever is invoked, replace direct `ElasticsearchRetriever` use with `FederatedRetriever`. Pass `system_override = (domain_filter or {}).get("system")`.

- [ ] **Step 3: Same change in `GraphAgentService`**

In `src/agentrag/agent/graph_service.py`, plumb `domain_filter` through the initial state and into the `retrieve` / `tool_exec` nodes.

- [ ] **Step 4: Run an integration smoke**

```bash
make stop || true
make dev &
sleep 5
NB=$(curl -s http://localhost:8000/on/api/notebooks -H "Authorization: Bearer demo123" | jq -r '.[0].id')
SES=$(curl -s -X POST http://localhost:8000/on/api/chat/sessions -H "Authorization: Bearer demo123" \
  -H "Content-Type: application/json" -d "{\"notebook_id\":\"$NB\"}" | jq -r '.id')
curl -s -X POST http://localhost:8000/on/api/chat/execute -H "Authorization: Bearer demo123" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"$SES\",\"message\":\"đau ngực\",\"context\":\"\",\"domain_filter\":{\"system\":\"tim_mach\"}}" \
  | jq '.messages[-1].tool_trace[0].tool_input'
```

Expected: tool_input includes `filters` with `systems: ["tim_mach"]` (or equivalent — depends on how tool_input is logged).

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/adapter/models.py src/agentrag/adapter/routers/chat.py src/agentrag/agent/service.py src/agentrag/agent/graph_service.py
git commit -m "feat(adapter,agent): plumb domain_filter through chat execute"
```

---

## Phase 4 — API + UI

### Task 15: New endpoint `GET /on/api/ontology/systems` + `/specialties`

**Files:**
- Create: `src/agentrag/adapter/routers/ontology.py`
- Modify: `src/agentrag/adapter/app.py` (mount router)

- [ ] **Step 1: Write failing test**

```python
# tests/adapter/test_ontology_endpoints.py
import pytest
from httpx import AsyncClient
from src.agentrag.adapter.app import adapter


@pytest.mark.asyncio
async def test_get_systems_returns_taxonomy():
    async with AsyncClient(app=adapter, base_url="http://test") as ac:
        r = await ac.get("/api/ontology/systems")
        assert r.status_code == 200
        data = r.json()
        assert any(item["value"] == "tim_mach" for item in data)


@pytest.mark.asyncio
async def test_get_specialties_returns_taxonomy():
    async with AsyncClient(app=adapter, base_url="http://test") as ac:
        r = await ac.get("/api/ontology/specialties")
        assert r.status_code == 200
        data = r.json()
        values = {item["value"] for item in data}
        assert "noi" in values and "cap_cuu" in values
```

- [ ] **Step 2: Implement**

```python
# src/agentrag/adapter/routers/ontology.py
"""Static taxonomy lookup endpoints for UI dropdowns."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/ontology")

_SYSTEMS = [
    ("tim_mach",       "Tim mạch"),
    ("ho_hap",         "Hô hấp"),
    ("tieu_hoa",       "Tiêu hóa"),
    ("than_kinh",      "Thần kinh"),
    ("noi_tiet",       "Nội tiết"),
    ("co_xuong_khop",  "Cơ - Xương - Khớp"),
    ("huyet_hoc",      "Huyết học"),
    ("tiet_nieu",      "Tiết niệu"),
    ("sinh_duc",       "Sinh dục"),
    ("da_lieu",        "Da liễu"),
    ("mat_tmh",        "Mắt - TMH"),
    ("tam_than",       "Tâm thần"),
    ("mien_dich",      "Miễn dịch / Dị ứng"),
    ("nhi_khoa",       "Nhi"),
    ("da_he",          "Đa hệ thống"),
]

_SPECIALTIES = [
    ("noi",                 "Nội"),
    ("ngoai",               "Ngoại"),
    ("san",                 "Sản phụ khoa"),
    ("nhi",                 "Nhi"),
    ("cap_cuu",             "Cấp cứu"),
    ("hoi_suc",             "Hồi sức tích cực"),
    ("truyen_nhiem",        "Truyền nhiễm"),
    ("ung_buou",            "Ung bướu"),
    ("chan_doan_hinh_anh",  "Chẩn đoán hình ảnh"),
    ("xet_nghiem",          "Xét nghiệm"),
    ("duoc_ly",             "Dược lý"),
    ("giai_phau",           "Giải phẫu"),
    ("sinh_ly_benh",        "Sinh lý bệnh"),
    ("general",             "Chung"),
]


@router.get("/systems")
async def list_systems() -> list[dict]:
    return [{"value": v, "label": l} for v, l in _SYSTEMS]


@router.get("/specialties")
async def list_specialties() -> list[dict]:
    return [{"value": v, "label": l} for v, l in _SPECIALTIES]
```

- [ ] **Step 3: Mount in adapter app**

In `src/agentrag/adapter/app.py`, near other `app.include_router(...)`:

```python
from src.agentrag.adapter.routers.ontology import router as ontology_router
app.include_router(ontology_router, prefix="/api")
```

- [ ] **Step 4: Run test + commit**

```bash
uv run pytest tests/adapter/test_ontology_endpoints.py -v
git add src/agentrag/adapter/routers/ontology.py src/agentrag/adapter/app.py tests/adapter/test_ontology_endpoints.py
git commit -m "feat(adapter): GET /api/ontology/{systems,specialties} taxonomy endpoints"
```

---

### Task 16: Frontend `DomainFilter.tsx` component

**Files:**
- Create: `frontend/src/components/notebook/DomainFilter.tsx`
- Create: `frontend/src/lib/api/ontology.ts`

- [ ] **Step 1: API client helper**

```typescript
// frontend/src/lib/api/ontology.ts
import { apiClient } from "./client"

export interface TaxonomyItem {
  value: string
  label: string
}

export const ontologyApi = {
  systems: async (): Promise<TaxonomyItem[]> => {
    const r = await apiClient.get<TaxonomyItem[]>("/ontology/systems")
    return r.data
  },
  specialties: async (): Promise<TaxonomyItem[]> => {
    const r = await apiClient.get<TaxonomyItem[]>("/ontology/specialties")
    return r.data
  },
}
```

- [ ] **Step 2: Dropdown component**

```tsx
// frontend/src/components/notebook/DomainFilter.tsx
"use client"
import * as React from "react"
import { ontologyApi, TaxonomyItem } from "@/lib/api/ontology"

interface Props {
  value: string | null
  onChange: (v: string | null) => void
}

export function DomainFilter({ value, onChange }: Props) {
  const [systems, setSystems] = React.useState<TaxonomyItem[]>([])

  React.useEffect(() => {
    ontologyApi.systems().then(setSystems).catch(() => setSystems([]))
  }, [])

  return (
    <select
      className="bg-muted border rounded px-2 py-1 text-sm"
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value || null)}
      aria-label="Filter by medical system"
    >
      <option value="">Tất cả hệ cơ quan</option>
      {systems.map((s) => (
        <option key={s.value} value={s.value}>
          {s.label}
        </option>
      ))}
    </select>
  )
}
```

- [ ] **Step 3: Wire into `ChatPanel.tsx`**

Find the chat input row. Add `DomainFilter` above or next to it:

```tsx
import { DomainFilter } from "@/components/notebook/DomainFilter"

const [domain, setDomain] = React.useState<string | null>(null)
// ...
<DomainFilter value={domain} onChange={setDomain} />

// When calling sendMessage, attach domain_filter:
await chatApi.sendMessage({
  session_id: sessionId,
  message,
  context,
  domain_filter: domain ? { system: domain } : null,
})
```

- [ ] **Step 4: Update `chatApi.sendMessage` type to accept domain_filter**

```typescript
// frontend/src/lib/types/api.ts — extend SendNotebookChatMessageRequest
export interface SendNotebookChatMessageRequest {
  session_id: string
  message: string
  context: { sources: ...; notes: ... }
  model_override?: string
  domain_filter?: { system?: string; specialties?: string[] } | null
}
```

- [ ] **Step 5: Smoke test in browser**

Open notebook page, pick "Tim mạch" from dropdown, send a message, check the admin trace shows `filters.systems = ["tim_mach"]`.

- [ ] **Step 6: Commit**

```bash
cd /home/nguyenquocdung/work/AgentRag
git add frontend/src/components/notebook/DomainFilter.tsx frontend/src/lib/api/ontology.ts frontend/src/components/notebook/ChatPanel.tsx frontend/src/lib/types/api.ts
git commit -m "feat(ui): DomainFilter dropdown + domain_filter in chat payload"
```

---

## Phase 5 — Docs + final verification

### Task 17: Update README + ontology module README

**Files:**
- Modify: `README.md`
- Create: `src/agentrag/ontology/README.md`

- [ ] **Step 1: Add §5.10 Ontology + Domain Routing to main README**

```markdown
### 5.10 Ontology & Domain Routing (S5)

Medical KB partitioned by `system_tag` (hệ cơ quan) × `specialty_tag`
(chuyên khoa). Routing handled by `DomainRouter` (SLM) with UI override.

```env
ONTOLOGY_ENABLED=true
TAGGING_ENABLED=true                       # SectionTagger during ingest
DOMAIN_FILTER_ENABLED=true                 # FederatedRetriever active
DOMAIN_ROUTER_CONFIDENCE_THRESHOLD=0.7
DOMAIN_ROUTER_TOP_K=3
```

Seed ontology after first run:
```bash
uv run python scripts/seed_ontology.py
uv run python scripts/backfill_tags.py        # backfill existing segments
```

See [src/agentrag/ontology/README.md](src/agentrag/ontology/README.md) for the term resolver, tagging pipeline, and taxonomy reference.
```

- [ ] **Step 2: Create ontology module README**

```markdown
# Module: `ontology` — Medical Knowledge Layer

**Vị trí:** `src/agentrag/ontology/`

Canonical Vietnamese medical terms, synonyms, hierarchy, and ICD-10 mapping. Backbone of the S5 KB partitioning (system_tag × specialty_tag).

## Files

| File | Mô tả |
|---|---|
| `models.py` | SQLAlchemy `OntologyTerm` table |
| `schema.py` | Pydantic `ResolvedTerm` wire schema |
| `resolver.py` | `TermResolver` — exact, synonym, fuzzy lookup + query expansion + entity scan |

## Storage

Postgres table `ontology_terms`:
- `canonical` / `canonical_norm` (diacritic-stripped)
- `synonyms` (JSONB array)
- `system_tag` / `specialty_tags` (JSONB)
- `parent_id` (self-FK for hierarchy)
- `icd10_code` (nullable)
- `source` (custom | icd10_vn | mesh)

Indexes: B-tree on `canonical_norm`, GIN on `synonyms`, GIN trigram on `canonical_norm`.

## Resolution order

1. Exact match on `canonical_norm`
2. Synonym JSONB contains
3. Trigram fuzzy (similarity > 0.55)
4. Returns `None` if all fail.

## Seeding

- `scripts/seed_ontology.py --yaml data/ontology/custom_terms.yaml --icd10 data/ontology/icd10_vn.csv`
- Idempotent: upsert by `(canonical_norm, source)`.
- ICD-10 VN CSV: download from Bộ Y tế, expected columns `code,name_vi,name_en`.

## Tương tác

| Module | Vai trò |
|---|---|
| `ingestion.section_tagger.SectionTagger` | Calls `resolve` and `find_in_text` to tag each chunk |
| `orchestration.domain_router.DomainRouter` | (Currently independent — could query ontology for richer routing later) |
| `retrieval.federated.FederatedRetriever` | Consumes router output, not ontology directly |
| `adapter.routers.ontology` | Returns static taxonomy lists for UI dropdown |

## Config liên quan

| Key | Default | Mô tả |
|---|---|---|
| `ONTOLOGY_ENABLED` | `true` | Master switch — when `false`, `TermResolver.resolve` returns None |
```

- [ ] **Step 3: Commit**

```bash
git add README.md src/agentrag/ontology/README.md
git commit -m "docs(s5): ontology module + main README §5.10"
```

---

### Task 18: End-to-end acceptance test

**Files:**
- Create: `tests/integration/test_s5_acceptance.py`

- [ ] **Step 1: Write acceptance scenarios**

```python
# tests/integration/test_s5_acceptance.py
"""S5 acceptance tests (spec §Acceptance criteria).

Requires a running API on localhost:8000 and a notebook seeded with at
least one PDF that has clearly tim_mach + co_xuong_khop content.

Run only when explicitly requested:
    uv run pytest tests/integration/test_s5_acceptance.py -v -m integration
"""
import os
import pytest
import httpx

API = os.getenv("AGENTRAG_API", "http://localhost:8000")
TOKEN = os.getenv("AGENTRAG_TOKEN", "demo123")
NB_ID = os.getenv("AGENTRAG_NOTEBOOK_ID")


pytestmark = pytest.mark.integration


@pytest.mark.skipif(not NB_ID, reason="set AGENTRAG_NOTEBOOK_ID")
def test_chat_with_system_override_only_returns_tim_mach():
    h = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    ses = httpx.post(
        f"{API}/on/api/chat/sessions", headers=h, json={"notebook_id": NB_ID}
    ).json()["id"]
    r = httpx.post(
        f"{API}/on/api/chat/execute",
        headers=h,
        json={
            "session_id": ses,
            "message": "phân loại gãy xương dài",
            "context": "",
            "domain_filter": {"system": "co_xuong_khop"},
        },
        timeout=180,
    ).json()
    last = r["messages"][-1]
    # No leak from tim_mach docs
    for cite in last.get("citations") or []:
        # All returned chunks should be tagged co_xuong_khop (or untagged legacy)
        assert cite.get("system_tag") in (None, "co_xuong_khop"), \
            f"Leak from {cite.get('system_tag')}: {cite}"


@pytest.mark.skipif(not NB_ID, reason="set AGENTRAG_NOTEBOOK_ID")
@pytest.mark.parametrize("query,expected_system", [
    ("đau ngực kèm khó thở", "tim_mach"),
    ("gãy xương cánh tay", "co_xuong_khop"),
    ("viêm phổi ở trẻ em", "ho_hap"),
])
def test_obvious_queries_route_to_expected_system(query, expected_system):
    h = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    ses = httpx.post(
        f"{API}/on/api/chat/sessions", headers=h, json={"notebook_id": NB_ID}
    ).json()["id"]
    r = httpx.post(
        f"{API}/on/api/chat/execute",
        headers=h,
        json={"session_id": ses, "message": query, "context": ""},
        timeout=180,
    ).json()
    # Verify the tool_trace shows the filter was applied
    tt = r["messages"][-1].get("tool_trace") or []
    if tt:
        ti = tt[0].get("tool_input") or {}
        filters = ti.get("filters") or ti.get("system_tag") or {}
        # Accept either flat key or nested
        all_systems = (filters.get("systems") if isinstance(filters, dict) else None) or []
        assert expected_system in all_systems, \
            f"Expected {expected_system} in route, got {all_systems}"
```

- [ ] **Step 2: Run manually after backfill**

```bash
# In one terminal
make dev

# In another, set notebook id and run
AGENTRAG_NOTEBOOK_ID=$(curl -s http://localhost:8000/on/api/notebooks -H "Authorization: Bearer demo123" | jq -r '.[0].id') \
  uv run pytest tests/integration/test_s5_acceptance.py -v -m integration
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_s5_acceptance.py
git commit -m "test(s5): integration acceptance suite"
```

---

### Task 19: Final commit + tag

- [ ] **Step 1: Confirm green test suite**

```bash
uv run pytest tests/ontology tests/ingestion tests/retrieval tests/orchestration tests/adapter -v
```

Expected: all PASS (skipping integration unless API up).

- [ ] **Step 2: Push branch**

```bash
git push origin structmem
```

- [ ] **Step 3: Tag S5 complete**

```bash
git tag -a s5-complete -m "S5: medical KB domain partitioning shipped"
git push origin s5-complete
```

---

## Roadmap after S5 (locked by user)

| # | Sub-project | Status |
|---|---|---|
| **S5** (this) | Medical KB partitioning | ← implementing now |
| S4 | Reasoning Plane / Execution Plane layering | next |
| S1 | Token cost dashboard | after S4 |
| S2 | LangGraph trace UI per-notebook | after S1 |
| S3 | Speed / latency optimisation | last (after S1 telemetry) |
