"""SQLAlchemy model for the ontology_terms table.

Layer 1 of the S5 medical KB partitioning: canonical Vietnamese medical
terms, synonyms, hierarchy, and ICD-10 mapping. See
docs/superpowers/specs/2026-05-15-medical-kb-domain-partition-design.md
"""
from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
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
    source = Column(String(16), nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
