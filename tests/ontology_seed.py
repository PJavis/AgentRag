"""Skip marker for tests that need ontology_terms seeded in Postgres.

Seed with:
    PYTHONPATH=. uv run python scripts/seed_ontology.py --yaml data/ontology/custom_terms.yaml
"""
from __future__ import annotations

import pytest


def _ontology_seeded() -> bool:
    try:
        from sqlalchemy import create_engine, text

        from src.agentrag.config import settings

        engine = create_engine(settings.DATABASE_URL, connect_args={"connect_timeout": 3})
        try:
            with engine.connect() as conn:
                return bool(conn.execute(
                    text("SELECT 1 FROM ontology_terms WHERE canonical = :c LIMIT 1"),
                    {"c": "Đau ngực"},
                ).first())
        finally:
            engine.dispose()
    except Exception:
        return False


requires_seeded_ontology = pytest.mark.skipif(
    not _ontology_seeded(),
    reason="ontology_terms not seeded (run scripts/seed_ontology.py)",
)
