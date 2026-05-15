"""add ontology_terms table

Revision ID: 2026051501
Revises: d7e2a4b9c1f0
Create Date: 2026-05-15 00:00:00.000000

Layer 1 of S5 medical KB partitioning. See
docs/superpowers/specs/2026-05-15-medical-kb-domain-partition-design.md.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "2026051501"
down_revision: Union[str, Sequence[str], None] = "d7e2a4b9c1f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ontology_terms",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("canonical", sa.String(255), nullable=False),
        sa.Column("canonical_norm", sa.String(255), nullable=False),
        sa.Column(
            "synonyms",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("system_tag", sa.String(32), nullable=True),
        sa.Column(
            "specialty_tags",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "parent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ontology_terms.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("icd10_code", sa.String(16), nullable=True),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_ontology_canonical_norm",
        "ontology_terms",
        ["canonical_norm"],
    )
    op.create_index("ix_ontology_icd10", "ontology_terms", ["icd10_code"])
    op.execute(
        "CREATE INDEX ix_ontology_synonyms_gin "
        "ON ontology_terms USING GIN (synonyms)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_ontology_synonyms_gin")
    op.drop_index("ix_ontology_icd10", table_name="ontology_terms")
    op.drop_index("ix_ontology_canonical_norm", table_name="ontology_terms")
    op.drop_table("ontology_terms")
