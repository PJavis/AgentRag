"""enable pg_trgm extension for fuzzy ontology match

Revision ID: 2026051502
Revises: 2026051501
Create Date: 2026-05-15 00:01:00.000000
"""
from typing import Sequence, Union

from alembic import op

revision: str = "2026051502"
down_revision: Union[str, Sequence[str], None] = "2026051501"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_ontology_canonical_trgm "
        "ON ontology_terms USING GIN (canonical_norm gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_ontology_canonical_trgm")
    # Leave pg_trgm extension in place — other code may depend on it.
