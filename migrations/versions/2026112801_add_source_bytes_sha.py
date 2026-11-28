"""add documents.source_bytes_sha

Revision ID: 2026112801
Revises: 2026062501
Create Date: 2026-11-28 00:00:00.000000

Duplicate detection keys on the file's bytes. content_hash mixes in the parse
settings, so the same file ingested under different settings never matched
(122 documents held 35 distinct files). Existing rows stay NULL until
`python -m src.agentrag.ingestion.dedupe_corpus --backfill` fills them.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "2026112801"
down_revision: Union[str, Sequence[str], None] = "2026062501"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("source_bytes_sha", sa.String(length=64), nullable=True))
    op.create_index("ix_documents_source_bytes_sha", "documents", ["source_bytes_sha"])


def downgrade() -> None:
    op.drop_index("ix_documents_source_bytes_sha", table_name="documents")
    op.drop_column("documents", "source_bytes_sha")
