"""add document graph_status and graph_last_error

Revision ID: b2e4f8a1c3d0
Revises: f9e648f70f63
Create Date: 2026-03-25

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b2e4f8a1c3d0"
down_revision: Union[str, Sequence[str], None] = "f9e648f70f63"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("graph_status", sa.String(length=32), nullable=True))
    op.add_column("documents", sa.Column("graph_last_error", sa.Text(), nullable=True))
    op.execute(
        sa.text(
            "UPDATE documents SET graph_status = CASE WHEN graph_synced IS TRUE THEN 'done' ELSE NULL END"
        )
    )


def downgrade() -> None:
    op.drop_column("documents", "graph_last_error")
    op.drop_column("documents", "graph_status")
