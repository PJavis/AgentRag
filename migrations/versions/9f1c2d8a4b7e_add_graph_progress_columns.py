"""add graph progress columns

Revision ID: 9f1c2d8a4b7e
Revises: aa5dd4a77554
Create Date: 2026-03-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "9f1c2d8a4b7e"
down_revision: Union[str, Sequence[str], None] = "aa5dd4a77554"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("graph_total_chunks", sa.Integer(), nullable=True, server_default="0"),
    )
    op.add_column(
        "documents",
        sa.Column("graph_processed_chunks", sa.Integer(), nullable=True, server_default="0"),
    )
    op.add_column(
        "documents",
        sa.Column("graph_failed_chunks", sa.Integer(), nullable=True, server_default="0"),
    )
    op.alter_column("documents", "graph_total_chunks", server_default=None)
    op.alter_column("documents", "graph_processed_chunks", server_default=None)
    op.alter_column("documents", "graph_failed_chunks", server_default=None)


def downgrade() -> None:
    op.drop_column("documents", "graph_failed_chunks")
    op.drop_column("documents", "graph_processed_chunks")
    op.drop_column("documents", "graph_total_chunks")
