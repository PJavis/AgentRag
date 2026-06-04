"""add parse page progress columns

Revision ID: 2026060501
Revises: 2026051601
Create Date: 2026-06-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "2026060501"
down_revision: Union[str, Sequence[str], None] = "2026051601"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("parse_total_pages", sa.Integer(), nullable=True, server_default="0"),
    )
    op.add_column(
        "documents",
        sa.Column("parse_done_pages", sa.Integer(), nullable=True, server_default="0"),
    )
    op.alter_column("documents", "parse_total_pages", server_default=None)
    op.alter_column("documents", "parse_done_pages", server_default=None)


def downgrade() -> None:
    op.drop_column("documents", "parse_done_pages")
    op.drop_column("documents", "parse_total_pages")
