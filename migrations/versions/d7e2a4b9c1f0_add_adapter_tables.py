"""add adapter + users tables (open-notebook compat + auth)

Revision ID: d7e2a4b9c1f0
Revises: c3f9f8b1d2e7
Create Date: 2026-05-09 00:00:00.000000

Adds tables for the open-notebook compatible adapter:
  - adapter_notebooks
  - adapter_notes
  - adapter_notebook_sources (M-N notebook ↔ document)
  - adapter_transformations
  - adapter_source_insights
  - users (auth — Phase 4)

These were previously auto-created at startup via create_adapter_tables().
This migration makes the schema explicit. Skips creation if a table already
exists so existing deployments can stamp-then-upgrade safely.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql


revision: str = "d7e2a4b9c1f0"
down_revision: Union[str, Sequence[str], None] = "c3f9f8b1d2e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(name: str) -> bool:
    bind = op.get_bind()
    return name in inspect(bind).get_table_names()


def upgrade() -> None:
    if not _table_exists("users"):
        op.create_table(
            "users",
            sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("email", sa.String(length=320), nullable=False),
            sa.Column("password_hash", sa.String(length=255), nullable=True),
            sa.Column("name", sa.String(length=255), nullable=True),
            sa.Column("avatar_url", sa.String(length=1024), nullable=True),
            sa.Column("google_id", sa.String(length=64), nullable=True),
            sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
            sa.Column("updated_at", sa.DateTime(timezone=True)),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_users_email", "users", ["email"], unique=True)
        op.create_index("ix_users_google_id", "users", ["google_id"])

    if not _table_exists("adapter_notebooks"):
        op.create_table(
            "adapter_notebooks",
            sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("archived", sa.Boolean(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _table_exists("adapter_notes"):
        op.create_table(
            "adapter_notes",
            sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("notebook_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("title", sa.Text(), nullable=True),
            sa.Column("content", sa.Text(), nullable=True),
            sa.Column("note_type", sa.String(length=32), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
            sa.PrimaryKeyConstraint("id"),
            sa.ForeignKeyConstraint(
                ["notebook_id"], ["adapter_notebooks.id"], ondelete="SET NULL"
            ),
        )

    if not _table_exists("adapter_notebook_sources"):
        op.create_table(
            "adapter_notebook_sources",
            sa.Column("notebook_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.PrimaryKeyConstraint("notebook_id", "document_id"),
            sa.ForeignKeyConstraint(
                ["notebook_id"], ["adapter_notebooks.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(
                ["document_id"], ["documents.id"], ondelete="CASCADE"
            ),
        )

    if not _table_exists("adapter_transformations"):
        op.create_table(
            "adapter_transformations",
            sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("title", sa.String(length=255), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("prompt", sa.Text(), nullable=False),
            sa.Column("apply_default", sa.Boolean(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _table_exists("adapter_source_insights"):
        op.create_table(
            "adapter_source_insights",
            sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("insight_type", sa.String(length=64), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
            sa.PrimaryKeyConstraint("id"),
            sa.ForeignKeyConstraint(
                ["source_id"], ["documents.id"], ondelete="CASCADE"
            ),
        )


def downgrade() -> None:
    op.drop_table("adapter_source_insights")
    op.drop_table("adapter_transformations")
    op.drop_table("adapter_notebook_sources")
    op.drop_table("adapter_notes")
    op.drop_table("adapter_notebooks")
    op.drop_index("ix_users_google_id", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
