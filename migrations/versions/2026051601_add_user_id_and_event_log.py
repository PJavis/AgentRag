"""S6 — user_id FK on conversations + documents, plus event_log table.

Revision ID: 2026051601
Revises: 2026051502
Create Date: 2026-05-16
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "2026051601"
down_revision: Union[str, Sequence[str], None] = "2026051502"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
    )
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])

    op.add_column(
        "documents",
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
    )
    op.create_index("ix_documents_user_id", "documents", ["user_id"])

    op.create_table(
        "event_log",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("target_kind", sa.String(32), nullable=True),
        sa.Column("target_id", UUID(as_uuid=True), nullable=True),
        sa.Column("payload", sa.JSON, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_event_log_user_created",
        "event_log",
        ["user_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_event_log_type_created",
        "event_log",
        ["event_type", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_event_log_target_id",
        "event_log",
        ["target_id"],
        postgresql_where=sa.text("target_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_event_log_target_id", table_name="event_log")
    op.drop_index("ix_event_log_type_created", table_name="event_log")
    op.drop_index("ix_event_log_user_created", table_name="event_log")
    op.drop_table("event_log")
    op.drop_index("ix_documents_user_id", table_name="documents")
    op.drop_column("documents", "user_id")
    op.drop_index("ix_conversations_user_id", table_name="conversations")
    op.drop_column("conversations", "user_id")
