"""add adapter_chat_feedback

Revision ID: 2026062501
Revises: 2026060501
Create Date: 2026-06-25
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

revision: str = "2026062501"
down_revision: Union[str, Sequence[str], None] = "2026060501"
branch_labels = None
depends_on = None

TABLE = "adapter_chat_feedback"


def upgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table(TABLE):
        return  # create_all(checkfirst=True) safety-net already created it
    op.create_table(
        TABLE,
        sa.Column("id", PG_UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("conversation_id", sa.String(128), nullable=True),
        sa.Column("turn_id", sa.String(128), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("reasoning_path", sa.String(32), nullable=True),
        sa.Column("extra_metadata", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_adapter_chat_feedback_user_id", TABLE, ["user_id"])
    op.create_index("ix_adapter_chat_feedback_conversation_id", TABLE, ["conversation_id"])
    op.create_index("ix_adapter_chat_feedback_turn_id", TABLE, ["turn_id"])


def downgrade() -> None:
    op.drop_index("ix_adapter_chat_feedback_turn_id", table_name=TABLE)
    op.drop_index("ix_adapter_chat_feedback_conversation_id", table_name=TABLE)
    op.drop_index("ix_adapter_chat_feedback_user_id", table_name=TABLE)
    op.drop_table(TABLE)
