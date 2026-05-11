"""Adapter-specific database models (auto-created on startup)."""
from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Table, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.sql import func

from src.agentrag.database.base import Base

# Notebook ↔ Document many-to-many
adapter_notebook_sources = Table(
    "adapter_notebook_sources",
    Base.metadata,
    Column(
        "notebook_id",
        PG_UUID(as_uuid=True),
        ForeignKey("adapter_notebooks.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "document_id",
        PG_UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    extend_existing=True,
)


class AdapterNotebook(Base):
    __tablename__ = "adapter_notebooks"
    __table_args__ = {"extend_existing": True}

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    archived = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AdapterNote(Base):
    __tablename__ = "adapter_notes"
    __table_args__ = {"extend_existing": True}

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    notebook_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("adapter_notebooks.id", ondelete="SET NULL"),
        nullable=True,
    )
    title = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    note_type = Column(String(32), default="human")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AdapterTransformation(Base):
    """User-defined transformation prompts (e.g. 'summary', 'key points')."""
    __tablename__ = "adapter_transformations"
    __table_args__ = {"extend_existing": True}

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(128), nullable=False)
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    prompt = Column(Text, nullable=False)
    apply_default = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AdapterSourceInsight(Base):
    """LLM-generated insights attached to a Document (source)."""
    __tablename__ = "adapter_source_insights"
    __table_args__ = {"extend_existing": True}

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    insight_type = Column(String(64), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AdapterChatFeedback(Base):
    """User thumbs-up/down on assistant turns. Builds dataset for later
    prompt tuning / DPO. One row per (turn_id, user_id) — re-rating updates."""
    __tablename__ = "adapter_chat_feedback"
    __table_args__ = {"extend_existing": True}

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(128), nullable=False, index=True)
    conversation_id = Column(String(128), nullable=True, index=True)
    turn_id = Column(String(128), nullable=False, index=True)
    rating = Column(Integer, nullable=False)  # +1 = up, -1 = down
    comment = Column(Text, nullable=True)
    question = Column(Text, nullable=True)
    answer = Column(Text, nullable=True)
    reasoning_path = Column(String(32), nullable=True)
    extra_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


async def create_adapter_tables() -> None:
    """Create adapter tables if they don't exist. Called once at startup."""
    from src.agentrag.database import engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, checkfirst=True)
