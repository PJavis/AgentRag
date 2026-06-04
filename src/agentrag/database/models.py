# src/pam/database/models.py
from sqlalchemy import Column, Integer, String, UUID, DateTime, func, ForeignKey, Enum, JSON, Boolean, Text, Index
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from .base import Base
import uuid
from enum import Enum as PyEnum


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        Index("ix_users_email", "email", unique=True),
        Index("ix_users_google_id", "google_id"),
        {"extend_existing": True},
    )

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(320), nullable=False)
    password_hash = Column(String(255), nullable=True)
    name = Column(String(255), nullable=True)
    avatar_url = Column(String(1024), nullable=True)
    google_id = Column(String(64), nullable=True)
    is_admin = Column(Boolean, default=False, nullable=False)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Project(Base):
    __tablename__ = "projects"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    documents = relationship("Document", back_populates="project")

class Document(Base):
    __tablename__ = "documents"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(PG_UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    source_type = Column(String, nullable=False)  # markdown, google_doc, google_sheet...
    source_id = Column(String)
    title = Column(String, nullable=False)
    content_hash = Column(String(64))
    graph_synced = Column(Boolean, default=False)
    #: queued | parsing | searchable | enriching | done | done_partial | failed
    #: (NULL = bản ghi cũ trước migration). searchable+ = usable for chat.
    graph_status = Column(String(32), nullable=True)
    graph_last_error = Column(Text, nullable=True)
    graph_total_chunks = Column(Integer, nullable=True, default=0)
    graph_processed_chunks = Column(Integer, nullable=True, default=0)
    graph_failed_chunks = Column(Integer, nullable=True, default=0)
    #: parse-phase page progress (coarse; live per-page comes via SSE)
    parse_total_pages = Column(Integer, nullable=True, default=0)
    parse_done_pages = Column(Integer, nullable=True, default=0)
    modified_at = Column(DateTime(timezone=True))
    last_synced = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    project = relationship("Project", back_populates="documents")
    # cascade="all, delete-orphan" makes session.delete(document) emit explicit
    # DELETE for child Segments instead of UPDATE segments SET document_id=NULL
    # (which violates segments.document_id NOT NULL constraint).
    segments = relationship(
        "Segment",
        back_populates="document",
        cascade="all, delete-orphan",
    )

class Segment(Base):
    __tablename__ = "segments"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(PG_UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    content = Column(String, nullable=False)
    content_hash = Column(String(64))
    segment_type = Column(String)  # text, table, heading...
    section_path = Column(String)
    position = Column(Integer)
    extra_metadata = Column(JSON)
    version = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="segments")

class SyncLog(Base):
    __tablename__ = "sync_log"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(PG_UUID(as_uuid=True), ForeignKey("documents.id"))
    action = Column(String)  # ingest, update, delete...
    segments_affected = Column(Integer)
    details = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    title = Column(String, nullable=True)
    extra_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    messages = relationship("ChatMessage", back_populates="conversation")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(PG_UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    role = Column(String(16), nullable=False)  # user | assistant | system
    content = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True)
    tool_trace = Column(JSON, nullable=True)
    timings_ms = Column(JSON, nullable=True)
    extra_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    conversation = relationship("Conversation", back_populates="messages")


class EventLog(Base):
    """S6 — per-user activity feed event."""
    __tablename__ = "event_log"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    event_type = Column(String(32), nullable=False)
    target_kind = Column(String(32), nullable=True)
    target_id = Column(PG_UUID(as_uuid=True), nullable=True)
    payload = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
