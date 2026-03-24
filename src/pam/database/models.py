# src/pam/database/models.py
from sqlalchemy import Column, Integer, String, UUID, DateTime, func, ForeignKey, Enum, JSON, Boolean, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from .base import Base
import uuid
from enum import Enum as PyEnum

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
    source_type = Column(String, nullable=False)  # markdown, google_doc, google_sheet...
    source_id = Column(String)
    title = Column(String, nullable=False)
    content_hash = Column(String(64))
    graph_synced = Column(Boolean, default=False)
    #: pending | queued | processing | done | failed (NULL = bản ghi cũ trước migration)
    graph_status = Column(String(32), nullable=True)
    graph_last_error = Column(Text, nullable=True)
    modified_at = Column(DateTime(timezone=True))
    last_synced = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    project = relationship("Project", back_populates="documents")
    segments = relationship("Segment", back_populates="document")

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