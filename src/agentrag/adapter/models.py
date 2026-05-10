"""Pydantic schemas matching open-notebook's API contract."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


# ── Notebooks ─────────────────────────────────────────────────────────────────

class NotebookCreate(BaseModel):
    name: str
    description: str = ""


class NotebookUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    archived: bool | None = None


class NotebookResponse(BaseModel):
    id: str
    name: str
    description: str
    archived: bool
    created: str
    updated: str
    source_count: int = 0
    note_count: int = 0


# ── Notes ─────────────────────────────────────────────────────────────────────

class NoteCreate(BaseModel):
    content: str | None = None
    note_type: str | None = "human"
    notebook_id: str | None = None
    title: str | None = None


class NoteUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    note_type: str | None = None


class NoteResponse(BaseModel):
    id: str
    title: str | None
    content: str | None
    note_type: str | None
    created: str
    updated: str
    command_id: str | None = None


# ── Sources ───────────────────────────────────────────────────────────────────

class SourceAsset(BaseModel):
    file_path: str | None = None
    url: str | None = None


class SourceResponse(BaseModel):
    id: str
    title: str | None
    topics: list[str] | None = None
    asset: SourceAsset | None = None
    full_text: str | None = None
    embedded: bool = False
    embedded_chunks: int = 0
    created: str
    updated: str
    command_id: str | None = None
    status: str | None = None
    notebooks: list[str] | None = None


class SourceStatusResponse(BaseModel):
    id: str
    status: str
    embedded: bool
    embedded_chunks: int


# ── Chat ──────────────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    notebook_id: str
    title: str | None = None
    model_override: str | None = None


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    model_override: str | None = None


class ChatSessionResponse(BaseModel):
    id: str
    notebook_id: str
    title: str | None
    model_override: str | None = None
    message_count: int = 0
    created: str
    updated: str


class ChatMessage(BaseModel):
    id: str | None = None              # cần cho React key prop ở frontend
    type: str                          # 'human' | 'ai' (open-notebook contract)
    role: str                          # 'user' | 'assistant' (giữ để tương thích cũ)
    content: str
    citations: list[Any] | None = None
    tool_trace: list[Any] | None = None
    timestamp: str | None = None


class ChatSessionWithMessagesResponse(ChatSessionResponse):
    messages: list[ChatMessage] = []


class ExecuteChatRequest(BaseModel):
    session_id: str
    message: str
    context: Any = None  # string (built context text) or dict — ignored, we do our own RAG
    model_override: str | None = None


class ExecuteChatResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]


class CreateSourceChatSessionRequest(BaseModel):
    source_id: str | None = None
    title: str | None = None
    model_override: str | None = None


class SendMessageRequest(BaseModel):
    message: str
    model_override: str | None = None


# ── Search ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    type: str = "text"
    limit: int = 10
    search_sources: bool = True
    search_notes: bool = True
    minimum_score: float = 0.0


class SearchResult(BaseModel):
    id: str
    title: str | None
    parent_id: str | None = None
    final_score: float
    type: str | None = None
    source_type: str | None = None
    created: str
    updated: str


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_count: int
    search_type: str


class AskRequest(BaseModel):
    question: str
    strategy_model: str | None = None
    answer_model: str | None = None
    final_answer_model: str | None = None
