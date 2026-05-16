"""Chat endpoints: map open-notebook chat sessions to AgentRag conversations."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid

from cachetools import TTLCache
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from src.agentrag.adapter.auth import is_admin
from src.agentrag.adapter.db import adapter_notebook_sources
from src.agentrag.adapter.models import (
    ChatMessage,
    ChatSessionResponse,
    ChatSessionWithMessagesResponse,
    CreateSessionRequest,
    CreateSourceChatSessionRequest,
    ExecuteChatRequest,
    ExecuteChatResponse,
    SendMessageRequest,
    UpdateSessionRequest,
)
from src.agentrag.agent.factory import get_agent_service
from src.agentrag.chat.history import ConversationStore
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Conversation, Document

router = APIRouter()

# ── Helpers ───────────────────────────────────────────────────────────────────

_SESSION_META_KEY = "open_notebook_session"


def _conv_to_session(conv: dict, message_count: int = 0) -> dict:
    meta = conv.get("extra_metadata") or {}
    return ChatSessionResponse(
        id=conv["conversation_id"],
        notebook_id=meta.get("notebook_id", ""),
        title=conv.get("title"),
        model_override=meta.get("model_override"),
        message_count=message_count,
        created=conv.get("created_at", ""),
        updated=conv.get("updated_at", ""),
    ).model_dump()


_ROLE_TO_TYPE = {"user": "human", "assistant": "ai", "human": "human", "ai": "ai"}


def _msg_to_chat(msg: dict) -> dict:
    # ID có thể từ store (message_id / id) hoặc fallback hash content+role+ts
    mid = msg.get("message_id") or msg.get("id")
    if not mid:
        import hashlib
        seed = f"{msg.get('role','')}|{msg.get('content','')[:120]}|{msg.get('created_at','')}"
        mid = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    role = msg["role"]
    extra = msg.get("extra_metadata") or {}
    return ChatMessage(
        id=mid,
        type=_ROLE_TO_TYPE.get(role, "ai"),
        role=role,
        content=msg["content"],
        citations=msg.get("citations"),
        tool_trace=msg.get("tool_trace"),
        timings_ms=msg.get("timings_ms"),
        reasoning_path=extra.get("reasoning_path"),
        plan_subqueries=extra.get("plan_subqueries"),
        sql_query=extra.get("sql_query"),
        timestamp=msg.get("created_at"),
    ).model_dump()


async def _get_document_title(source_id: str) -> str | None:
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, uuid.UUID(source_id))
        return doc.title if doc else None


# ── Notebook chat sessions ─────────────────────────────────────────────────────

notebook_router = APIRouter(prefix="/chat")


@notebook_router.get("/sessions")
async def list_sessions(notebook_id: str):
    store = ConversationStore()
    all_convs = await store.list_conversations(limit=200)
    result = []
    for c in all_convs:
        meta = (c.get("extra_metadata") or {})
        if meta.get("notebook_id") == notebook_id:
            msgs = await store.list_messages(c["conversation_id"], limit=1000)
            result.append(_conv_to_session(c, len(msgs)))
    return result


@notebook_router.post("/sessions")
async def create_session(body: CreateSessionRequest):
    store = ConversationStore()
    conv = await store.create_conversation(
        title=body.title,
        extra_metadata={
            "notebook_id": body.notebook_id,
            "model_override": body.model_override,
            _SESSION_META_KEY: True,
        },
    )
    return _conv_to_session(conv, 0)


@notebook_router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    store = ConversationStore()
    conv = await store.get_conversation(session_id)
    if not conv:
        raise HTTPException(404, "Session not found")
    msgs = await store.list_messages(session_id, limit=1000)
    session = _conv_to_session(conv, len(msgs))
    return ChatSessionWithMessagesResponse(
        **session, messages=[_msg_to_chat(m) for m in msgs]
    ).model_dump()


@notebook_router.put("/sessions/{session_id}")
async def update_session(session_id: str, body: UpdateSessionRequest):
    store = ConversationStore()
    conv = await store.get_conversation(session_id)
    if not conv:
        raise HTTPException(404, "Session not found")
    meta = conv.get("extra_metadata") or {}
    if body.model_override is not None:
        meta["model_override"] = body.model_override
    msgs = await store.list_messages(session_id, limit=1000)
    return _conv_to_session(conv, len(msgs))


@notebook_router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    store = ConversationStore()
    await store.delete_conversation(session_id)
    return {"message": "Session deleted"}


@notebook_router.post("/execute")
async def execute_chat(body: ExecuteChatRequest, request: Request):
    store = ConversationStore()
    conv = await store.get_conversation(body.session_id)
    if not conv:
        raise HTTPException(404, "Session not found")

    meta = conv.get("extra_metadata") or {}
    notebook_id = meta.get("notebook_id")

    # Find document_title from notebook sources (search all docs in notebook)
    document_title: str | None = None

    history = await store.list_messages(body.session_id, limit=20)
    await store.append_message(body.session_id, role="user", content=body.message)

    agent = get_agent_service()
    result = await agent.chat(
        question=body.message,
        document_title=document_title,
        chat_history=history,
        conversation_id=body.session_id,
        domain_filter=body.domain_filter,
    )

    await store.append_message(
        body.session_id,
        role="assistant",
        content=result.get("answer", ""),
        citations=result.get("citations", []),
        tool_trace=result.get("tool_trace", []),
        timings_ms=result.get("timings_ms", {}),
        extra_metadata={
            "reasoning_path": result.get("reasoning_path"),
            "plan_subqueries": result.get("plan_subqueries") or [],
            "sql_query": result.get("sql_query"),
        },
    )

    msgs = await store.list_messages(body.session_id, limit=1000)
    return ExecuteChatResponse(
        session_id=body.session_id,
        messages=[_msg_to_chat(m) for m in msgs],
    ).model_dump()


@notebook_router.post("/context")
async def build_context(body: dict):
    return {"context": "", "source_count": 0, "token_count": 0, "char_count": 0}


# ── Source-based chat (streaming) ─────────────────────────────────────────────

source_router = APIRouter()


@source_router.post("/sources/{source_id}/chat/sessions")
async def create_source_session(source_id: str, body: CreateSourceChatSessionRequest):
    store = ConversationStore()
    doc_title = await _get_document_title(source_id)
    conv = await store.create_conversation(
        title=body.title or doc_title,
        extra_metadata={
            "source_id": source_id,
            "document_title": doc_title,
            "model_override": body.model_override,
            _SESSION_META_KEY: True,
        },
    )
    return ChatSessionResponse(
        id=conv["conversation_id"],
        notebook_id="",
        title=conv.get("title"),
        model_override=body.model_override,
        message_count=0,
        created=conv.get("created_at", ""),
        updated=conv.get("updated_at", ""),
    ).model_dump()


@source_router.get("/sources/{source_id}/chat/sessions")
async def list_source_sessions(source_id: str):
    store = ConversationStore()
    all_convs = await store.list_conversations(limit=200)
    result = []
    for c in all_convs:
        if (c.get("extra_metadata") or {}).get("source_id") == source_id:
            msgs = await store.list_messages(c["conversation_id"], limit=1000)
            result.append(
                ChatSessionResponse(
                    id=c["conversation_id"],
                    notebook_id="",
                    title=c.get("title"),
                    message_count=len(msgs),
                    created=c.get("created_at", ""),
                    updated=c.get("updated_at", ""),
                ).model_dump()
            )
    return result


@source_router.get("/sources/{source_id}/chat/sessions/{session_id}")
async def get_source_session(source_id: str, session_id: str):
    store = ConversationStore()
    conv = await store.get_conversation(session_id)
    if not conv:
        raise HTTPException(404, "Session not found")
    msgs = await store.list_messages(session_id, limit=1000)
    return ChatSessionWithMessagesResponse(
        id=session_id,
        notebook_id="",
        title=conv.get("title"),
        message_count=len(msgs),
        created=conv.get("created_at", ""),
        updated=conv.get("updated_at", ""),
        messages=[_msg_to_chat(m) for m in msgs],
    ).model_dump()


@source_router.put("/sources/{source_id}/chat/sessions/{session_id}")
async def update_source_session(source_id: str, session_id: str, body: UpdateSessionRequest):
    store = ConversationStore()
    conv = await store.get_conversation(session_id)
    if not conv:
        raise HTTPException(404, "Session not found")
    msgs = await store.list_messages(session_id, limit=1000)
    return ChatSessionResponse(
        id=session_id, notebook_id="", title=conv.get("title"),
        message_count=len(msgs), created=conv.get("created_at", ""),
        updated=conv.get("updated_at", ""),
    ).model_dump()


@source_router.delete("/sources/{source_id}/chat/sessions/{session_id}")
async def delete_source_session(source_id: str, session_id: str):
    store = ConversationStore()
    await store.delete_conversation(session_id)
    return {"message": "Session deleted"}


@source_router.post("/sources/{source_id}/chat/sessions/{session_id}/messages")
async def send_message(
    source_id: str,
    session_id: str,
    body: SendMessageRequest,
    request: Request,
):
    """SSE streaming chat using direct RAG — strictly filtered to the source document."""
    store = ConversationStore()
    conv = await store.get_conversation(session_id)
    if not conv:
        raise HTTPException(404, "Session not found")

    meta = conv.get("extra_metadata") or {}
    document_title = meta.get("document_title")

    history = await store.list_messages(session_id, limit=10)
    await store.append_message(session_id, role="user", content=body.message)

    admin = is_admin(request)

    async def event_stream():
        yield json.dumps({"type": "user", "content": body.message}) + "\n"
        try:
            answer, citations, highlights, tool_trace = await _direct_rag(
                question=body.message,
                document_title=document_title,
                history=history,
            )
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
            return

        # Stream answer word-by-word so the UI sees streaming
        words = answer.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == len(words) - 1 else word + " "
            yield json.dumps({"type": "ai", "content": chunk}) + "\n"
            await asyncio.sleep(0)  # yield control

        await store.append_message(
            session_id,
            role="assistant",
            content=answer,
            citations=citations,
            tool_trace=tool_trace,
        )

        on_sources = [
            {
                "id": c.get("content_hash", ""),
                "title": c.get("document_title", ""),
                "excerpt": c.get("excerpt", ""),
                "page": c.get("page"),
                "segment_type": c.get("segment_type", "text"),
            }
            for c in citations
        ]
        yield json.dumps({
            "type": "context",
            "sources": on_sources,
            "insights": [],
            "notes": [],
            "highlights": highlights,
        }) + "\n"

        if admin and tool_trace:
            yield json.dumps({"type": "reasoning", "tool_trace": tool_trace}) + "\n"

        yield json.dumps({"type": "complete"}) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Cache identical (question, document) answers for 5 minutes — page reloads
# from the UI fire the same chat call repeatedly while the user is reading.
_DIRECT_RAG_CACHE: TTLCache[str, tuple] = TTLCache(maxsize=512, ttl=300)


def _direct_rag_cache_key(question: str, document_title: str | None, history: list[dict]) -> str:
    h = hashlib.sha256()
    h.update((question or "").strip().lower().encode("utf-8"))
    h.update(b"|")
    h.update((document_title or "").encode("utf-8"))
    h.update(b"|")
    # Include last 2 turns so multi-turn refinements still work
    for m in (history or [])[-2:]:
        h.update((m.get("role", "") + ":" + (m.get("content") or "")).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


async def _direct_rag(
    question: str,
    document_title: str | None,
    history: list[dict],
) -> tuple[str, list[dict], list[str], list[dict]]:
    """Retrieve filtered chunks then answer with a single LLM call.

    Uses `hybrid_kg` retrieval (so StructMem entries enrich the context) but
    strictly client-filters to `document_title` — preserves the per-source
    isolation the UI promises while still benefiting from richer signals.
    """
    from src.agentrag.config import settings
    from src.agentrag.retrieval.elasticsearch_retriever import ElasticsearchRetriever
    from src.agentrag.services.llm_gateway import LLMGateway

    cache_key = _direct_rag_cache_key(question, document_title, history)
    cached = _DIRECT_RAG_CACHE.get(cache_key)
    if cached is not None:
        return cached

    retriever = ElasticsearchRetriever()
    # Use hybrid_kg if StructMem is enabled, otherwise plain hybrid.
    retrieval_mode = "hybrid_kg" if settings.STRUCTMEM_ENABLED else "hybrid"
    result = await retriever.search(
        query=question,
        mode=retrieval_mode,
        top_k=12,  # bumped from 8 — reranker culls the long tail
        document_title=document_title,
        rerank=settings.RETRIEVAL_RERANK_ENABLED,
    )
    hits = result.get("results", [])

    if document_title:
        filtered = [h for h in hits if h.get("document_title") == document_title]
        hits = filtered if filtered else hits  # fall back if filter empties list

    # Trim to keep the LLM context tight.
    hits = hits[: settings.AGENT_MAX_CONTEXT_CHUNKS]

    context_parts = []
    for h in hits:
        section = h.get("section_path", "")
        content = h.get("content", "")
        page = h.get("page_start")
        prefix = f"[{section}" + (f" • p.{page}" if page else "") + "]" if section or page else ""
        context_parts.append(f"{prefix}\n{content}" if prefix else content)
    context_text = "\n\n---\n\n".join(context_parts)

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in (history or [])[-6:]
    )

    system_prompt = (
        "You are a helpful medical-study assistant. Answer ONLY from the provided context. "
        "Use the same language as the question (Vietnamese if question is Vietnamese). "
        "Use **bold** for key terms. If page numbers appear in section headers, "
        "cite them inline like (p.47). "
        "Return a JSON object with exactly two keys:\n"
        "  \"answer\": <full answer text in markdown>,\n"
        "  \"highlights\": [<3-5 key bullet points as strings>]\n"
        "If the context does not contain the answer, say so plainly in the answer field. "
        "ONLY return the JSON object, nothing else."
    )
    user_prompt = json.dumps({
        "question": question,
        "history": history_text,
        "context": context_text,
    }, ensure_ascii=False)

    llm = LLMGateway()
    data, _ = await llm.json_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        task="answer",
    )

    answer = data.get("answer", "")
    highlights = data.get("highlights", [])

    citations = [
        {
            "content_hash": h.get("content_hash", ""),
            "document_title": h.get("document_title", ""),
            "section_path": h.get("section_path", ""),
            "excerpt": (h.get("content") or "")[:300],
            "page": h.get("page_start"),
            "segment_type": h.get("segment_type", "text"),
        }
        for h in hits
        if h.get("content_hash")
    ]
    tool_trace = [
        {
            "tool_name": retrieval_mode + "_search",
            "tool_input": {"query": question, "document_title": document_title},
            "result_count": len(hits),
            "rerank_reason": result.get("rerank_reason"),
        }
    ]
    payload = (answer, citations, highlights, tool_trace)
    _DIRECT_RAG_CACHE[cache_key] = payload
    return payload


# ── Feedback (thumbs up/down) ─────────────────────────────────────────────────


@notebook_router.post("/feedback")
async def submit_chat_feedback(body: dict, request: Request):
    """Upsert thumbs-up/down on an assistant turn. Used later for prompt
    tuning / preference-pair datasets.

    Body: { turn_id, rating: +1|-1, session_id?, comment?, question?, answer?, reasoning_path? }
    """
    from src.agentrag.adapter.auth import get_identity
    from src.agentrag.adapter.db import AdapterChatFeedback

    turn_id = (body or {}).get("turn_id")
    rating = (body or {}).get("rating")
    if not turn_id or rating not in (1, -1, "1", "-1"):
        raise HTTPException(400, "turn_id + rating (+1 or -1) required")
    rating_int = int(rating)

    identity = get_identity(request)
    user_id = identity.user_id if identity else "anonymous"

    async with AsyncSessionLocal() as session:
        # Upsert: find existing row for (user, turn) → update; else insert.
        existing = (
            await session.execute(
                select(AdapterChatFeedback).where(
                    AdapterChatFeedback.user_id == user_id,
                    AdapterChatFeedback.turn_id == str(turn_id),
                )
            )
        ).scalar_one_or_none()
        if existing:
            existing.rating = rating_int
            if "comment" in body:
                existing.comment = body.get("comment") or None
        else:
            session.add(AdapterChatFeedback(
                user_id=user_id,
                conversation_id=body.get("session_id"),
                turn_id=str(turn_id),
                rating=rating_int,
                comment=body.get("comment") or None,
                question=body.get("question"),
                answer=body.get("answer"),
                reasoning_path=body.get("reasoning_path"),
            ))
        await session.commit()
    return {"ok": True, "rating": rating_int}


router.include_router(notebook_router)
