"""Chat endpoints: map open-notebook chat sessions to AgentRag conversations."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid

from cachetools import TTLCache
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from src.agentrag.adapter.auth import get_identity, is_admin
from src.agentrag.observability.activity import record_event
from src.agentrag.adapter.db import adapter_notebook_sources
from src.agentrag.adapter.models import (
    ChatMessage,
    ChatSessionResponse,
    ChatSessionWithMessagesResponse,
    CreateSessionRequest,
    CreateSourceChatSessionRequest,
    ExecuteChatRequest,
    ExecuteChatResponse,
    RegenerateChatRequest,
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
        follow_ups=extra.get("follow_ups"),
        timestamp=msg.get("created_at"),
    ).model_dump()


async def _get_document_title(source_id: str) -> str | None:
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, uuid.UUID(source_id))
        return doc.title if doc else None


# Patterns the user might type to refer to a document by short name:
#   "lec10", "Lec 10", "LEC 10", "lec.10"
#   "bài 12", "Bài12"
#   "chương 3", "chuong 3", "Chuong3"
#   "file <foo>"
_FILENAME_HINT_RE = re.compile(
    # No trailing `\b` so 'Lec10' is captured as ('lec', '10').
    # Word-style kinds (chương/bài/file) still get the leading boundary.
    r"\b(?P<kind>chương|chuong|chapter|bài|bai|file|lec)\.?\s*(?P<num>\d{1,3})?",
    re.IGNORECASE | re.UNICODE,
)


async def _list_notebook_document_titles(notebook_id: str | None, limit: int = 10) -> list[str]:
    """Return Document.title list for sources linked to the given notebook."""
    if not notebook_id:
        return []
    try:
        nb_uuid = uuid.UUID(notebook_id)
    except (TypeError, ValueError):
        return []
    async with AsyncSessionLocal() as session:
        q = (
            select(Document.title)
            .join(
                adapter_notebook_sources,
                adapter_notebook_sources.c.document_id == Document.id,
            )
            .where(adapter_notebook_sources.c.notebook_id == nb_uuid)
            .limit(limit)
        )
        rows = (await session.execute(q)).all()
    return [r[0] for r in rows if r and r[0]]


async def _resolve_document_hint(question: str, notebook_id: str | None) -> str | None:
    """Scan the user message for filename hints ('lec10', 'chương 3', etc.)
    and return the best-matching Document.title from the notebook.

    Cheap ILIKE pass; no embeddings. Returns None when nothing convincing
    matches so the agent can still answer cross-document.
    """
    hits = list(_FILENAME_HINT_RE.finditer(question))
    if not hits:
        return None
    # Build LIKE patterns. Require a number for kinds that are also common
    # English/Vietnamese words ('lec', 'file'); otherwise too many false
    # positives ('lecture', 'file Canxi').
    patterns: list[str] = []
    _REQUIRES_NUM = {"lec", "file"}
    for m in hits:
        kind = (m.group("kind") or "").lower()
        num = (m.group("num") or "").strip()
        if not num and kind in _REQUIRES_NUM:
            continue
        if num:
            patterns.append(f"%{kind}%{num}%")
            patterns.append(f"%{kind}{num}%")
        else:
            patterns.append(f"%{kind}%")
    if not patterns:
        return None

    async with AsyncSessionLocal() as session:
        if notebook_id:
            try:
                nb_uuid = uuid.UUID(notebook_id)
            except (TypeError, ValueError):
                nb_uuid = None
            if nb_uuid is not None:
                # Restrict to documents linked to the current notebook.
                q = (
                    select(Document.title)
                    .join(
                        adapter_notebook_sources,
                        adapter_notebook_sources.c.document_id == Document.id,
                    )
                    .where(adapter_notebook_sources.c.notebook_id == nb_uuid)
                )
                ors = [func.lower(Document.title).like(p.lower()) for p in patterns]
                from sqlalchemy import or_
                q = q.where(or_(*ors)).limit(5)
                rows = (await session.execute(q)).all()
                if rows:
                    return rows[0][0]
        # Fallback: search globally.
        from sqlalchemy import or_
        q2 = select(Document.title).where(
            or_(*[func.lower(Document.title).like(p.lower()) for p in patterns])
        ).limit(5)
        rows = (await session.execute(q2)).all()
        return rows[0][0] if rows else None


def _summary_to_markdown(summary: dict) -> str:
    """Render a SummaryService.generate() result into a Markdown chat answer."""
    title = summary.get("title") or "Tóm tắt"
    parts: list[str] = [f"## Tóm tắt: {title}\n"]
    overview = summary.get("overview")
    if isinstance(overview, str) and overview.strip():
        parts.append(overview.strip() + "\n")
    sections = summary.get("sections") or []
    for sec in sections:
        heading = sec.get("heading") or ""
        if heading:
            parts.append(f"\n### {heading}\n")
        body_text = sec.get("summary")
        if isinstance(body_text, str) and body_text.strip():
            parts.append(body_text.strip())
        key_points = sec.get("key_points") or []
        if isinstance(key_points, list) and key_points:
            parts.append("")
            for kp in key_points:
                if isinstance(kp, str) and kp.strip():
                    clean = kp.strip().lstrip("-•* ").strip()
                    parts.append(f"- {clean}")
        # Note: important_terms intentionally NOT rendered as a separate block.
        # Terms are highlighted inline via **bold** within summary/key_points.
    return "\n".join(parts).strip() or "Không trích xuất được tóm tắt từ tài liệu."


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
async def create_session(body: CreateSessionRequest, request: Request):
    store = ConversationStore()
    identity = get_identity(request)
    conv = await store.create_conversation(
        title=body.title,
        user_id=identity.user_id if identity else None,
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
    import logging
    _log = logging.getLogger(__name__)
    store = ConversationStore()
    conv = await store.get_conversation(body.session_id)
    if not conv:
        raise HTTPException(404, "Session not found")

    # Find document_title from notebook sources (search all docs in notebook).
    # Try to resolve filename hints in the user message ('lec10', 'chương 3'…)
    # to a concrete title from the linked notebook docs.
    notebook_id = (conv.get("extra_metadata") or {}).get("notebook_id")
    document_title: str | None = None
    try:
        document_title = await _resolve_document_hint(body.message, notebook_id)
        if document_title:
            _log.info("execute_chat: resolved document hint → %r", document_title)
    except Exception:
        _log.exception("execute_chat: _resolve_document_hint failed")

    # Summary intent: do NOT auto-pin here — the SummaryService routing block
    # below handles single-doc (when filename hint resolved) vs multi-doc
    # (iterate all notebook sources) on its own. For non-summary chat we want
    # cross-document retrieval, so leave document_title=None unless explicitly
    # hinted.

    history: list[dict] = []
    user_appended_ok = False
    try:
        history = await store.list_messages(body.session_id, limit=20)
    except Exception:
        _log.exception("execute_chat: list_messages(history) failed")

    try:
        await store.append_message(body.session_id, role="user", content=body.message)
        user_appended_ok = True
    except Exception:
        _log.exception("execute_chat: append_message(user) failed")

    # S5: auto domain detection when caller didn't pick a domain manually.
    # Reasoning plane drives routing — call DomainRouter then forward picks
    # to the retriever via domain_filter. UI override still wins.
    effective_domain_filter = body.domain_filter
    if not effective_domain_filter:
        from src.agentrag.config import settings as _settings
        if _settings.DOMAIN_FILTER_ENABLED:
            try:
                from src.agentrag.services.container import get_container
                route = await get_container().domain_router.classify(body.message)
                if route.systems or route.specialties:
                    effective_domain_filter = {
                        **({"system": route.systems[0]} if route.systems else {}),
                        **({"specialties": route.specialties} if route.specialties else {}),
                    }
            except Exception:
                _log.exception("execute_chat: domain_router.classify failed")
                effective_domain_filter = None

    # Summary intent + resolved document → bypass agent loop, use the dedicated
    # SummaryService which iterates a structured medical template per section.
    # The 7B agent JSON-output mode is unreliable for full-doc summaries; the
    # dedicated service produces deterministic markdown output.
    # Multi-doc: if notebook has 2-5 sources, summarize all and merge.
    summary_routed = False
    result: dict
    try:
        from src.agentrag.agent.service import _is_verbose_followup
        if _is_verbose_followup(body.message):
            from src.agentrag.generation.summary_service import SummaryService
            # Collect doc titles to summarize. If filename hint matched →
            # single-doc; else all linked sources (capped to 5).
            doc_titles: list[str] = []
            if document_title:
                doc_titles = [document_title]
            else:
                doc_titles = await _list_notebook_document_titles(notebook_id, limit=5)
            if doc_titles:
                _log.info(
                    "execute_chat: SummaryService multi-doc → %d titles: %s",
                    len(doc_titles), [t[:40] for t in doc_titles],
                )
                svc = SummaryService()
                if len(doc_titles) == 1:
                    summary = await svc.generate(document_title=doc_titles[0], style="study_note")
                    markdown = _summary_to_markdown(summary)
                else:
                    # Run all summaries in parallel; merge into one Markdown answer.
                    summaries = await asyncio.gather(
                        *[svc.generate(document_title=t, style="study_note") for t in doc_titles],
                        return_exceptions=True,
                    )
                    parts: list[str] = [f"## Tóm tắt notebook ({len(doc_titles)} tài liệu)\n"]
                    for i, s in enumerate(summaries, 1):
                        if isinstance(s, Exception):
                            parts.append(f"\n### {i}. {doc_titles[i-1]}\n\n⚠️ Lỗi: {s}\n")
                            continue
                        parts.append(f"\n---\n\n### {i}. {doc_titles[i-1]}\n")
                        md_one = _summary_to_markdown(s)
                        # Strip duplicate top-level "## Tóm tắt:" heading
                        if md_one.startswith("## "):
                            nl = md_one.find("\n")
                            if nl > 0:
                                md_one = md_one[nl+1:].lstrip()
                        parts.append(md_one)
                    markdown = "\n".join(parts).strip()
                result = {
                    "answer": markdown,
                    "citations": [],
                    "tool_trace": [],
                    "timings_ms": {},
                    "reasoning_path": "summary",
                    "plan_subqueries": [],
                    "sql_query": None,
                    "highlights": [],
                }
                summary_routed = True
    except Exception:
        _log.exception("execute_chat: SummaryService routing failed, falling back to agent")

    if not summary_routed:
        agent = get_agent_service()
        try:
            result = await agent.chat(
                question=body.message,
                document_title=document_title,
                chat_history=history,
                conversation_id=body.session_id,
                domain_filter=effective_domain_filter,
                verbosity=body.verbosity,
            )
        except Exception as exc:
            _log.exception("execute_chat: agent.chat failed")
            result = {
                "answer": f"⚠️ Agent failed: {type(exc).__name__}: {str(exc)[:300]}",
                "citations": [],
                "tool_trace": [],
                "timings_ms": {},
                "reasoning_path": "error",
                "plan_subqueries": [],
                "sql_query": None,
            }

    appended: dict | None = None
    try:
        appended = await store.append_message(
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
    except Exception:
        _log.exception("execute_chat: append_message(assistant) failed")

    # B2 — Follow-up suggestions. Best-effort: log + ignore on failure.
    follow_ups: list[str] = []
    try:
        from src.agentrag.agent.followups import generate_followups
        from src.agentrag.services.container import get_container
        follow_ups = await generate_followups(
            question=body.message,
            answer=result.get("answer", ""),
            citations=result.get("citations") or [],
            llm_gateway=get_container().llm,
        )
    except Exception:
        _log.exception("execute_chat: generate_followups failed")
    # Persist on the assistant turn's extra_metadata
    if follow_ups and isinstance(appended, dict):
        try:
            from src.agentrag.database import AsyncSessionLocal
            from src.agentrag.database.models import ChatMessage as _CM
            msg_id = appended.get("message_id")
            if msg_id:
                async with AsyncSessionLocal() as _s:
                    row = await _s.get(_CM, uuid.UUID(msg_id))
                    if row is not None:
                        meta = dict(row.extra_metadata or {})
                        meta["follow_ups"] = follow_ups
                        row.extra_metadata = meta
                        await _s.commit()
                # Invalidate the messages cache so the very next list_messages
                # picks up the updated extra_metadata immediately.
                try:
                    await store._delete_messages_cache(body.session_id)
                except Exception:
                    pass
        except Exception:
            _log.exception("execute_chat: persist follow_ups failed")

    # S6 — record chat_turn event for the activity feed (best-effort)
    try:
        identity = get_identity(request)
        timings = result.get("timings_ms") or {}
        await record_event(
            user_id=identity.user_id if identity else None,
            event_type="chat_turn",
            target_kind="conversation",
            target_id=body.session_id,
            payload={
                "message": body.message[:200],
                "message_id": appended.get("message_id") if isinstance(appended, dict) else None,
                "tokens_in": timings.get("tokens_in"),
                "tokens_out": timings.get("tokens_out"),
                "latency_ms": timings.get("total"),
                "reasoning_path": result.get("reasoning_path"),
            },
        )
    except Exception:
        _log.exception("execute_chat: record_event failed")

    # Final response — try fresh list_messages, fall back to a hand-rolled
    # message pair if DB read fails so the UI still shows both turns.
    try:
        msgs = await store.list_messages(body.session_id, limit=1000)
        return ExecuteChatResponse(
            session_id=body.session_id,
            messages=[_msg_to_chat(m) for m in msgs],
        ).model_dump()
    except Exception:
        _log.exception("execute_chat: final list_messages failed")
        fallback: list[dict] = []
        if user_appended_ok or True:
            fallback.append(_msg_to_chat({
                "role": "user", "content": body.message,
                "message_id": f"local-user-{body.session_id}",
            }))
        fallback.append(_msg_to_chat({
            "role": "assistant", "content": result.get("answer", ""),
            "citations": result.get("citations", []),
            "tool_trace": result.get("tool_trace", []),
            "timings_ms": result.get("timings_ms", {}),
            "extra_metadata": {
                "reasoning_path": result.get("reasoning_path"),
                "plan_subqueries": result.get("plan_subqueries") or [],
                "sql_query": result.get("sql_query"),
            },
            "message_id": f"local-ai-{body.session_id}",
        }))
        return ExecuteChatResponse(
            session_id=body.session_id,
            messages=fallback,
        ).model_dump()


@notebook_router.post("/regenerate")
async def regenerate_chat(body: RegenerateChatRequest, request: Request):
    """Re-run the agent for the user message that produced the given assistant
    turn. Deletes the stale assistant message, runs `agent.chat()` again with
    the same question + truncated history, appends the fresh assistant turn,
    returns full message list."""
    import logging
    _log = logging.getLogger(__name__)
    store = ConversationStore()
    conv = await store.get_conversation(body.session_id)
    if not conv:
        raise HTTPException(404, "Session not found")

    msgs = await store.list_messages(body.session_id, limit=1000)
    target_idx = next(
        (i for i, m in enumerate(msgs) if m.get("message_id") == body.assistant_message_id),
        -1,
    )
    if target_idx < 0:
        raise HTTPException(404, "Assistant message not found in session")
    if msgs[target_idx].get("role") != "assistant":
        raise HTTPException(400, "Target message is not an assistant turn")

    user_idx = target_idx - 1
    while user_idx >= 0 and msgs[user_idx].get("role") != "user":
        user_idx -= 1
    if user_idx < 0:
        raise HTTPException(400, "No preceding user message to regenerate from")

    user_question = msgs[user_idx].get("content", "")
    history = msgs[:user_idx + 1]  # keep prior turns + user msg, drop everything after

    # Drop the stale assistant turn + ALL subsequent turns (user + assistant).
    # Regenerating from an old turn invalidates the downstream conversation
    # branch — keeping orphaned follow-ups produces broken history on the next
    # turn (agent thinks user already asked X when they did not).
    for stale in msgs[target_idx:]:
        try:
            await store.delete_message(body.session_id, stale["message_id"])
        except Exception:
            _log.exception("regenerate: delete_message failed for %s", stale.get("message_id"))

    # Resolve document hint + domain filter (same as execute_chat)
    notebook_id = (conv.get("extra_metadata") or {}).get("notebook_id")
    document_title: str | None = None
    try:
        document_title = await _resolve_document_hint(user_question, notebook_id)
    except Exception:
        _log.exception("regenerate: _resolve_document_hint failed")

    # (Summary-intent multi-doc routing is handled in the SummaryService block below.)

    effective_domain_filter = body.domain_filter
    if not effective_domain_filter:
        from src.agentrag.config import settings as _settings
        if _settings.DOMAIN_FILTER_ENABLED:
            try:
                from src.agentrag.services.container import get_container
                route = await get_container().domain_router.classify(user_question)
                if route.systems or route.specialties:
                    effective_domain_filter = {
                        **({"system": route.systems[0]} if route.systems else {}),
                        **({"specialties": route.specialties} if route.specialties else {}),
                    }
            except Exception:
                _log.exception("regenerate: domain_router.classify failed")

    # Same summary-intent routing as execute_chat — single-doc OR multi-doc.
    summary_routed = False
    result: dict
    try:
        from src.agentrag.agent.service import _is_verbose_followup
        if _is_verbose_followup(user_question):
            from src.agentrag.generation.summary_service import SummaryService
            doc_titles = [document_title] if document_title else await _list_notebook_document_titles(notebook_id, limit=5)
            if doc_titles:
                _log.info(
                    "regenerate: SummaryService multi-doc → %d titles",
                    len(doc_titles),
                )
                svc = SummaryService()
                if len(doc_titles) == 1:
                    summary = await svc.generate(document_title=doc_titles[0], style="study_note")
                    markdown = _summary_to_markdown(summary)
                else:
                    summaries = await asyncio.gather(
                        *[svc.generate(document_title=t, style="study_note") for t in doc_titles],
                        return_exceptions=True,
                    )
                    parts: list[str] = [f"## Tóm tắt notebook ({len(doc_titles)} tài liệu)\n"]
                    for i, s in enumerate(summaries, 1):
                        if isinstance(s, Exception):
                            parts.append(f"\n### {i}. {doc_titles[i-1]}\n\n⚠️ Lỗi: {s}\n")
                            continue
                        parts.append(f"\n---\n\n### {i}. {doc_titles[i-1]}\n")
                        md_one = _summary_to_markdown(s)
                        if md_one.startswith("## "):
                            nl = md_one.find("\n")
                            if nl > 0:
                                md_one = md_one[nl+1:].lstrip()
                        parts.append(md_one)
                    markdown = "\n".join(parts).strip()
                result = {
                    "answer": markdown,
                    "citations": [],
                    "tool_trace": [],
                    "timings_ms": {},
                    "reasoning_path": "summary",
                    "plan_subqueries": [],
                    "sql_query": None,
                    "highlights": [],
                }
                summary_routed = True
    except Exception:
        _log.exception("regenerate: SummaryService routing failed, falling back to agent")

    if not summary_routed:
        agent = get_agent_service()
        try:
            result = await agent.chat(
                question=user_question,
                document_title=document_title,
                chat_history=history,
                conversation_id=body.session_id,
                domain_filter=effective_domain_filter,
                verbosity=body.verbosity,
            )
        except Exception as exc:
            _log.exception("regenerate: agent.chat failed")
            result = {
                "answer": f"⚠️ Agent failed: {type(exc).__name__}: {str(exc)[:300]}",
                "citations": [],
                "tool_trace": [],
                "timings_ms": {},
                "reasoning_path": "error",
                "plan_subqueries": [],
                "sql_query": None,
            }

    appended: dict | None = None
    try:
        appended = await store.append_message(
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
                "regenerated": True,
            },
        )
    except Exception:
        _log.exception("regenerate: append_message failed")

    # Follow-ups (best-effort, mirrors execute_chat)
    follow_ups: list[str] = []
    try:
        from src.agentrag.agent.followups import generate_followups
        from src.agentrag.services.container import get_container
        follow_ups = await generate_followups(
            question=user_question,
            answer=result.get("answer", ""),
            citations=result.get("citations") or [],
            llm_gateway=get_container().llm,
        )
    except Exception:
        _log.exception("regenerate: generate_followups failed")
    if follow_ups and isinstance(appended, dict):
        try:
            from src.agentrag.database import AsyncSessionLocal as _ASL
            from src.agentrag.database.models import ChatMessage as _CM
            msg_id = appended.get("message_id")
            if msg_id:
                async with _ASL() as _s:
                    row = await _s.get(_CM, uuid.UUID(msg_id))
                    if row is not None:
                        meta = dict(row.extra_metadata or {})
                        meta["follow_ups"] = follow_ups
                        row.extra_metadata = meta
                        await _s.commit()
                try:
                    await store._delete_messages_cache(body.session_id)
                except Exception:
                    pass
        except Exception:
            _log.exception("regenerate: persist follow_ups failed")

    try:
        msgs = await store.list_messages(body.session_id, limit=1000)
        return ExecuteChatResponse(
            session_id=body.session_id,
            messages=[_msg_to_chat(m) for m in msgs],
        ).model_dump()
    except Exception:
        _log.exception("regenerate: final list_messages failed")
        raise HTTPException(500, "Regenerate succeeded but final read failed")


@notebook_router.post("/context")
async def build_context(body: dict):
    return {"context": "", "source_count": 0, "token_count": 0, "char_count": 0}


@notebook_router.post("/execute-stream")
async def execute_chat_stream(body: ExecuteChatRequest, request: Request):
    """SSE variant of /chat/execute. Emits `event: token` chunks for the
    semantic answer path; falls back to a single `event: done` for the summary
    path (which is already fast and produces one structured block)."""
    import logging
    import json as _json
    _log = logging.getLogger(__name__)

    store = ConversationStore()
    conv = await store.get_conversation(body.session_id)
    if not conv:
        raise HTTPException(404, "Session not found")

    notebook_id = (conv.get("extra_metadata") or {}).get("notebook_id")
    document_title: str | None = None
    try:
        document_title = await _resolve_document_hint(body.message, notebook_id)
    except Exception:
        _log.exception("execute_chat_stream: _resolve_document_hint failed")

    history: list[dict] = []
    try:
        history = await store.list_messages(body.session_id, limit=20)
    except Exception:
        pass
    try:
        await store.append_message(body.session_id, role="user", content=body.message)
    except Exception:
        pass

    effective_domain_filter = body.domain_filter
    if not effective_domain_filter:
        from src.agentrag.config import settings as _settings
        if _settings.DOMAIN_FILTER_ENABLED:
            try:
                from src.agentrag.services.container import get_container
                route = await get_container().domain_router.classify(body.message)
                if route.systems or route.specialties:
                    effective_domain_filter = {
                        **({"system": route.systems[0]} if route.systems else {}),
                        **({"specialties": route.specialties} if route.specialties else {}),
                    }
            except Exception:
                _log.exception("execute_chat_stream: domain_router.classify failed")

    # Summary intent → bypass agent loop. Emit a single block via SSE for UX
    # consistency (frontend just appends and renders).
    from src.agentrag.agent.service import _is_verbose_followup
    is_summary = _is_verbose_followup(body.message)
    summary_titles: list[str] = []
    if is_summary:
        summary_titles = [document_title] if document_title else await _list_notebook_document_titles(notebook_id, limit=5)

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"

    async def event_generator():
        try:
            if is_summary and summary_titles:
                yield _sse("status", {"step": "summary", "doc_count": len(summary_titles)})
                from src.agentrag.generation.summary_service import (
                    SummaryService,
                    _MEDICAL_TEMPLATE_VI,
                )
                svc = SummaryService()
                acc_parts: list[str] = []

                async def stream_doc(doc_title: str, doc_idx: int | None):
                    """Yield ('token'|'status', payload) per section so the UI
                    sees incremental progress instead of a long blank wait."""
                    if doc_idx is None:
                        prefix = f"## {doc_title}\n\n"
                    else:
                        prefix = f"## {doc_idx}. {doc_title}\n\n"
                    acc_parts.append(prefix)
                    yield ("token", prefix)

                    overview = await svc._generate_overview(doc_title)
                    if overview:
                        chunk = overview.strip() + "\n\n"
                        acc_parts.append(chunk)
                        yield ("token", chunk)

                    image_chunks = await svc._fetch_image_chunks(doc_title)
                    for heading in _MEDICAL_TEMPLATE_VI:
                        yield ("status", {"step": "section", "heading": heading, "doc": doc_title})
                        sec = await svc._summarize_section(doc_title, heading, image_chunks)
                        if not (sec.get("summary") or sec.get("key_points")):
                            continue
                        lines = [f"### {sec.get('heading') or heading}\n"]
                        sec_sum = (sec.get("summary") or "").strip()
                        if sec_sum:
                            lines.append(sec_sum + "\n")
                        for kp in (sec.get("key_points") or []):
                            if isinstance(kp, str) and kp.strip():
                                clean = kp.strip().lstrip("-•* ").strip()
                                lines.append(f"- {clean}")
                        # important_terms intentionally not rendered as a list
                        # — key terms are already inline-bolded via **bold**.
                        lines.append("\n")
                        block = "\n".join(lines)
                        acc_parts.append(block)
                        yield ("token", block)

                if len(summary_titles) == 1:
                    async for kind, payload in stream_doc(summary_titles[0], None):
                        if kind == "token":
                            yield _sse("token", {"text": payload})
                        else:
                            yield _sse("status", payload)
                else:
                    header = f"## Tóm tắt notebook ({len(summary_titles)} tài liệu)\n\n"
                    acc_parts.append(header)
                    yield _sse("token", {"text": header})
                    for idx, dt in enumerate(summary_titles, 1):
                        sep = "\n---\n\n"
                        acc_parts.append(sep)
                        yield _sse("token", {"text": sep})
                        async for kind, payload in stream_doc(dt, idx):
                            if kind == "token":
                                yield _sse("token", {"text": payload})
                            else:
                                yield _sse("status", payload)

                answer = "".join(acc_parts).strip()
                try:
                    await store.append_message(
                        body.session_id, role="assistant", content=answer,
                        extra_metadata={"reasoning_path": "summary"},
                    )
                except Exception:
                    _log.exception("execute_chat_stream: append summary failed")
                yield _sse("done", {"reasoning_path": "summary", "answer_length": len(answer)})
                return

            # Semantic path — stream via agent.chat_stream.
            from src.agentrag.retrieval.context import set_domain_filter
            set_domain_filter(effective_domain_filter)
            agent = get_agent_service()
            # The picker sends a `prefix::provider::model` id — take the bare
            # model name (e.g. deepseek-v4-pro) so AgentLLM prefix-routing applies.
            _ov = body.model_override
            _ov = _ov.split("::")[-1] if _ov and "::" in _ov else _ov
            collected: list[str] = []
            async for chunk in agent.chat_stream(
                question=body.message,
                document_title=document_title,
                chat_history=history,
                conversation_id=body.session_id,
                model_override=_ov,
                verbosity=body.verbosity,
            ):
                if "event: token" in chunk:
                    try:
                        data_line = chunk.split("data: ", 1)[-1].strip()
                        token = _json.loads(data_line).get("text", "")
                        if token:
                            collected.append(token)
                    except Exception:
                        pass
                yield chunk
            full_answer = "".join(collected)
            if full_answer:
                try:
                    await store.append_message(
                        body.session_id, role="assistant", content=full_answer,
                        extra_metadata={"reasoning_path": "semantic"},
                    )
                except Exception:
                    _log.exception("execute_chat_stream: append assistant failed")
        except Exception as exc:
            _log.exception("execute_chat_stream: error")
            yield _sse("error", {"detail": f"{type(exc).__name__}: {str(exc)[:200]}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Source-based chat (streaming) ─────────────────────────────────────────────

source_router = APIRouter()


@source_router.post("/sources/{source_id}/chat/sessions")
async def create_source_session(source_id: str, body: CreateSourceChatSessionRequest, request: Request):
    store = ConversationStore()
    doc_title = await _get_document_title(source_id)
    identity = get_identity(request)
    conv = await store.create_conversation(
        title=body.title or doc_title,
        user_id=identity.user_id if identity else None,
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

        appended = await store.append_message(
            session_id,
            role="assistant",
            content=answer,
            citations=citations,
            tool_trace=tool_trace,
        )
        # S6 — activity event for source chat turn
        identity_src = get_identity(request)
        await record_event(
            user_id=identity_src.user_id if identity_src else None,
            event_type="chat_turn",
            target_kind="conversation",
            target_id=session_id,
            payload={
                "message": body.message[:200],
                "message_id": appended.get("message_id") if isinstance(appended, dict) else None,
                "source_id": source_id,
            },
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


# ── Chat with attached image (ad-hoc vision Q&A) ─────────────────────────────


@notebook_router.post("/with-image")
async def chat_with_image(
    request: Request,
    session_id: str = Form(...),
    message: str = Form(...),
    image: UploadFile = File(...),
):
    """Ad-hoc multimodal chat — user drops an image into the chat and asks
    a question about it. Image is saved under data/images/chat-uploads/,
    then VisionService describes it AND the answer LLM receives the actual
    bytes (via multimodal). Result persists to the conversation."""
    import logging, os as _os, time as _time, uuid as _uuid
    from pathlib import Path as _Path
    _log = logging.getLogger(__name__)

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "image content_type must be image/*")

    store = ConversationStore()
    conv = await store.get_conversation(session_id)
    if not conv:
        raise HTTPException(404, "Session not found")

    raw = await image.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "image too large (max 20MB)")
    ext = (image.filename or "uploaded.jpg").rsplit(".", 1)[-1].lower()
    if ext not in ("jpg", "jpeg", "png", "webp", "gif"):
        ext = "jpg"
    fname = f"{_uuid.uuid4().hex}.{ext}"
    upload_dir = _Path(settings.IMAGE_STORAGE_DIR) / "chat-uploads" / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / fname
    save_path.write_bytes(raw)
    rel_url = f"/images/chat-uploads/{session_id}/{fname}"
    public_url = (
        settings.PUBLIC_BASE_URL.rstrip("/") + rel_url
        if getattr(settings, "PUBLIC_BASE_URL", None)
        else rel_url
    )

    # Persist user turn with image marker so UI can render thumbnail.
    user_content = message.strip() or "[Hỏi về hình đã đính kèm]"
    try:
        await store.append_message(
            session_id, role="user", content=user_content,
            extra_metadata={"attached_image_url": rel_url, "attached_image_mime": image.content_type},
        )
    except Exception:
        _log.exception("with-image: append user msg failed")

    # Call vision-capable LLM directly (no retrieval — ad-hoc Q&A on image).
    answer_text: str
    try:
        from src.agentrag.services.container import get_container
        gateway = get_container().llm
        system_prompt = (
            "Bạn là trợ lý y khoa. Trả lời câu hỏi của người dùng dựa vào hình "
            "ảnh được đính kèm. Sử dụng **bold** cho thuật ngữ y khoa, "
            "$...$ cho công thức nếu cần."
        )
        desc, _lat = await gateway.vision_response(
            system_prompt=system_prompt,
            text_prompt=user_content,
            image_bytes=raw,
            mime_type=image.content_type,
            task="answer",
        )
        answer_text = (desc or "").strip()
        if not answer_text:
            answer_text = "Không tạo được mô tả từ hình ảnh."
    except Exception as exc:
        _log.exception("with-image: vision call failed")
        answer_text = f"⚠️ Lỗi xử lý ảnh: {type(exc).__name__}"

    try:
        await store.append_message(
            session_id, role="assistant", content=answer_text,
            extra_metadata={
                "reasoning_path": "vision_chat",
                "attached_image_url": rel_url,
            },
        )
    except Exception:
        _log.exception("with-image: append assistant msg failed")

    try:
        from src.agentrag.observability.activity import record_event
        identity = get_identity(request)
        await record_event(
            user_id=identity.user_id if identity else None,
            event_type="chat_turn",
            target_kind="conversation",
            target_id=session_id,
            payload={"message": user_content[:200], "has_image": True},
        )
    except Exception:
        pass

    msgs = await store.list_messages(session_id, limit=1000)
    return ExecuteChatResponse(
        session_id=session_id,
        messages=[_msg_to_chat(m) for m in msgs],
    ).model_dump()


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
    # Emit activity event so feedback shows up in user's activity feed.
    try:
        from src.agentrag.observability.activity import record_event
        await record_event(
            user_id=user_id if user_id != "anonymous" else None,
            event_type="chat_feedback",
            target_kind="chat_turn",
            target_id=str(turn_id),
            payload={
                "rating": rating_int,
                "session_id": body.get("session_id"),
                "reasoning_path": body.get("reasoning_path"),
            },
        )
    except Exception:
        pass
    return {"ok": True, "rating": rating_int}


router.include_router(notebook_router)
