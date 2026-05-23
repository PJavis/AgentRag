"""S6 — Activity panel routes.

Two surfaces:
  - /api/activity/*           — personal (requires login; SQL scoped to identity.user_id)
  - /api/admin/activity/*     — admin global (is_admin OR X-Admin-Token)

Decision logic lives nowhere here — pure SQL aggregation + serialization
over the event_log table populated by observability/activity.record_event().
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import desc, func, select

from src.agentrag.adapter.auth import get_identity, is_admin
from src.agentrag.adapter.models import (
    ActivityCounts,
    ActivityEvent,
    ActivityHeatmapCell,
    ActivitySummary,
    AdminUserEntry,
)
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document, EventLog, User

router = APIRouter(prefix="/activity")
admin_router = APIRouter(prefix="/admin/activity")


def _require_user(request: Request) -> uuid.UUID:
    identity = get_identity(request)
    if not identity or identity.user_id == "anonymous":
        raise HTTPException(401, "Activity requires login")
    try:
        return uuid.UUID(identity.user_id)
    except (TypeError, ValueError):
        raise HTTPException(401, "Invalid identity")


def _require_admin(request: Request) -> None:
    if not is_admin(request):
        raise HTTPException(403, "Admin only")


def _serialize(r: EventLog) -> dict:
    return ActivityEvent(
        id=str(r.id),
        user_id=str(r.user_id) if r.user_id else None,
        event_type=r.event_type,
        target_kind=r.target_kind,
        target_id=str(r.target_id) if r.target_id else None,
        payload=r.payload or {},
        created_at=r.created_at.isoformat() if r.created_at else "",
    ).model_dump()


async def _summary_for(user_filter: uuid.UUID | None) -> ActivitySummary:
    async with AsyncSessionLocal() as session:
        q_counts = select(EventLog.event_type, func.count())
        if user_filter is not None:
            q_counts = q_counts.where(EventLog.user_id == user_filter)
        q_counts = q_counts.group_by(EventLog.event_type)
        rows = (await session.execute(q_counts)).all()
        valid_fields = ActivityCounts.model_fields.keys()
        counts_kwargs = {typ: cnt for typ, cnt in rows if typ in valid_fields}
        counts = ActivityCounts(**counts_kwargs)

        q_payload = select(EventLog.payload).where(EventLog.event_type == "chat_turn")
        if user_filter is not None:
            q_payload = q_payload.where(EventLog.user_id == user_filter)
        tokens_total = 0
        usd_total = 0.0
        for (p,) in (await session.execute(q_payload)).all():
            p = p or {}
            tokens_total += int(p.get("tokens_in") or 0) + int(p.get("tokens_out") or 0)
            usd_total += float(p.get("usd") or 0.0)

        cutoff = datetime.now(timezone.utc) - timedelta(days=28)
        q_heat = (
            select(
                func.date_trunc("day", EventLog.created_at).label("day"),
                func.count(),
            )
            .where(EventLog.created_at >= cutoff)
        )
        if user_filter is not None:
            q_heat = q_heat.where(EventLog.user_id == user_filter)
        q_heat = q_heat.group_by("day").order_by("day")
        heatmap = [
            ActivityHeatmapCell(date=row[0].date().isoformat(), count=row[1])
            for row in (await session.execute(q_heat)).all()
        ]

    return ActivitySummary(
        counts=counts,
        tokens_total=tokens_total,
        usd_estimate=round(usd_total, 6),
        heatmap=heatmap,
    )


async def _events_for(
    user_filter: uuid.UUID | None,
    *,
    event_type: str | None,
    limit: int,
    before_id: str | None,
) -> dict:
    limit = max(1, min(limit, 200))
    async with AsyncSessionLocal() as session:
        q = select(EventLog)
        if user_filter is not None:
            q = q.where(EventLog.user_id == user_filter)
        if event_type:
            q = q.where(EventLog.event_type == event_type)
        if before_id:
            try:
                anchor = await session.get(EventLog, uuid.UUID(before_id))
                if anchor and anchor.created_at:
                    q = q.where(EventLog.created_at < anchor.created_at)
            except ValueError:
                pass
        q = q.order_by(desc(EventLog.created_at)).limit(limit)
        rows = (await session.execute(q)).scalars().all()
    entries = [_serialize(r) for r in rows]
    next_before_id = str(rows[-1].id) if len(rows) >= limit else None
    return {"entries": entries, "next_before_id": next_before_id}


@router.get("/summary")
async def summary(request: Request):
    uid = _require_user(request)
    return (await _summary_for(uid)).model_dump()


@router.get("/events")
async def events(
    request: Request,
    type: str | None = None,
    limit: int = 50,
    before_id: str | None = None,
):
    uid = _require_user(request)
    return await _events_for(uid, event_type=type, limit=limit, before_id=before_id)


@router.get("/events/{event_id}")
async def event_detail(event_id: str, request: Request):
    uid = _require_user(request)
    try:
        eid = uuid.UUID(event_id)
    except ValueError:
        raise HTTPException(404, "Not found")
    async with AsyncSessionLocal() as session:
        row = await session.get(EventLog, eid)
        if not row or row.user_id != uid:
            raise HTTPException(404, "Not found")
        return _serialize(row)


@admin_router.get("/summary")
async def admin_summary(request: Request, user_id: str | None = None):
    _require_admin(request)
    f: uuid.UUID | None = None
    if user_id:
        try:
            f = uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(400, "Invalid user_id")
    return (await _summary_for(f)).model_dump()


@admin_router.get("/events")
async def admin_events(
    request: Request,
    user_id: str | None = None,
    type: str | None = None,
    limit: int = 50,
    before_id: str | None = None,
):
    _require_admin(request)
    f: uuid.UUID | None = None
    if user_id:
        try:
            f = uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(400, "Invalid user_id")
    return await _events_for(f, event_type=type, limit=limit, before_id=before_id)


@admin_router.get("/users")
async def admin_users(request: Request):
    _require_admin(request)
    async with AsyncSessionLocal() as session:
        q = (
            select(
                EventLog.user_id,
                func.count().label("event_count"),
                func.max(EventLog.created_at).label("last_seen"),
            )
            .group_by(EventLog.user_id)
            .order_by(desc("last_seen"))
        )
        rows = (await session.execute(q)).all()
        ids = [r[0] for r in rows if r[0]]
        users_by_id: dict[uuid.UUID, User] = {}
        if ids:
            ures = await session.execute(select(User).where(User.id.in_(ids)))
            users_by_id = {u.id: u for u in ures.scalars().all()}
    entries = []
    for r in rows:
        u = users_by_id.get(r[0]) if r[0] else None
        entries.append(
            AdminUserEntry(
                user_id=str(r[0]) if r[0] else None,
                email=u.email if u else None,
                name=u.name if u else None,
                event_count=r[1],
                last_seen=r[2].isoformat() if r[2] else None,
            ).model_dump()
        )
    return entries


# ── Ingest progress (live document processing) ────────────────────────────────

_INGEST_STATUS_ORDER = {"processing": 0, "queued": 1, "pending": 2, "failed": 3, "done": 4}


def _doc_progress_row(d: Document) -> dict:
    total = d.graph_total_chunks or 0
    processed = d.graph_processed_chunks or 0
    failed_chunks = d.graph_failed_chunks or 0
    pct = round(processed / total * 100, 1) if total else (
        100.0 if d.graph_status == "done" else 0.0
    )
    return {
        "document_id": str(d.id),
        "title": d.title,
        "source_type": d.source_type,
        "graph_status": d.graph_status,
        "progress_pct": pct,
        "chunks_total": total,
        "chunks_done": processed,
        "chunks_failed": failed_chunks,
        "graph_last_error": d.graph_last_error,
        "created_at": d.created_at.isoformat() if d.created_at else None,
        "updated_at": d.updated_at.isoformat() if d.updated_at else None,
    }


async def _ingest_progress_for(user_filter: uuid.UUID | None, include_done: bool, limit: int):
    """Return active (and optionally recent-done) ingest jobs sorted by status."""
    async with AsyncSessionLocal() as session:
        q = select(Document)
        if user_filter is not None:
            q = q.where(Document.user_id == user_filter)
        if not include_done:
            q = q.where(Document.graph_status.in_(("pending", "queued", "processing", "failed")))
        q = q.order_by(desc(Document.updated_at)).limit(limit)
        docs = (await session.execute(q)).scalars().all()
    rows = [_doc_progress_row(d) for d in docs]
    rows.sort(key=lambda r: (
        _INGEST_STATUS_ORDER.get(r["graph_status"] or "", 5),
        -(r["progress_pct"] or 0),
    ))
    return {"items": rows, "active_count": sum(1 for r in rows if r["graph_status"] in ("pending", "queued", "processing"))}


@router.get("/ingest-progress")
async def ingest_progress(
    request: Request,
    include_done: bool = False,
    limit: int = 30,
):
    """Per-user list of documents currently being ingested + recent done/failed."""
    user_id = _require_user(request)
    return await _ingest_progress_for(user_id, include_done=include_done, limit=limit)


@admin_router.get("/ingest-progress")
async def admin_ingest_progress(
    request: Request,
    user_id: str | None = None,
    include_done: bool = False,
    limit: int = 50,
):
    """Global ingest progress feed (admin only). Optional user_id filter."""
    _require_admin(request)
    f: uuid.UUID | None = None
    if user_id:
        try:
            f = uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(400, "Invalid user_id")
    return await _ingest_progress_for(f, include_done=include_done, limit=limit)
