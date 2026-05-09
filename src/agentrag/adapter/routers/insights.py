"""Source insights: per-source LLM-generated artifacts.

Open-notebook UI lets the user click "Generate insight" on a source. We
run the picked transformation against the source's full text (concatenated
chunks) and store the result in `adapter_source_insights` keyed by
(source_id, insight_type).
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from src.agentrag.adapter.db import AdapterNote, AdapterSourceInsight, AdapterTransformation
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document, Segment

router = APIRouter()


def _fmt(insight: AdapterSourceInsight) -> dict[str, Any]:
    return {
        "id": str(insight.id),
        "source_id": str(insight.source_id),
        "insight_type": insight.insight_type,
        "content": insight.content,
        "created": insight.created_at.isoformat() if insight.created_at else "",
        "updated": insight.updated_at.isoformat() if insight.updated_at else "",
    }


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _source_full_text(source_id: uuid.UUID, max_chars: int = 60_000) -> str:
    """Concatenate the source's text segments. Caps at max_chars to bound LLM cost."""
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(Segment)
                .where(Segment.document_id == source_id)
                .order_by(Segment.position.asc())
            )
        ).scalars().all()
    texts = [s.content for s in rows if s.content and (s.segment_type or "text") == "text"]
    full = "\n\n".join(texts)
    if len(full) > max_chars:
        full = full[:max_chars] + "\n\n[... truncated ...]"
    return full


async def run_transformation(
    source_id: str, transformation_name: str, request: Request | None = None
) -> dict[str, Any]:
    """Run the named transformation on a source's full text and store the result."""
    from src.agentrag.adapter.routers.transformations import _ensure_seeded
    from src.agentrag.services.llm_gateway import LLMGateway

    await _ensure_seeded()
    try:
        sid = uuid.UUID(source_id.removeprefix("source:"))
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid source_id")

    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, sid)
        if not doc:
            raise HTTPException(404, "Source not found")
        row = await session.execute(
            select(AdapterTransformation).where(AdapterTransformation.name == transformation_name)
        )
        transformation = row.scalar_one_or_none()
    if transformation is None:
        raise HTTPException(404, f"Transformation '{transformation_name}' not found")

    text = await _source_full_text(sid)
    if not text.strip():
        raise HTTPException(400, "Source has no extracted text yet — wait for ingestion to finish")

    llm = LLMGateway()
    output = await llm.text_response(
        system_prompt=transformation.prompt,
        user_prompt=text,
        task="answer",
    )

    async with AsyncSessionLocal() as session:
        # Upsert by (source_id, insight_type)
        existing = await session.execute(
            select(AdapterSourceInsight).where(
                AdapterSourceInsight.source_id == sid,
                AdapterSourceInsight.insight_type == transformation_name,
            )
        )
        insight = existing.scalar_one_or_none()
        if insight is None:
            insight = AdapterSourceInsight(
                source_id=sid,
                insight_type=transformation_name,
                content=output,
            )
            session.add(insight)
        else:
            insight.content = output
        await session.commit()
        await session.refresh(insight)
        return _fmt(insight)


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/sources/{source_id}/insights")
async def list_source_insights(source_id: str):
    try:
        sid = uuid.UUID(source_id.removeprefix("source:"))
    except (ValueError, TypeError):
        return []
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(AdapterSourceInsight)
                .where(AdapterSourceInsight.source_id == sid)
                .order_by(AdapterSourceInsight.created_at.desc())
            )
        ).scalars().all()
        return [_fmt(r) for r in rows]


@router.post("/sources/{source_id}/insights")
async def create_source_insight(source_id: str, body: dict, request: Request):
    """Create a new insight by running a transformation on the source.

    Body: {"transformation": "summary"} or {"insight_type": "summary"}.
    """
    name = body.get("transformation") or body.get("insight_type") or body.get("type") or "summary"
    return await run_transformation(source_id, name, request)


@router.get("/insights/{insight_id}")
async def get_insight(insight_id: str):
    async with AsyncSessionLocal() as session:
        try:
            iid = uuid.UUID(insight_id)
        except (ValueError, TypeError):
            raise HTTPException(400, "Invalid id")
        ins = await session.get(AdapterSourceInsight, iid)
        if not ins:
            raise HTTPException(404, "Insight not found")
        return _fmt(ins)


@router.delete("/insights/{insight_id}")
async def delete_insight(insight_id: str):
    async with AsyncSessionLocal() as session:
        ins = await session.get(AdapterSourceInsight, uuid.UUID(insight_id))
        if not ins:
            raise HTTPException(404, "Insight not found")
        await session.delete(ins)
        await session.commit()
        return {"message": "Insight deleted"}


@router.post("/insights/{insight_id}/save-as-note")
async def save_insight_as_note(insight_id: str, body: dict):
    """Convert an insight into a notebook note."""
    nb_id = body.get("notebook_id")
    async with AsyncSessionLocal() as session:
        ins = await session.get(AdapterSourceInsight, uuid.UUID(insight_id))
        if not ins:
            raise HTTPException(404, "Insight not found")
        note = AdapterNote(
            notebook_id=uuid.UUID(nb_id) if nb_id else None,
            title=f"{ins.insight_type.title()} — {ins.created_at.strftime('%Y-%m-%d') if ins.created_at else ''}",
            content=ins.content,
            note_type="ai",
        )
        session.add(note)
        await session.commit()
        await session.refresh(note)
        return {
            "id": str(note.id),
            "title": note.title,
            "content": note.content,
            "note_type": note.note_type,
            "created": note.created_at.isoformat() if note.created_at else "",
            "updated": note.updated_at.isoformat() if note.updated_at else "",
        }
