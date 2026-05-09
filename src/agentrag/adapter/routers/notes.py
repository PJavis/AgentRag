"""Notes CRUD backed by adapter_notes table."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from src.agentrag.adapter.db import AdapterNote
from src.agentrag.adapter.models import NoteCreate, NoteResponse, NoteUpdate
from src.agentrag.database import AsyncSessionLocal

router = APIRouter(prefix="/notes")


def _fmt(note: AdapterNote) -> dict:
    return NoteResponse(
        id=str(note.id),
        title=note.title,
        content=note.content,
        note_type=note.note_type,
        created=note.created_at.isoformat() if note.created_at else "",
        updated=note.updated_at.isoformat() if note.updated_at else "",
    ).model_dump()


@router.get("")
async def list_notes(notebook_id: str | None = None):
    async with AsyncSessionLocal() as session:
        q = select(AdapterNote)
        if notebook_id:
            q = q.where(AdapterNote.notebook_id == uuid.UUID(notebook_id))
        notes = (await session.execute(q)).scalars().all()
        return [_fmt(n) for n in notes]


@router.post("")
async def create_note(body: NoteCreate):
    async with AsyncSessionLocal() as session:
        note = AdapterNote(
            notebook_id=uuid.UUID(body.notebook_id) if body.notebook_id else None,
            title=body.title,
            content=body.content,
            note_type=body.note_type or "human",
        )
        session.add(note)
        await session.commit()
        await session.refresh(note)
        return _fmt(note)


@router.get("/{note_id}")
async def get_note(note_id: str):
    async with AsyncSessionLocal() as session:
        note = await session.get(AdapterNote, uuid.UUID(note_id))
        if not note:
            raise HTTPException(404, "Note not found")
        return _fmt(note)


@router.put("/{note_id}")
async def update_note(note_id: str, body: NoteUpdate):
    async with AsyncSessionLocal() as session:
        note = await session.get(AdapterNote, uuid.UUID(note_id))
        if not note:
            raise HTTPException(404, "Note not found")
        if body.title is not None:
            note.title = body.title
        if body.content is not None:
            note.content = body.content
        if body.note_type is not None:
            note.note_type = body.note_type
        await session.commit()
        await session.refresh(note)
        return _fmt(note)


@router.delete("/{note_id}")
async def delete_note(note_id: str):
    async with AsyncSessionLocal() as session:
        note = await session.get(AdapterNote, uuid.UUID(note_id))
        if not note:
            raise HTTPException(404, "Note not found")
        await session.delete(note)
        await session.commit()
        return {"message": "Note deleted"}
