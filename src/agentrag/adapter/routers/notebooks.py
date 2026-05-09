"""Notebook CRUD endpoints mapping to adapter_notebooks table."""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert

from src.agentrag.adapter.db import AdapterNotebook, AdapterNote, adapter_notebook_sources
from src.agentrag.adapter.models import NotebookCreate, NotebookResponse, NotebookUpdate
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document

router = APIRouter(prefix="/notebooks")


def _fmt(nb: AdapterNotebook, source_count: int = 0, note_count: int = 0) -> dict:
    return NotebookResponse(
        id=str(nb.id),
        name=nb.name,
        description=nb.description or "",
        archived=nb.archived or False,
        created=nb.created_at.isoformat() if nb.created_at else "",
        updated=nb.updated_at.isoformat() if nb.updated_at else "",
        source_count=source_count,
        note_count=note_count,
    ).model_dump()


@router.get("")
async def list_notebooks(archived: bool = False, order_by: str = "created"):
    async with AsyncSessionLocal() as session:
        q = select(AdapterNotebook).where(AdapterNotebook.archived == archived)
        rows = (await session.execute(q)).scalars().all()

        result = []
        for nb in rows:
            sc = (
                await session.execute(
                    select(func.count()).select_from(adapter_notebook_sources).where(
                        adapter_notebook_sources.c.notebook_id == nb.id
                    )
                )
            ).scalar() or 0
            nc = (
                await session.execute(
                    select(func.count()).where(AdapterNote.notebook_id == nb.id)
                )
            ).scalar() or 0
            result.append(_fmt(nb, sc, nc))
        return result


@router.post("")
async def create_notebook(body: NotebookCreate):
    async with AsyncSessionLocal() as session:
        nb = AdapterNotebook(name=body.name, description=body.description)
        session.add(nb)
        await session.commit()
        await session.refresh(nb)
        return _fmt(nb)


@router.get("/{notebook_id}")
async def get_notebook(notebook_id: str):
    async with AsyncSessionLocal() as session:
        nb = await session.get(AdapterNotebook, uuid.UUID(notebook_id))
        if not nb:
            raise HTTPException(404, "Notebook not found")
        sc = (
            await session.execute(
                select(func.count()).select_from(adapter_notebook_sources).where(
                    adapter_notebook_sources.c.notebook_id == nb.id
                )
            )
        ).scalar() or 0
        nc = (
            await session.execute(
                select(func.count()).where(AdapterNote.notebook_id == nb.id)
            )
        ).scalar() or 0
        return _fmt(nb, sc, nc)


@router.put("/{notebook_id}")
async def update_notebook(notebook_id: str, body: NotebookUpdate):
    async with AsyncSessionLocal() as session:
        nb = await session.get(AdapterNotebook, uuid.UUID(notebook_id))
        if not nb:
            raise HTTPException(404, "Notebook not found")
        if body.name is not None:
            nb.name = body.name
        if body.description is not None:
            nb.description = body.description
        if body.archived is not None:
            nb.archived = body.archived
        await session.commit()
        await session.refresh(nb)
        return _fmt(nb)


@router.get("/{notebook_id}/delete-preview")
async def delete_preview(notebook_id: str):
    async with AsyncSessionLocal() as session:
        sc = (
            await session.execute(
                select(func.count()).select_from(adapter_notebook_sources).where(
                    adapter_notebook_sources.c.notebook_id == uuid.UUID(notebook_id)
                )
            )
        ).scalar() or 0
        return {"source_count": sc, "exclusive_source_count": sc}


@router.delete("/{notebook_id}")
async def delete_notebook(notebook_id: str, delete_exclusive_sources: bool = False):
    async with AsyncSessionLocal() as session:
        nb_uuid = uuid.UUID(notebook_id)
        if delete_exclusive_sources:
            doc_ids = (
                await session.execute(
                    select(adapter_notebook_sources.c.document_id).where(
                        adapter_notebook_sources.c.notebook_id == nb_uuid
                    )
                )
            ).scalars().all()
            for doc_id in doc_ids:
                doc = await session.get(Document, doc_id)
                if doc:
                    await session.delete(doc)
        await session.execute(
            delete(AdapterNotebook).where(AdapterNotebook.id == nb_uuid)
        )
        await session.commit()
        return {"message": "Notebook deleted"}


@router.post("/{notebook_id}/sources/{source_id}")
async def add_source_to_notebook(notebook_id: str, source_id: str):
    async with AsyncSessionLocal() as session:
        await session.execute(
            insert(adapter_notebook_sources).values(
                notebook_id=uuid.UUID(notebook_id),
                document_id=uuid.UUID(source_id),
            ).on_conflict_do_nothing()
        )
        await session.commit()
        return {"message": "Source added to notebook"}


@router.delete("/{notebook_id}/sources/{source_id}")
async def remove_source_from_notebook(notebook_id: str, source_id: str):
    async with AsyncSessionLocal() as session:
        await session.execute(
            delete(adapter_notebook_sources).where(
                adapter_notebook_sources.c.notebook_id == uuid.UUID(notebook_id),
                adapter_notebook_sources.c.document_id == uuid.UUID(source_id),
            )
        )
        await session.commit()
        return {"message": "Source removed from notebook"}


@router.post("/{notebook_id}/context")
async def get_notebook_context(notebook_id: str, body: dict = {}):
    return {
        "context": "",
        "source_count": 0,
        "token_count": 0,
    }
