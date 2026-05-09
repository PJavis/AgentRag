"""Source endpoints: map open-notebook Sources to AgentRag Documents."""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert

from src.agentrag.adapter.db import adapter_notebook_sources
from src.agentrag.adapter.models import SourceResponse, SourceStatusResponse
from src.agentrag.adapter.upload_dedupe import find_existing_document, hash_bytes
from src.agentrag.config import settings
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document, Segment
from src.agentrag.ingestion.pipeline import ingest_folder

router = APIRouter(prefix="/sources")


def _parse_source_id(source_id: str) -> uuid.UUID:
    """Strip optional 'source:' prefix and parse UUID."""
    clean = source_id.removeprefix("source:")
    return uuid.UUID(clean)


# queued = ES indexing done (searchable); only "pending" means still processing
_STATUS_MAP = {
    "pending": "processing",
    "queued": "completed",
    "processing": "processing",
    "done": "completed",
    "failed": "failed",
    None: "processing",
}


async def _segment_count(session, doc_id) -> int:
    row = await session.execute(
        select(func.count()).where(Segment.document_id == doc_id)
    )
    return row.scalar() or 0


async def _doc_to_source(
    session, doc: Document, notebook_ids: list[str] | None = None
) -> dict:
    status = _STATUS_MAP.get(doc.graph_status, "processing")
    seg_count = await _segment_count(session, doc.id)
    return SourceResponse(
        id=str(doc.id),
        title=doc.title,
        topics=None,
        asset=None,
        full_text=None,
        embedded=seg_count > 0,
        embedded_chunks=seg_count,
        created=doc.created_at.isoformat() if doc.created_at else "",
        updated=doc.updated_at.isoformat() if doc.updated_at else "",
        status=status,
        notebooks=notebook_ids,
    ).model_dump()


@router.get("")
async def list_sources(
    notebook_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    sort_by: str = "created",
    sort_order: str = "desc",
):
    async with AsyncSessionLocal() as session:
        if notebook_id:
            doc_ids = (
                await session.execute(
                    select(adapter_notebook_sources.c.document_id).where(
                        adapter_notebook_sources.c.notebook_id == uuid.UUID(notebook_id)
                    )
                )
            ).scalars().all()
            q = select(Document).where(Document.id.in_(doc_ids))
        else:
            q = select(Document)

        order_col = Document.created_at if sort_by == "created" else Document.updated_at
        q = q.order_by(order_col.desc() if sort_order == "desc" else order_col.asc())
        q = q.offset(offset).limit(limit)
        docs = (await session.execute(q)).scalars().all()

        result = []
        for doc in docs:
            nb_ids = (
                await session.execute(
                    select(adapter_notebook_sources.c.notebook_id).where(
                        adapter_notebook_sources.c.document_id == doc.id
                    )
                )
            ).scalars().all()
            result.append(await _doc_to_source(session, doc, [str(n) for n in nb_ids]))
        return result


# Concurrent-upload throttle: prevent the same user spamming uploads from
# all running in parallel and saturating the ingestion pipeline.
_UPLOAD_SEMAPHORES: dict[str, asyncio.Semaphore] = {}


def _user_upload_semaphore(user_id: str) -> asyncio.Semaphore:
    sem = _UPLOAD_SEMAPHORES.get(user_id)
    if sem is None:
        sem = asyncio.Semaphore(2)  # at most 2 concurrent ingests per user
        _UPLOAD_SEMAPHORES[user_id] = sem
    return sem


async def _link_to_notebooks(session, doc_id: uuid.UUID, notebook_ids: list[str]) -> None:
    for nb_id in notebook_ids:
        try:
            nb_uuid = uuid.UUID(nb_id)
        except (ValueError, TypeError):
            continue
        await session.execute(
            insert(adapter_notebook_sources)
            .values(notebook_id=nb_uuid, document_id=doc_id)
            .on_conflict_do_nothing()
        )


@router.post("")
async def create_source(
    request: Request,
    file: UploadFile = File(None),
    type: str = Form("upload"),
    notebooks: list[str] = Form(default=[]),
    notebook_id: str = Form(None),
    title: str = Form(None),
    url: str = Form(None),
    content: str = Form(None),
    async_processing: bool = Form(False),
):
    """Upload a file and ingest it into AgentRag with dedupe + concurrency control."""
    if file is None and content is None and url is None:
        raise HTTPException(400, "file, content, or url is required")

    identity = getattr(request.state, "auth_identity", None)
    user_id = identity.user_id if identity else "anonymous"
    all_notebooks = list({n for n in ([notebook_id] + (notebooks or [])) if n})

    # Read source bytes once so we can hash + dedupe BEFORE invoking the pipeline.
    if file is not None:
        raw = await file.read()
        if len(raw) == 0:
            raise HTTPException(400, "Uploaded file is empty")
        if len(raw) > settings.UPLOAD_MAX_BYTES:
            raise HTTPException(
                413, f"File exceeds maximum size of {settings.UPLOAD_MAX_BYTES} bytes"
            )
        filename = file.filename or "upload"
    elif content:
        raw = content.encode("utf-8")
        filename = (title or "note") + ".md"
    else:
        raise HTTPException(400, "URL ingestion not yet supported")

    content_hash = hash_bytes(raw)

    # Dedupe: if we've already ingested identical bytes, link to notebooks and return.
    if settings.UPLOAD_DEDUPE_BY_HASH:
        existing = await find_existing_document(content_hash)
        if existing is not None:
            async with AsyncSessionLocal() as session:
                await _link_to_notebooks(session, existing.id, all_notebooks)
                await session.commit()
                doc = await session.get(Document, existing.id)
                payload = await _doc_to_source(session, doc, all_notebooks)
                payload["deduplicated"] = True
                return payload

    sem = _user_upload_semaphore(user_id)
    async with sem:
        tmp_dir = tempfile.mkdtemp(prefix="adapter_upload_")
        try:
            dest = os.path.join(tmp_dir, filename)
            with open(dest, "wb") as f:
                f.write(raw)
            result = await ingest_folder(tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    docs = result.get("documents", [])
    if not docs:
        raise HTTPException(500, "Ingestion returned no documents")
    doc_id = docs[0].get("document_id")
    if not doc_id:
        raise HTTPException(500, "Could not determine document_id after ingestion")

    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, uuid.UUID(doc_id))
        if not doc:
            raise HTTPException(500, "Document not found after ingestion")
        # Stamp content hash so future uploads dedupe.
        if not doc.content_hash and content_hash:
            doc.content_hash = content_hash
        await _link_to_notebooks(session, doc.id, all_notebooks)
        await session.commit()
        return await _doc_to_source(session, doc, all_notebooks)


@router.post("/json")
async def create_source_json(request: Request, body: dict):
    """Create source from JSON body (text content)."""
    return await create_source(
        request,
        file=None,
        type="text",
        notebooks=body.get("notebooks", []),
        notebook_id=body.get("notebook_id"),
        title=body.get("title", "note"),
        url=None,
        content=body.get("content", ""),
        async_processing=False,
    )


@router.get("/{source_id}")
async def get_source(source_id: str):
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, _parse_source_id(source_id))
        if not doc:
            raise HTTPException(404, "Source not found")
        nb_ids = (
            await session.execute(
                select(adapter_notebook_sources.c.notebook_id).where(
                    adapter_notebook_sources.c.document_id == doc.id
                )
            )
        ).scalars().all()
        return await _doc_to_source(session, doc, [str(n) for n in nb_ids])


@router.put("/{source_id}")
async def update_source(source_id: str, body: dict):
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, _parse_source_id(source_id))
        if not doc:
            raise HTTPException(404, "Source not found")
        if "title" in body:
            doc.title = body["title"]
        await session.commit()
        await session.refresh(doc)
        return await _doc_to_source(session, doc)


@router.delete("/{source_id}")
async def delete_source(source_id: str):
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, _parse_source_id(source_id))
        if not doc:
            raise HTTPException(404, "Source not found")
        await session.execute(
            delete(Segment).where(Segment.document_id == doc.id)
        )
        await session.delete(doc)
        await session.commit()
        return {"message": "Source deleted"}


@router.get("/{source_id}/status")
async def source_status(source_id: str):
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, _parse_source_id(source_id))
        if not doc:
            raise HTTPException(404, "Source not found")
        seg_count = await _segment_count(session, doc.id)
        return SourceStatusResponse(
            id=str(doc.id),
            status=_STATUS_MAP.get(doc.graph_status, "processing"),
            embedded=seg_count > 0,
            embedded_chunks=seg_count,
        ).model_dump()


@router.post("/{source_id}/retry")
async def retry_source(source_id: str):
    return await get_source(source_id)


@router.get("/{source_id}/insights")
async def list_source_insights(source_id: str):
    """Return insights cached on Document.extra_metadata (set by transformations)."""
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, _parse_source_id(source_id))
        if not doc:
            return []
    return []  # actual insight storage handled by adapter.routers.insights


@router.post("/{source_id}/insights")
async def trigger_insight(source_id: str, body: dict, request: Request):
    """Async-create an insight (summary/key-points/etc) via transformation."""
    from src.agentrag.adapter.routers.insights import run_transformation

    transformation = body.get("transformation") or body.get("type") or "summary"
    return await run_transformation(source_id, transformation, request)
