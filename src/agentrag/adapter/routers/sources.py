"""Source endpoints: map open-notebook Sources to AgentRag Documents."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert

from src.agentrag.adapter.db import adapter_notebook_sources
from src.agentrag.adapter.models import SourceResponse, SourceStatusResponse
from src.agentrag.adapter.upload_dedupe import find_existing_document, hash_bytes
from src.agentrag.config import settings
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document, Project, Segment
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


async def _stitched_full_text(session, doc_id) -> str | None:
    """Concatenate all segment contents ordered by position for preview UI.

    Cheap enough for typical docs (<100 chunks). Returns None if no segments.
    """
    rows = (
        await session.execute(
            select(Segment.content, Segment.section_path, Segment.position)
            .where(Segment.document_id == doc_id)
            .order_by(Segment.position.asc())
        )
    ).all()
    if not rows:
        return None
    parts: list[str] = []
    for content, section_path, _pos in rows:
        if section_path:
            parts.append(f"## {section_path}")
        if content:
            parts.append(content)
    return "\n\n".join(parts) if parts else None


async def _doc_to_source(
    session,
    doc: Document,
    notebook_ids: list[str] | None = None,
    *,
    include_full_text: bool = False,
) -> dict:
    status = _STATUS_MAP.get(doc.graph_status, "processing")
    seg_count = await _segment_count(session, doc.id)
    full_text = await _stitched_full_text(session, doc.id) if include_full_text else None
    return SourceResponse(
        id=str(doc.id),
        title=doc.title,
        topics=None,
        asset=None,
        full_text=full_text,
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


async def _run_ingest_background(
    file_path: str,
    user_id: str,
    notebook_ids: list[str],
    content_hash: str | None,
) -> None:
    """Run ingest_folder in background, then link notebooks + cleanup tmp dir."""
    import logging
    logger = logging.getLogger(__name__)
    sem = _user_upload_semaphore(user_id)
    parent_dir = os.path.dirname(file_path)
    try:
        async with sem:
            result = await ingest_folder(parent_dir)
        docs = result.get("documents", [])
        if not docs:
            logger.error("Background ingest returned no documents for %s", file_path)
            return
        doc_id = docs[0].get("document_id")
        if not doc_id:
            return
        async with AsyncSessionLocal() as session:
            doc = await session.get(Document, uuid.UUID(doc_id))
            if doc is None:
                return
            if not doc.content_hash and content_hash:
                doc.content_hash = content_hash
            await _link_to_notebooks(session, doc.id, notebook_ids)
            await session.commit()
    except Exception:
        logger.exception("Background ingest failed for %s", file_path)
    finally:
        shutil.rmtree(parent_dir, ignore_errors=True)


@router.post("")
async def create_source(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    type: str = Form("upload"),
    notebooks: list[str] = Form(default=[]),
    notebook_id: str = Form(None),
    title: str = Form(None),
    url: str = Form(None),
    content: str = Form(None),
    async_processing: bool = Form(False),
):
    """Upload a file. Returns a placeholder ASAP; ingest runs in background."""
    if file is None and content is None and url is None:
        raise HTTPException(400, "file, content, or url is required")

    identity = getattr(request.state, "auth_identity", None)
    user_id = identity.user_id if identity else "anonymous"

    # Frontend sends `notebooks` as a single JSON-stringified form field
    # (multipart with one entry, not repeated fields). FastAPI then hands us
    # `['["uuid","uuid"]']` instead of `["uuid","uuid"]`. Unwrap that.
    flat_notebooks: list[str] = []
    for raw in (notebooks or []):
        if not isinstance(raw, str):
            continue
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    flat_notebooks.extend(str(x) for x in parsed if x)
                    continue
            except json.JSONDecodeError:
                pass
        flat_notebooks.append(s)
    all_notebooks = list({n for n in ([notebook_id] + flat_notebooks) if n})

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

    # Dedupe: identical bytes already ingested → link + return real doc.
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

    # Persist file outside request lifecycle.
    persist_dir = tempfile.mkdtemp(prefix="adapter_upload_")
    dest = os.path.join(persist_dir, filename)
    with open(dest, "wb") as f:
        f.write(raw)

    # Pre-create skeleton Document (status=pending, no segments). Pipeline's
    # postgres_store will detect it by source_id+hash and POPULATE in place
    # → frontend GET /sources sees this row immediately.
    doc_title = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1].lstrip(".") or "upload"
    async with AsyncSessionLocal() as session:
        # documents.project_id is NOT NULL — reuse / create default project
        # (pipeline's postgres_store does the same).
        proj = (await session.execute(
            select(Project).where(Project.name == "Default Project").limit(1)
        )).scalar_one_or_none()
        if proj is None:
            proj = Project(name="Default Project", description="Auto created")
            session.add(proj)
            await session.flush()
        skeleton = Document(
            project_id=proj.id,
            title=doc_title,
            source_type=ext,
            source_id=filename,            # matches FolderConnector.source_id
            content_hash=content_hash,
            graph_synced=False,
            graph_status="pending",
        )
        session.add(skeleton)
        await session.commit()
        await session.refresh(skeleton)
        await _link_to_notebooks(session, skeleton.id, all_notebooks)
        await session.commit()
        payload = await _doc_to_source(session, skeleton, all_notebooks)

    background_tasks.add_task(
        _run_ingest_background, dest, user_id, all_notebooks, content_hash
    )
    payload["status"] = "processing"
    return payload


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
        return await _doc_to_source(
            session, doc, [str(n) for n in nb_ids], include_full_text=True
        )


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
