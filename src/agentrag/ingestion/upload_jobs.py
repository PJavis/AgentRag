"""Ingest for a UI upload — runs in the ARQ worker (default) or in-process.

The adapter's `POST /sources` writes the upload to a staging directory and
creates a skeleton Document; this module does the rest: parse + index via
`ingest_folder`, link the document to its notebooks, record the activity event
and remove the staging directory.

It used to run only as a FastAPI BackgroundTask inside the api process, where
sync parsers (PDF/DOCX/OCR) block the event loop for every other request, and
an api restart mid-ingest silently drops the job, leaving the document
"pending" forever. As an ARQ job it runs in the worker and survives restarts.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import time
import uuid

from sqlalchemy.dialects.postgresql import insert

from src.agentrag.adapter.db import adapter_notebook_sources
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document

logger = logging.getLogger(__name__)


async def link_to_notebooks(session, doc_id: uuid.UUID, notebook_ids: list[str]) -> None:
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


async def _mark_failed(doc_id: str | None, error: str) -> None:
    """A job that dies before the pipeline touches the skeleton must not leave
    it "pending" forever — the UI would spin on it indefinitely."""
    if not doc_id:
        return
    try:
        async with AsyncSessionLocal() as session:
            doc = await session.get(Document, uuid.UUID(doc_id))
            if doc is not None and doc.graph_status in (None, "pending", "parsing"):
                doc.graph_status = "failed"
                doc.graph_last_error = error[:8000]
                await session.commit()
    except Exception:
        logger.exception("upload ingest: could not mark %s failed", doc_id)


async def run_upload_ingest(
    file_path: str,
    user_id: str,
    notebook_ids: list[str],
    content_hash: str | None,
    *,
    skeleton_id: str | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> str | None:
    """Ingest one staged upload. Returns the document id, or None on failure.

    Never raises: the caller is a background task or a worker job, and a
    failure is recorded on the document and in the activity log instead.
    """
    from src.agentrag.ingestion.pipeline import ingest_folder
    from src.agentrag.observability.activity import record_event

    parent_dir = os.path.dirname(file_path)
    started = time.perf_counter()
    doc_id: str | None = None
    segment_count: int | None = None
    error_msg: str | None = None
    try:
        if not os.path.exists(file_path):
            error_msg = f"staged upload missing: {file_path}"
            logger.error("upload ingest: %s (is the staging dir shared with the worker?)", error_msg)
            return None
        async with semaphore if semaphore is not None else contextlib.nullcontext():
            result = await ingest_folder(parent_dir, user_id=user_id)
        docs = result.get("documents", [])
        if not docs:
            logger.error("upload ingest returned no documents for %s", file_path)
            error_msg = "no documents produced"
            return None
        doc_id = docs[0].get("document_id")
        segment_count = docs[0].get("segments_indexed") or docs[0].get("chunks")
        if not doc_id:
            return None
        async with AsyncSessionLocal() as session:
            doc = await session.get(Document, uuid.UUID(doc_id))
            if doc is None:
                return None
            if not doc.content_hash and content_hash:
                doc.content_hash = content_hash
            await link_to_notebooks(session, doc.id, notebook_ids)
            await session.commit()
        return doc_id
    except Exception as exc:
        logger.exception("upload ingest failed for %s", file_path)
        error_msg = str(exc)[:500]
        return None
    finally:
        shutil.rmtree(parent_dir, ignore_errors=True)
        if error_msg:
            await _mark_failed(doc_id or skeleton_id, error_msg)
        duration_ms = round((time.perf_counter() - started) * 1000.0, 1)
        try:
            await record_event(
                user_id=user_id,
                event_type="ingest_failed" if error_msg else "ingest_done",
                target_kind="document",
                target_id=doc_id or skeleton_id,
                payload={
                    "duration_ms": duration_ms,
                    "segment_count": segment_count,
                    "error": error_msg,
                },
            )
        except Exception:
            logger.debug("upload ingest: activity event failed", exc_info=True)
