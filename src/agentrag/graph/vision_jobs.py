"""Vision extraction job — async describe + index image segments.

Runs after the main ingest pipeline has stored text segments (so sparse + dense
retrieval is ready). Reads images from disk, calls Vision LLM concurrently with
a token-bucket RPM cap (so we don't burst past the provider's per-minute quota),
builds image segments, then upserts them into ES (and PG segments table).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

from src.agentrag.config import settings
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document, Segment
from src.agentrag.ingestion.embedders.factory import build_embedding_provider
from src.agentrag.ingestion.parsers.image_parser import ImageParser
from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore
from src.agentrag.services.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)


@dataclass
class VisionExtractJob:
    document_id: uuid.UUID
    title: str
    image_records: list[dict[str, Any]]   # [{path, page, mime, url}]


class _RpmBucket:
    """Async token bucket: refills `rpm` tokens per 60s. acquire() blocks until a token is free."""

    def __init__(self, rpm: int):
        self.rpm = max(0, int(rpm))
        self._lock = asyncio.Lock()
        # Timestamps of recent acquisitions (monotonic seconds).
        self._stamps: list[float] = []

    async def acquire(self) -> None:
        if self.rpm <= 0:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                cutoff = now - 60.0
                self._stamps = [t for t in self._stamps if t > cutoff]
                if len(self._stamps) < self.rpm:
                    self._stamps.append(now)
                    return
                # Need to wait until oldest stamp slides out of the window.
                wait = 60.0 - (now - self._stamps[0]) + 0.05
            await asyncio.sleep(max(0.0, wait))


# Transient errors worth retrying. We sniff on substring so we don't have to
# import provider-specific exception classes.
_TRANSIENT_HINTS = (
    "429", "rate_limit", "rate limit", "RESOURCE_EXHAUSTED",
    "503", "504", "timeout", "TimeoutError",
)


def _is_transient(exc: BaseException) -> bool:
    s = f"{type(exc).__name__}: {exc}"
    return any(h.lower() in s.lower() for h in _TRANSIENT_HINTS)


async def _describe_one(
    image_parser: ImageParser,
    img: dict[str, Any],
    title: str,
    sem: asyncio.Semaphore,
    bucket: _RpmBucket,
    retries: int,
) -> dict[str, Any] | None:
    path = Path(img["path"])
    if not path.exists():
        logger.warning("vision_extract: missing image %s", path)
        return None
    try:
        img_bytes = path.read_bytes()
    except OSError as exc:
        logger.warning("vision_extract: cannot read %s: %s", path, exc)
        return None

    attempt = 0
    while True:
        async with sem:
            await bucket.acquire()
            try:
                description = await image_parser.describe(
                    img_bytes, img.get("mime", "image/jpeg"), context=title
                )
            except Exception as exc:
                if attempt < retries and _is_transient(exc):
                    backoff = min(2 ** attempt, 30)
                    logger.warning(
                        "vision_extract: transient error on %s (attempt %d/%d): %s — retry in %ds",
                        path, attempt + 1, retries, exc, backoff,
                    )
                    attempt += 1
                    await asyncio.sleep(backoff)
                    continue
                logger.exception("vision_extract: describe failed for %s: %s", path, exc)
                return None
            break

    if not description or description.startswith("[image"):
        return None

    return {
        "content": description,
        "content_hash": hashlib.sha256(description.encode("utf-8")).hexdigest(),
        "segment_type": "image",
        "section_path": f"page_{img['page']}_image",
        "position": 0,  # rewritten by caller after collecting all
        "page_start": img["page"],
        "page_end": img["page"],
        "metadata": {
            "document_title": title,
            "image_url": img.get("url", ""),
            "image_path": img["path"],
        },
    }


async def process_vision_job(job: VisionExtractJob) -> dict[str, Any]:
    """Describe each image concurrently, embed, then upsert segments to PG + ES."""
    if not job.image_records:
        return {"described": 0, "indexed": 0}

    if not settings.VISION_PROVIDER:
        logger.warning("vision_extract: VISION_PROVIDER not set; skipping doc %s", job.document_id)
        return {"described": 0, "indexed": 0, "reason": "vision_disabled"}

    image_parser = ImageParser(LLMGateway())
    es_store = ElasticsearchStore()
    embedder = build_embedding_provider(settings)

    sem = asyncio.Semaphore(max(1, settings.VISION_MAX_CONCURRENCY))
    bucket = _RpmBucket(settings.VISION_MAX_RPM)
    retries = max(0, settings.VISION_PER_IMAGE_RETRIES)
    batch_size = max(1, settings.VISION_FLUSH_BATCH_SIZE)

    started = time.perf_counter()

    # Resolve starting position once — image chunks share document with text chunks.
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Segment.position).where(Segment.document_id == job.document_id)
        )
        positions = [r[0] for r in result.all()]
    base_pos = (max(positions) + 1) if positions else 0

    pending: list[dict[str, Any]] = []
    total_indexed = 0
    described = 0

    async def _flush() -> None:
        nonlocal total_indexed
        if not pending:
            return
        # Embed + assign positions starting at base_pos + total_indexed.
        embeddings = await embedder.embed([c["content"] for c in pending])
        for i, (c, emb) in enumerate(zip(pending, embeddings)):
            c["embedding"] = emb
            c["position"] = base_pos + total_indexed + i

        # P0.2: also compute CLIP visual embedding for each image so the
        # retriever can do cross-modal kNN (text query → image vector).
        if settings.VISUAL_EMBEDDING_ENABLED:
            try:
                from src.agentrag.ingestion.embedders.visual_embedder import VisualEmbedder
                ve = VisualEmbedder.get()
                img_bytes_list: list[bytes] = []
                for c in pending:
                    p = Path(c["metadata"].get("image_path", ""))
                    try:
                        img_bytes_list.append(p.read_bytes() if p.exists() else b"")
                    except OSError:
                        img_bytes_list.append(b"")
                # Filter out zero-byte (missing) entries to keep alignment.
                valid = [(i, b) for i, b in enumerate(img_bytes_list) if b]
                if valid:
                    valid_bytes = [b for _, b in valid]
                    vecs = ve.embed_images(valid_bytes)
                    for (idx, _), v in zip(valid, vecs):
                        pending[idx]["image_embedding"] = v
            except Exception:
                logger.exception("vision_extract: visual embedding failed (skipping)")

        async with AsyncSessionLocal() as session:
            for c in pending:
                session.add(Segment(
                    document_id=job.document_id,
                    content=c["content"],
                    content_hash=c["content_hash"],
                    segment_type=c["segment_type"],
                    section_path=c["section_path"],
                    position=c["position"],
                    extra_metadata=c["metadata"],
                    version=1,
                ))
            await session.commit()

        await es_store.index_segments(pending, job.title)
        total_indexed += len(pending)
        logger.info(
            "vision_extract: flushed batch (cumulative indexed=%d) doc=%s",
            total_indexed, job.document_id,
        )
        pending.clear()

    # Batch describe — group N images per LLM call to cut RPM cost. Set
    # VISION_DESCRIBE_BATCH=1 to disable (per-image). Default 4 keeps free
    # Gemini 12 RPM viable for docs with many figures.
    describe_batch = max(1, getattr(settings, "VISION_DESCRIBE_BATCH", 4))
    if describe_batch == 1:
        tasks = [
            asyncio.create_task(
                _describe_one(image_parser, img, job.title, sem, bucket, retries)
            )
            for img in job.image_records
        ]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            described += 1
            if result is not None:
                pending.append(result)
            if len(pending) >= batch_size:
                await _flush()
    else:
        records = job.image_records
        for start in range(0, len(records), describe_batch):
            chunk = records[start:start + describe_batch]
            loaded: list[tuple[bytes, str]] = []
            chunk_valid: list[dict[str, Any]] = []
            for img in chunk:
                p = Path(img["path"])
                if not p.exists():
                    continue
                try:
                    loaded.append((p.read_bytes(), img.get("mime", "image/jpeg")))
                    chunk_valid.append(img)
                except OSError:
                    continue
            if not loaded:
                continue
            await bucket.acquire()
            try:
                descs = await image_parser.describe_batch(loaded, context=job.title)
            except Exception:
                logger.exception("vision_extract: batch describe failed; per-image fallback")
                descs = []
                for img_bytes, mime in loaded:
                    d = await image_parser._describe(img_bytes, mime, context=job.title)
                    descs.append(d)
            for img, desc in zip(chunk_valid, descs):
                described += 1
                if not desc or desc.startswith("[image"):
                    continue
                pending.append({
                    "content": desc,
                    "content_hash": hashlib.sha256(desc.encode("utf-8")).hexdigest(),
                    "segment_type": "image",
                    "section_path": f"page_{img['page']}_image",
                    "position": 0,
                    "page_start": img["page"],
                    "page_end": img["page"],
                    "metadata": {
                        "document_title": job.title,
                        "image_url": img.get("url", ""),
                        "image_path": img["path"],
                    },
                })
                if len(pending) >= batch_size:
                    await _flush()

    await _flush()

    elapsed = time.perf_counter() - started
    logger.info(
        "vision_extract: described=%d/%d indexed=%d in %.1fs doc=%s",
        total_indexed, len(job.image_records), total_indexed, elapsed, job.document_id,
    )
    return {"described": total_indexed, "indexed": total_indexed}
