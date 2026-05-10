"""Vision extraction job — async describe + index image segments.

Runs after the main ingest pipeline has stored text segments (so sparse + dense
retrieval is ready). Reads images from disk, calls Vision LLM, builds image
segments, then upserts them into ES (and PG segments table).
"""
from __future__ import annotations

import hashlib
import logging
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


async def process_vision_job(job: VisionExtractJob) -> dict[str, Any]:
    """Describe each image, embed, then upsert image segments to PG + ES."""
    if not job.image_records:
        return {"described": 0, "indexed": 0}

    if not settings.VISION_PROVIDER:
        logger.warning("vision_extract: VISION_PROVIDER not set; skipping doc %s", job.document_id)
        return {"described": 0, "indexed": 0, "reason": "vision_disabled"}

    image_parser = ImageParser(LLMGateway())
    es_store = ElasticsearchStore()
    embedder = build_embedding_provider(settings)

    img_chunks: list[dict[str, Any]] = []
    next_pos = 0  # placeholder offset; final position read from existing segments

    for img in job.image_records:
        path = Path(img["path"])
        if not path.exists():
            logger.warning("vision_extract: missing image %s", path)
            continue
        try:
            img_bytes = path.read_bytes()
            description = await image_parser.describe(
                img_bytes, img.get("mime", "image/jpeg"), context=job.title
            )
        except Exception as e:
            logger.exception("vision_extract: describe failed for %s: %s", path, e)
            continue
        if not description or description.startswith("[image"):
            continue
        img_chunks.append({
            "content": description,
            "content_hash": hashlib.sha256(description.encode("utf-8")).hexdigest(),
            "segment_type": "image",
            "section_path": f"page_{img['page']}_image",
            "position": next_pos,
            "page_start": img["page"],
            "page_end": img["page"],
            "metadata": {
                "document_title": job.title,
                "image_url": img.get("url", ""),
                "image_path": img["path"],
            },
        })
        next_pos += 1

    if not img_chunks:
        logger.info("vision_extract: 0 valid descriptions for doc %s", job.document_id)
        return {"described": 0, "indexed": 0}

    # Embed
    embeddings = await embedder.embed([c["content"] for c in img_chunks])
    for c, emb in zip(img_chunks, embeddings):
        c["embedding"] = emb

    # Insert PG segments + index ES
    async with AsyncSessionLocal() as session:
        # Find max existing position to avoid collision
        result = await session.execute(
            select(Segment.position).where(Segment.document_id == job.document_id)
        )
        positions = [r[0] for r in result.all()]
        base_pos = (max(positions) + 1) if positions else 0

        for offset, chunk in enumerate(img_chunks):
            chunk["position"] = base_pos + offset
            seg = Segment(
                document_id=job.document_id,
                content=chunk["content"],
                content_hash=chunk["content_hash"],
                segment_type=chunk["segment_type"],
                section_path=chunk["section_path"],
                position=chunk["position"],
                extra_metadata=chunk["metadata"],
                version=1,
            )
            session.add(seg)
        await session.commit()

    await es_store.index_segments(img_chunks, job.title)

    logger.info(
        "vision_extract: described=%d indexed=%d for doc %s",
        len(img_chunks), len(img_chunks), job.document_id,
    )
    return {"described": len(img_chunks), "indexed": len(img_chunks)}
