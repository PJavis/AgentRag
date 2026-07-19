"""Slice C Phase 2 — augment already-ingested prod docs with gemini image captions.

Drives the existing vision_extract worker (process_vision_job) over each Document,
adding segment_type=image segments ONLY (text segments untouched). NOT a re-ingest.

The worker is not idempotent (appends, never dedups), so we DELETE existing image
segments for a doc (Postgres + Elasticsearch) before (re)processing it — making the
script safe to re-run and resumable.

Env (export BEFORE running — settings is an import-time singleton):
  VISION_PROVIDER=gemini VISION_MODEL=gemini-2.5-flash VISION_MAX_RPM=10
  VISION_DESCRIBE_BATCH=1 VISUAL_EMBEDDING_ENABLED=false
  (+ live POSTGRES_DB / ELASTICSEARCH_INDEX_NAME for prod, or scratch for a dry-run)
"""
import argparse, asyncio, sys
sys.path.insert(0, ".")
from sqlalchemy import select, delete
from elasticsearch import AsyncElasticsearch
from src.agentrag.config import settings
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document, Segment
from src.agentrag.ingestion.parsers.pdf_parser import PDFParser
from src.agentrag.graph.vision_jobs import process_vision_job, VisionExtractJob
import os

PDF_DIR = settings.ORIGINALS_DIR


def _pdf_path_for(doc) -> str | None:
    """Resolve the on-disk original PDF for a Document.

    The app's own convention (src/agentrag/adapter/routers/sources.py
    `_original_path`) names files `<Document.id>.pdf`. But live prod data
    (checked 2026-07-19: 0/115 docs resolve by id, 114/115 resolve by
    `doc.title`, which holds the original upload UUID) shows files are
    actually on disk under `<Document.title>.pdf` — a pre-existing mismatch
    from an earlier re-ingest that reassigned Document.id without renaming
    the stored file. Try id first (correct future convention), then title
    (matches current reality), so this keeps working either way.
    """
    for stem in (str(doc.id), doc.title):
        p = os.path.join(PDF_DIR, f"{stem}.pdf")
        if os.path.exists(p):
            return p
    return None


async def _delete_existing_image_segments(doc_id, title):
    async with AsyncSessionLocal() as s:
        await s.execute(delete(Segment).where(
            Segment.document_id == doc_id, Segment.segment_type == "image"))
        await s.commit()
    es = AsyncElasticsearch(settings.ELASTICSEARCH_URL)
    try:
        await es.delete_by_query(
            index=settings.ELASTICSEARCH_INDEX_NAME, conflicts="proceed", refresh=True,
            query={"bool": {"must": [
                {"term": {"segment_type": "image"}},
                {"term": {"document_title.keyword": title}}]}})
    finally:
        await es.close()


async def augment_doc(doc, parser):
    pdf = _pdf_path_for(doc)
    if not pdf:
        return {"doc": doc.title, "images": 0, "indexed": 0, "skip": "no-pdf"}
    imgs = parser.extract_images(pdf, doc.title)  # [{page,path,url,bytes,mime,byte_hash}]
    if not imgs:
        return {"doc": doc.title, "images": 0, "indexed": 0}
    records = [{"path": i["path"], "page": i["page"], "mime": i["mime"], "url": i["url"]} for i in imgs]
    await _delete_existing_image_segments(doc.id, doc.title)
    job = VisionExtractJob(document_id=doc.id, title=doc.title, image_records=records)
    rep = await process_vision_job(job)
    return {"doc": doc.title, "images": len(records), "indexed": rep.get("indexed", 0)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--doc-id", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true", help="only print doc/image counts, no writes")
    args = ap.parse_args()
    assert settings.VISION_PROVIDER == "gemini", f"expected gemini, got {settings.VISION_PROVIDER}"

    async with AsyncSessionLocal() as s:
        q = select(Document).order_by(Document.created_at.desc())
        docs = (await s.execute(q)).scalars().all()
    if args.doc_id:
        docs = [d for d in docs if str(d.id) == args.doc_id]
    if args.limit:
        docs = docs[: args.limit]
    print(f"index={settings.ELASTICSEARCH_INDEX_NAME} db={settings.POSTGRES_DB} docs={len(docs)}")

    parser = PDFParser()
    tot_img = tot_idx = 0
    for i, doc in enumerate(docs, 1):
        if args.dry_run:
            pdf = _pdf_path_for(doc)
            n = len(parser.extract_images(pdf, doc.title)) if pdf else 0
            print(f"  [{i}/{len(docs)}] {doc.title[:40]:40} images={n} (dry-run)")
            tot_img += n
            continue
        r = await augment_doc(doc, parser)
        tot_img += r["images"]; tot_idx += r["indexed"]
        print(f"  [{i}/{len(docs)}] {r['doc'][:40]:40} images={r['images']} indexed={r['indexed']} {r.get('skip','')}")
    print(f"TOTAL images={tot_img} indexed={tot_idx}")


asyncio.run(main())
