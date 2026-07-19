"""Slice C Phase 2 — augment already-ingested prod docs with gemini image captions.

Drives the existing vision_extract worker (process_vision_job) over each Document,
adding segment_type=image segments ONLY (text segments untouched). NOT a re-ingest.

The worker is not idempotent (appends, never dedups), so we DELETE existing image
segments for a doc (Postgres + Elasticsearch) before (re)processing it — making the
script safe to re-run and resumable.

`--dry-run` is strictly READ-ONLY: it opens each PDF with `fitz` and counts raw
image xrefs per page (`page.get_images(full=True)`) without ever calling
`PDFParser.extract_images()` — that method decodes, dedups, filters, AND WRITES
each kept image to disk under `IMAGE_STORAGE_DIR`, so it is never invoked here.
Dry-run counts are therefore approximate (printed with "≈") — the real run's
kept-image count can be lower once small/duplicate images are filtered out.

Hardened for an unattended ~2h/114-doc run (Task 3): per-doc exceptions are
caught and logged (the batch continues), stale image segments are cleaned up
even when a doc yields zero images, and "soft refusal" captions (e.g. "I'm
unable to view the image..." — seen both from a real vision model declining
an image, and as the tell-tale symptom of the VISION_BASE_URL footgun below,
where a non-vision text model silently stands in for gemini) are detected and
removed after indexing rather than left in the index as if they were real
content.

Env (export BEFORE running — settings is an import-time singleton):
  VISION_PROVIDER=gemini VISION_MODEL=gemini-2.5-flash VISION_BASE_URL= VISION_MAX_RPM=10
  VISION_DESCRIBE_BATCH=1 VISUAL_EMBEDDING_ENABLED=false
  (+ live POSTGRES_DB / ELASTICSEARCH_INDEX_NAME for prod)

FOOTGUN: the home .env sets VISION_BASE_URL to a local Ollama URL. If
VISION_PROVIDER=gemini and VISION_BASE_URL still points at Ollama, gemini calls
silently misroute to Ollama; since the gemini model name doesn't exist there,
it silently falls further back to LLM_FALLBACK_MODEL (a small non-vision text
model) — no error, garbage/refusal-shaped captions. WORSE: config.py's
env_ignore_empty=True means the documented `VISION_BASE_URL=` (empty) shell
override does NOT actually clear it (pydantic-settings treats an empty OS env
var as unset and falls through to .env's non-empty value) — so this script
detects the caller's intent from the raw OS env var and force-clears
settings.VISION_BASE_URL in-process when it sees `VISION_BASE_URL=` was
explicitly passed empty; it only raises a hard error when VISION_BASE_URL was
left completely unexported (or pointed at Ollama on purpose). Always export
`VISION_BASE_URL=` (empty) alongside VISION_PROVIDER=gemini regardless.
"""
import argparse, asyncio, sys
sys.path.insert(0, ".")
from sqlalchemy import select, delete
from elasticsearch import AsyncElasticsearch
from src.agentrag.config import settings
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document, Segment
from src.agentrag.ingestion.parsers.pdf_parser import PDFParser, _SAFE_DIRNAME_RE
from src.agentrag.graph.vision_jobs import process_vision_job, VisionExtractJob
import os

PDF_DIR = settings.ORIGINALS_DIR

# Soft-refusal patterns that can show up in place of a real caption — either
# a genuine vision-model refusal, or (observed in practice) a non-vision text
# model silently standing in for gemini due to the VISION_BASE_URL footgun
# below, producing "I can't view images, please describe it" style answers.
# process_vision_job's own quality gate only rejects its own "[image ...]"
# placeholder text, so these slip through and get indexed as if they were
# real content.
_REFUSAL_HINTS = (
    "unable to view", "can't view", "cannot view", "i'm unable",
    "cannot assist", "provide more context", "unable to assist",
)


def _is_refusal(content: str) -> bool:
    c = (content or "").lower()
    return any(h in c for h in _REFUSAL_HINTS)


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

    `doc.title` is user-editable/unsanitized (not a trusted filesystem name),
    so before joining it into a path we (a) sanitize it with the same
    `_SAFE_DIRNAME_RE` the codebase already uses for filesystem-derived names
    (pdf_parser.py) and (b) verify the resolved realpath stays inside
    PDF_DIR — belt-and-suspenders against path traversal (e.g. a title of
    "../../etc/passwd").
    """
    root = os.path.realpath(PDF_DIR) + os.sep
    for raw_stem in (str(doc.id), doc.title or ""):
        stem = _SAFE_DIRNAME_RE.sub("_", raw_stem)[:80]
        if not stem:
            continue
        candidate = os.path.join(PDF_DIR, f"{stem}.pdf")
        real = os.path.realpath(candidate)
        if not real.startswith(root):
            continue  # path-traversal guard: resolved path escaped PDF_DIR
        if os.path.exists(real):
            return real
    return None


def _count_images_readonly(pdf_path: str) -> int:
    """Approximate image count for --dry-run WITHOUT extracting or writing anything.

    Unlike PDFParser.extract_images() — which decodes each raster, dedups by
    byte-hash, filters images below IMAGE_MIN_SIZE_BYTES, and writes each kept
    image to disk under IMAGE_STORAGE_DIR — this only opens the PDF and counts
    raw image xrefs per page via fitz. No decode, no filter, no filesystem
    writes beyond fitz's own read of the PDF bytes.
    """
    import fitz
    doc = fitz.open(pdf_path)
    try:
        return sum(len(page.get_images(full=True)) for page in doc)
    finally:
        doc.close()


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


async def _remove_refusal_captions(doc_id, title) -> int:
    """Delete this doc's freshly-indexed image segments whose content is a
    vision-model soft refusal rather than an actual caption (see
    _REFUSAL_HINTS). Only ever touches image-type segments for THIS
    document_id — called right after indexing a doc's own fresh batch.

    PG and ES are correlated via `content_hash` (identical sha256 of the
    caption text, stored on both sides by vision_jobs.py / elasticsearch_store.py)
    rather than a text wildcard on `content` — an exact keyword match is more
    precise than substring-matching an analyzed text field.

    Returns the number of refusal segments removed.
    """
    async with AsyncSessionLocal() as s:
        result = await s.execute(
            select(Segment.id, Segment.content, Segment.content_hash).where(
                Segment.document_id == doc_id, Segment.segment_type == "image"))
        refused = [(rid, chash) for rid, content, chash in result.all() if _is_refusal(content)]
        if not refused:
            return 0
        await s.execute(delete(Segment).where(Segment.id.in_([rid for rid, _ in refused])))
        await s.commit()

    hashes = [h for _, h in refused if h]
    if hashes:
        es = AsyncElasticsearch(settings.ELASTICSEARCH_URL)
        try:
            await es.delete_by_query(
                index=settings.ELASTICSEARCH_INDEX_NAME, conflicts="proceed", refresh=True,
                query={"bool": {"must": [
                    {"term": {"segment_type": "image"}},
                    {"term": {"document_title.keyword": title}},
                    {"terms": {"content_hash": hashes}}]}})
        finally:
            await es.close()
    return len(refused)


async def augment_doc(doc, parser):
    pdf = _pdf_path_for(doc)
    if not pdf:
        return {"doc": doc.title, "images": 0, "indexed": 0, "refused": 0, "skip": "no-pdf"}
    imgs = parser.extract_images(pdf, doc.title)  # [{page,path,url,bytes,mime,byte_hash}]
    # Delete stale image segments BEFORE the empty-check: even a doc that now
    # yields zero images (e.g. re-extracted PDF, changed filter) must not be
    # left with leftover image segments from a prior run.
    await _delete_existing_image_segments(doc.id, doc.title)
    if not imgs:
        return {"doc": doc.title, "images": 0, "indexed": 0, "refused": 0}
    records = [{"path": i["path"], "page": i["page"], "mime": i["mime"], "url": i["url"]} for i in imgs]
    job = VisionExtractJob(document_id=doc.id, title=doc.title, image_records=records)
    rep = await process_vision_job(job)
    refused = await _remove_refusal_captions(doc.id, doc.title)
    return {"doc": doc.title, "images": len(records), "indexed": rep.get("indexed", 0), "refused": refused}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--doc-id", type=str, default=None)
    ap.add_argument(
        "--dry-run", action="store_true",
        help="read-only: approximate per-page image count via a fitz page scan "
             "(no extract_images() call, no writes to data/images, DB, or ES)",
    )
    args = ap.parse_args()
    assert settings.VISION_PROVIDER == "gemini", f"expected gemini, got {settings.VISION_PROVIDER}"

    # FOOTGUN: config.py sets env_ignore_empty=True ("treat empty env vars as
    # missing, use defaults"). That means `VISION_BASE_URL=` (empty) at the
    # shell — the documented workaround — does NOT clear settings.VISION_BASE_URL:
    # pydantic-settings treats the empty OS env var as "not provided" and falls
    # through to the non-empty Ollama URL baked into .env. Task 1's own report
    # independently hit this (same "cannot view images" fallback signature) and
    # worked around it by passing the FULL gemini URL explicitly (a non-empty
    # override propagates fine). We detect the caller's *intent* from the raw
    # OS env var (present-and-empty = "please clear it") and honor it in-process,
    # rather than silently trusting settings.VISION_BASE_URL (which may still
    # hold the stale Ollama value despite that intent).
    _raw_base_url = os.environ.get("VISION_BASE_URL")
    if _raw_base_url == "":
        settings.VISION_BASE_URL = None
    elif settings.VISION_BASE_URL and any(
        bad in settings.VISION_BASE_URL.lower() for bad in ("11434", "ollama")
    ):
        raise RuntimeError(
            f"VISION_BASE_URL={settings.VISION_BASE_URL!r} looks like a local Ollama endpoint "
            "while VISION_PROVIDER=gemini — this silently misroutes gemini calls to Ollama "
            "(then, since the model doesn't exist there, falls back to LLM_FALLBACK_MODEL, a "
            "non-vision text model) with no error and produces garbage captions. Export "
            "VISION_BASE_URL= (empty) before running this script — and note that alone is not "
            "always enough: rerun and confirm this error clears, since env_ignore_empty=True can "
            "make an empty override silently no-op if this script's own workaround above isn't "
            "reached (e.g. if this guard is bypassed by importing main() differently)."
        )

    async with AsyncSessionLocal() as s:
        q = select(Document).order_by(Document.created_at.desc())
        docs = (await s.execute(q)).scalars().all()
    if args.doc_id:
        docs = [d for d in docs if str(d.id) == args.doc_id]
    if args.limit:
        docs = docs[: args.limit]
    print(f"index={settings.ELASTICSEARCH_INDEX_NAME} db={settings.POSTGRES_DB} docs={len(docs)}")

    parser = PDFParser()
    tot_img = tot_idx = tot_refused = n_failed = 0
    for i, doc in enumerate(docs, 1):
        if args.dry_run:
            pdf = _pdf_path_for(doc)
            n = _count_images_readonly(pdf) if pdf else 0
            print(f"  [{i}/{len(docs)}] {doc.title[:40]:40} images≈{n} (dry-run, read-only)")
            tot_img += n
            continue
        try:
            r = await augment_doc(doc, parser)
        except Exception as exc:
            n_failed += 1
            print(f"  [{i}/{len(docs)}] {doc.title[:40]:40} FAILED: {type(exc).__name__}: {exc}")
            continue
        tot_img += r["images"]; tot_idx += r["indexed"]; tot_refused += r.get("refused", 0)
        print(
            f"  [{i}/{len(docs)}] {r['doc'][:40]:40} images={r['images']} indexed={r['indexed']} "
            f"refused={r.get('refused', 0)} {r.get('skip', '')}"
        )
    if args.dry_run:
        print(f"TOTAL images≈{tot_img} (dry-run, read-only, no writes)")
    else:
        print(f"TOTAL images={tot_img} indexed={tot_idx} refused={tot_refused} failed={n_failed}")


asyncio.run(main())
