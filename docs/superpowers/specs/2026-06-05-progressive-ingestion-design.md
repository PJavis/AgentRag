# Progressive Ingestion + Live Status — Design

Date: 2026-06-05
Status: approved

## Problem

Uploading docs (esp. scanned/image-heavy PDFs) shows an infinite "Submitting source
for processing…" spinner. Root causes: (a) PDF parser escalates the WHOLE PDF to
MinerU `vlm-auto-engine` if ANY page's text layer is thin → minutes of VLM on a 16GB
GPU; (b) the doc is unusable until the entire parse→chunk→embed→index→extract chain
finishes; (c) the UI only has a binary spinner, no stage/progress. Many docs → bad UX.

## Goals

1. **PyMuPDF-first** parse → doc text-searchable fast; MinerU VLM runs later as
   background enrichment for thin/scanned pages only.
2. **Progressive per-doc status**: `queued → parsing → searchable → enriching → done`.
3. **Non-blocking upload UI** — user keeps working; doc shows a live progress chip.
4. **Searchable-first, enrich-later** — chat works as soon as text segments indexed.

## Decisions (locked)

- Scanned/image-only PDFs: index whatever text PyMuPDF extracts, enrich via MinerU
  (re-index appended segments). No "hold until MinerU".
- Progress delivery: **SSE** over Valkey pub/sub (not polling).
- MinerU runs on **thin pages only** (per-page routing via temp sub-PDF), not whole doc.

## Architecture

### 1. Pipeline phases (worker `graph_ingest` / `graph_jobs.process_graph_job`)

- **PARSE-FAST** — PyMuPDF per page. Keep text pages; collect pages with
  `len(text) < PDF_OCR_MIN_TEXT_CHARS` into `thin_pages[]`. Publish
  `parse_done/parse_total`. (Replaces the all-or-nothing `any(thin)→whole-PDF-MinerU`.)
- **INDEX-TEXT** — chunk PyMuPDF text → embed (TEI) → index (ES). Set status
  `searchable`. Publish per-batch `chunks_done/chunks_total`.
- **ENRICH (background tail, same job)** — if `thin_pages` non-empty: split those
  pages into a temp sub-PDF (PyMuPDF `select`), run MinerU `vlm-auto-engine` on it,
  map recovered text+images back to real page numbers, chunk/embed/index (append).
  Then vision captions + StructMem extract (DeepSeek). Set status `done`.
  - MinerU enrich serialized (worker concurrency 1 for the enrich step) → avoid 16GB OOM.
  - On ENRICH failure: status `done_partial`, store `graph_last_error`; doc stays
    usable (text already indexed). Never stuck/`failed` once `searchable`.

### 2. Status model (`Document`, Postgres)

- Extend `graph_status`: `queued | parsing | searchable | enriching | done | done_partial | failed`.
- New columns (alembic migration): `parse_total_pages INT`, `parse_done_pages INT`.
  (`graph_total_chunks`/`graph_processed_chunks` already exist for chunk progress.)
- Frontend `_STATUS_MAP` maps: `searchable|enriching|done|done_partial` → "completed"
  (usable), `queued|parsing` → "processing", `failed` → "failed". A separate
  `stage` field drives the chip label/sub-progress.

### 3. Live progress (SSE + Valkey pub/sub)

- Worker publishes JSON to Valkey channel `ingest:progress:{user_id}` at each tick:
  `{source_id, stage, parse_done, parse_total, chunks_done, chunks_total, error?}`.
- New `GET /on/api/sources/progress/stream` (auth): subscribe the user's channel,
  relay as SSE `event: progress`. Heartbeat every ~15s. One connection per tab.

### 4. Non-blocking UI (Next.js)

- Upload POST already returns 200 immediately. FE: optimistically insert the doc as a
  `queued` chip, **close the dialog at once**, drop the infinite "Submitting…" modal →
  toast "Upload started".
- `useIngestProgress` hook opens one `EventSource` at notebook/app-shell level
  (survives navigation) → progress store. Each source row renders a **stage chip +
  progress bar** (`Parsing 12/40` → `Searchable ✓ · enriching…` → `Ready`).
  On `searchable`/`done`/`done_partial` → invalidate the sources query.

## Components & boundaries

- `ingestion/parsers/pdf_parser.py` — return `(text_pages, thin_page_numbers)`; stop
  whole-PDF MinerU escalation.
- `ingestion/parsers/mineru_parser.py` — add `parse_pages(path, page_numbers)` (sub-PDF).
- `graph/graph_jobs.py` — orchestrate phases + publish progress + set stages.
- `common/progress.py` (new) — thin `publish_progress(user_id, payload)` Valkey helper.
- `adapter/routers/sources.py` — add `/progress/stream` SSE endpoint.
- FE: `lib/hooks/useIngestProgress.ts`, `components/source/IngestStatusChip.tsx`,
  upload dialog + source list wiring.

## Out of scope (YAGNI)

- Pause/resume/cancel of in-flight ingest.
- Multi-file batch progress bar (single overall %); per-doc chips suffice.
- Reordering/priority queue.

## Testing

- Unit: `thin_pages` detection; sub-PDF page mapping; `_STATUS_MAP` stage mapping.
- Integration: text-rich PDF → `searchable` quickly, no MinerU; mixed PDF → MinerU
  only on thin pages; enrich failure → `done_partial` and doc still searchable.
- SSE: publish → stream delivers events for the right user only.
