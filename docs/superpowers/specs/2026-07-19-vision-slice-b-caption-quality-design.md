# Vision Slice B — local caption-quality assessment (2026-07-19)

## Why

Slice A routed answer-time multimodal to a vision model but is inert until images are
ingested. Slice B is the **go/no-go gate for the expensive full-corpus re-ingest (Slice C)**:
it measures whether a *local* vision model (Ollama `qwen2.5-vl:7b`) produces
**medically-useful captions** of this corpus's images. Discovery found there is **no
image-dependent eval-Q set** to measure end-to-end QA against (a full scan of every eval file
returned exactly one image-referencing question, and it is a leaked exam-proctoring screenshot,
not a medical figure). So we measure caption quality **directly** with an independent judge
(Gemini), not downstream QA. Cheap, decisive, PHI-local at ingest.

## Goal / success criteria

A measured caption-quality signal + a pre-registered Slice C verdict. Done when:

1. ~5 deduplicated image-heavy PDFs are ingested into a non-destructive sandbox with
   `VISION_PROVIDER=ollama` / `VISION_MODEL=qwen2.5-vl:7b`, sync captioning, and image
   segments (`segment_type=image`, `content=<caption>`, `metadata.image_url`) land in the
   scratch ES index.
2. Up to ~40 captioned images are judged by Gemini (image + qwen caption → medical
   accuracy/usefulness score 1–5 + failure-mode label).
3. A results doc reports mean score, score distribution, failure-mode tally, and the
   pre-registered Slice C go/no-go.

## Non-goals

- End-to-end image-QA measurement (no eval-Q set exists — explicitly out of scope; would be a
  separate eval-set-construction effort).
- Full-corpus re-ingest (Slice C).
- Committing `VISION_PROVIDER` on by default — config default stays `None`; sandbox uses env.
- CLIP visual embeddings / async worker path.

## Key discovery facts (drive the design)

- **Wiring (sync):** `pipeline.py:203-227` — for each `pdf_parser.extract_images` result,
  `image_parser.describe(bytes, mime, context) → vision_response(task="vision") →
  _get_vision_client()` → for `VISION_PROVIDER=ollama`, `make_async_openai(base_url=OLLAMA_BASE_URL)`
  and an OpenAI-chat `image_url` (base64 data URL) call. Caption failure → `[image …]` string →
  silently dropped (`pipeline.py:210`). Image segment fields: `pipeline.py:212-225`.
- **Standalone images** (`source_type=image`) → `image_parser.parse` captions then chunks the
  description; PDFs are the relevant case here (all corpus docs are PDFs).
- **Sandbox isolation:** `ingest_folder(folder_path)` takes NO db/index arg — both come from the
  `settings` singleton read at first import, so scratch overrides MUST be OS env vars exported
  **before** the process starts: `POSTGRES_DB=rag_scratch`, `ELASTICSEARCH_INDEX_NAME=agentrag_segments_scratch`,
  `TAGGING_ENABLED=false` (SectionTagger hits `ontology_terms`, which `Base.metadata.create_all`
  does NOT create + needs pg_trgm — off avoids it), `VISION_PROVIDER=ollama`,
  `VISION_MODEL=qwen2.5-vl:7b`, `VISION_INGEST_MODE=sync`. Leave `STRUCTMEM_INGEST_MODE=async`
  (its `enqueue_job` raises `RuntimeError` in a bare script — caught per-doc `pipeline.py:398-416`;
  do NOT `init_pool` — a shared-Redis worker would pick the job into prod graph state).
- **Scratch bootstrap:** `psql -h127.0.0.1 -p5433 -U postgres -c "CREATE DATABASE rag_scratch"`;
  then `from src.agentrag.database import engine, Base; await conn.run_sync(Base.metadata.create_all)`.
- **Model:** `qwen2.5-vl:7b` NOT local → `ollama pull qwen2.5-vl:7b` (~6GB). ollama dies on this
  WSL box; `nohup ollama serve &` + verify before ingest.
- **Corpus duplicates:** 114 PDFs with heavy byte-identical duplicates — pick from the
  DEDUPED list. extract_images additionally filters small icons + byte-hash-dedups, so ingested
  image counts run lower than raw xref counts.

## Design

### Doc subset (~5, deduped, distinct md5)
- `0c560778-9ed1-428a-ac4e-0c4c900f2e4e.pdf` — 161 img / 26 pp, text+embedded figures (dense).
- `28f3ad1c-faf0-4d51-9af4-da45d0b22069.pdf` — 52 img / 25 pp, text+embedded.
- `534533a9-55eb-47ad-a762-b10435291892.pdf` — 31 img / 13 pp, text+embedded.
- `1617bcff-9bf1-41de-aa8d-9b5fcfc5f78e.pdf` — 13 img / 10 pp, text+embedded.
- `162d54a5-eeac-4454-8ecb-ffdfef710dec.pdf` — 25 img / 22 pp, SCANNED (0 text) — tests the
  scanned-page path too.

Copy these 5 into a scratch folder; ingest that folder (not all of data/originals).

### Pipeline
```
setup env + scratch DB/index → ingest_folder(scratch_5_docs)  [qwen2.5-vl sync captions, LOCAL]
  → scratch ES has segment_type=image chunks (content=caption, metadata.image_url/image_path)
sample ≤40 image segments → for each: read image bytes from image_path
  → Gemini judge(image + caption) → {score 1-5, failure_mode, note}
aggregate → results doc → pre-registered Slice C verdict
```

### Judge (independent, Gemini)
For each sampled image segment, send the actual image + the qwen caption to
`gemini-2.5-flash`: "Score how accurately + usefully this caption describes this MEDICAL image
for retrieval (1=wrong/hallucinated, 5=accurate+specific). Label failure mode ∈
{accurate, generic-uninformative, missed-key-finding, hallucinated-finding, ocr-of-text-slide,
unreadable}." Return JSON. (Reuses the project's gemini path / a small script.)

### Pre-registered Slice C verdict
- **GO (recommend full re-ingest):** mean score ≥ 3.5/5 AND hallucinated-finding rate < 15%.
- **NO-GO / try better model:** mean < 3.5 or hallucinated ≥ 15% → qwen2.5-vl not good enough;
  reconsider model (larger VL / gemini-ingest despite PHI) before spending Slice C.
- Report the split for scanned vs text+embedded images separately (scanned pages may need OCR,
  not captioning).

## PHI note
Ingest captioning is 100% local (Ollama). The **assessment judge sends the sampled images to
Gemini** (cloud) — bounded, one-time, ~40 images, assessment-only (not a prod path), and
consistent with the Slice-A mixed-provider choice. Documented, not silent.

## Testing / verification
- Assert the sandbox ES index has `segment_type=image` segments with non-empty `content`
  (captions) after ingest, and that ingest made zero cloud calls (all vision to localhost:11434).
- The judge run is the measurement; its aggregate + verdict is the deliverable.
- Sandbox (scratch DB + parallel index + IMAGE_STORAGE_DIR scratch) dropped at the end.

## Deliverable
`docs/eval/vision_caption_quality_2026-07-19.md` — subset, per-image scores, failure-mode tally,
scanned-vs-embedded split, and the Slice C go/no-go. Verification-heavy slice: committed
artifacts are the results doc + the reusable judge/ingest scripts (if worth keeping) + any
wiring fix discovered; no prod config default flip.
