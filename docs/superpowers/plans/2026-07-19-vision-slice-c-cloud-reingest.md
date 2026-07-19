# Vision Slice C — cloud-vision full-corpus re-ingest + synthesized image eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn vision on for real — land gemini-captioned image segments in the production retrieval index — and measure the answer-quality lift end-to-end.

**Architecture:** Cloud captioning (`gemini-2.5-flash`). Phase 1 re-validates the caption-quality gate with gemini on the Slice-B sandbox. Phase 2 drives the existing `vision_extract` worker over the 114 already-ingested prod docs to ADD `segment_type=image` segments only (a full re-ingest would duplicate text segments because ES assigns random `_id`s). Phase 3 synthesizes an image-dependent eval-Q set (text-answerable questions filtered out). Phase 4 A/Bs answer-time vision ON vs OFF and reports the correctness delta. Controller-run scripts; no prod-code behavior change.

**Tech Stack:** Python/uv, Ollama-free (cloud gemini via OpenAI-compat), Elasticsearch, Postgres (docker `agentrag-postgres`), the project's `vision_extract` worker, `PDFParser.extract_images`, the `oracle_probe`/`run_refusal_ab` eval harness (`load_local_jsonl` + `GraphAgentService.chat` + `score_correctness`).

## Global Constraints

- **Settings is an import-time singleton** (read once at first import). ALL env overrides MUST be exported BEFORE the Python process starts. Never rely on mutating `settings.*` for anything read at import.
- **Corpus deemed non-PHI (2026-07-19 decision)** — cloud captioning sanctioned for ingest. This is the one deliberate departure from the earlier PHI-local-ingest posture.
- **Phase 2 is ADDITIVE to prod.** Prod has ZERO image segments today (vision was off). Touch only `segment_type=image`. Never delete/modify text segments. Rollback = delete image segments.
- **`process_vision_job` is NOT idempotent** — it appends, never dedups. Re-running duplicates image segments in BOTH Postgres and ES. The augment script MUST delete existing image segments for a doc (PG + ES) BEFORE (re)processing it.
- **ES `index_segments` uses `_id = uuid.uuid4()`** (`elasticsearch_store.py:331`) — never content-hash-idempotent. This is WHY Phase 2 augments instead of re-ingesting.
- **Original PDF path = `data/originals/<Document.id>.pdf`** (keyed on the doc UUID, `ORIGINALS_DIR` default `data/originals`, `config.py:204`).
- **Pre-registered caption gate (Phase 1, fixed):** GO iff mean ≥ 3.5 AND hallucinated-finding rate < 0.15.
- **Vision worker env knobs** (`config.py`): `VISION_PROVIDER` (gate), `VISION_MODEL` (read inside `ImageParser`/`LLMGateway`, not the worker), `VISION_MAX_RPM` (default 10 = gemini free tier; token-bucket cap), `VISION_MAX_CONCURRENCY` (4), `VISION_DESCRIBE_BATCH` (default 4; **set to 1 for per-image transient-retry/backoff on 429/RESOURCE_EXHAUSTED**), `VISION_PER_IMAGE_RETRIES` (3), `VISUAL_EMBEDDING_ENABLED` (default True — **set false**: CLIP visual retrieval is a non-goal, avoids the VisualEmbedder model load).
- **Postgres CLI** = `docker exec agentrag-postgres psql -U postgres` (no host `psql`).
- **FOOTGUN (found in Task 1):** the home `.env` sets `VISION_BASE_URL=http://127.0.0.1:11434/v1/` (Ollama) alongside `VISION_PROVIDER=gemini`. The ingest/caption path (`_get_vision_client`) uses `VISION_BASE_URL` when non-empty, so gemini calls silently misroute to Ollama (→ wrong/garbage captions, no error). EVERY gemini-captioning command in this plan MUST export `VISION_BASE_URL=` (empty) to override it and use gemini's default endpoint.
- Requires a working `GEMINI_API_KEY` with enough quota for ~1,334 captions; at `VISION_MAX_RPM=10` the full run is ~2–2.5h. Confirm the key before Phase 2.

---

### Task 1: Phase 1 — re-validate the caption gate with gemini

Re-run the Slice-B harness on the same 5 sandbox PDFs, captioning with gemini instead of qwen, and confirm the pre-registered gate passes before touching prod.

**Files:**
- Reuse (no change): `scripts/eval/vision_sandbox_ingest.py`, `scripts/eval/judge_vision_captions.py`
- Create: `docs/eval/vision_caption_quality_gemini_2026-07-19.md`

**Interfaces:**
- Consumes: `data/originals/{0c560778-82d5-4344-8e5e-db081547b14b,28f3ad1c-faf0-4d51-9af4-da45d0b22069,534533a9-55eb-47ad-a762-b10435291892,1617bcff-9bf1-41de-aa8d-9b5fcfc5f78e,162d54a5-eeac-4454-8ecb-ffdfef710dec}.pdf`; `GEMINI_API_KEY`.
- Produces: gate PASS/FAIL numbers + the results doc.

- [ ] **Step 1: Confirm gemini reachable + create scratch DB**

```bash
cd /home/nguyenquocdung/work/AgentRag && source .env
curl -s -m20 "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions" -H "Authorization: Bearer $GEMINI_API_KEY" -H 'Content-Type: application/json' -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":[{"type":"text","text":"OK"}]}],"max_tokens":5}' -o /dev/null -w "http=%{http_code}\n"
docker exec agentrag-postgres psql -U postgres -c "DROP DATABASE IF EXISTS rag_scratch;" -c "CREATE DATABASE rag_scratch;"
rm -rf /tmp/vis_docs /tmp/vis_images && mkdir -p /tmp/vis_docs
for f in 0c560778-82d5-4344-8e5e-db081547b14b 28f3ad1c-faf0-4d51-9af4-da45d0b22069 534533a9-55eb-47ad-a762-b10435291892 1617bcff-9bf1-41de-aa8d-9b5fcfc5f78e 162d54a5-eeac-4454-8ecb-ffdfef710dec; do cp "data/originals/$f.pdf" /tmp/vis_docs/; done
ls /tmp/vis_docs | wc -l   # expect 5
```
Expected: `http=200`, `CREATE DATABASE`, `5`.

- [ ] **Step 2: Ingest the 5 sandbox docs with gemini captioning**

```bash
POSTGRES_DB=rag_scratch ELASTICSEARCH_INDEX_NAME=agentrag_segments_scratch TAGGING_ENABLED=false \
VISION_PROVIDER=gemini VISION_MODEL=gemini-2.5-flash VISION_INGEST_MODE=sync IMAGE_STORAGE_DIR=/tmp/vis_images \
VISUAL_EMBEDDING_ENABLED=false \
PYTHONPATH=. uv run python scripts/eval/vision_sandbox_ingest.py 2>&1 | tail -5
```
Note: `scripts/eval/vision_sandbox_ingest.py` asserts `VISION_MODEL == "qwen2.5vl:7b"` and `IMAGE_STORAGE_DIR == "/tmp/vis_images"`. Relax the model assert first:

Edit `scripts/eval/vision_sandbox_ingest.py` — change the line
`assert settings.VISION_MODEL == "qwen2.5vl:7b", settings.VISION_MODEL`
to
`assert settings.VISION_PROVIDER in ("ollama", "gemini"), settings.VISION_PROVIDER`
then re-run the command above.
Expected: `INGEST: {'status': 'success', 'ingested': 0, ...}` (ingested=0 is benign — async graph enqueue skip), no traceback.

- [ ] **Step 3: Verify image segments landed in the scratch index**

```bash
ELASTICSEARCH_INDEX_NAME=agentrag_segments_scratch PYTHONPATH=. uv run python -c "
import asyncio
from elasticsearch import AsyncElasticsearch
from src.agentrag.config import settings
async def main():
    es=AsyncElasticsearch(settings.ELASTICSEARCH_URL)
    n=(await es.count(index='agentrag_segments_scratch', query={'term':{'segment_type':'image'}}))['count']
    print('image segments=', n)
    await es.close()
asyncio.run(main())
"
```
Expected: `image segments=` ~100+. If 0, STOP (captioning dropped — check gemini errors).

- [ ] **Step 4: Judge the gemini captions**

```bash
source .env
ELASTICSEARCH_INDEX_NAME=agentrag_segments_scratch GEMINI_API_KEY="$GEMINI_API_KEY" PYTHONPATH=. \
  uv run python scripts/eval/judge_vision_captions.py 2>&1 | tail -6
```
Expected: `MEAN=<x.xx> HALLUC_RATE=<0.xx>` + `VERDICT: GO` (mean ≥ 3.5 AND halluc < 0.15). If NO-GO, STOP the whole slice and report — gemini captions are unexpectedly poor; do not proceed to prod.

- [ ] **Step 5: Write + commit the Phase-1 result doc**

Create `docs/eval/vision_caption_quality_gemini_2026-07-19.md` with: the 5-doc subset, image-segment count, MEAN + HALLUC_RATE + failure-mode tally + per-doc means (from `/tmp/vis_scores.json`), 3–5 example captions, and:
> **Phase-1 gate:** GO iff MEAN ≥ 3.5 AND HALLUC < 0.15. Measured MEAN=⟨…⟩ HALLUC=⟨…⟩ → **⟨GO⟩**. Cleared to proceed to prod augmentation (Phase 2).

```bash
docker exec agentrag-postgres psql -U postgres -c "DROP DATABASE IF EXISTS rag_scratch;"
curl -s -X DELETE "localhost:9200/agentrag_segments_scratch" >/dev/null; rm -rf /tmp/vis_docs /tmp/vis_images
git add scripts/eval/vision_sandbox_ingest.py docs/eval/vision_caption_quality_gemini_2026-07-19.md
git commit -m "docs(eval): vision slice C phase 1 — gemini caption gate re-validation (GO)"
```

---

### Task 2: Phase 2a — prod image-augment script + snapshot + single-doc dry-run

Build the augment driver and prove it on ONE doc into a scratch index first — no prod writes yet.

**Files:**
- Create: `scripts/eval/vision_prod_augment.py`
- Test (manual dry-run): scratch index `agentrag_segments_scratch`

**Interfaces:**
- Consumes: `Document` table (list of docs); `data/originals/<id>.pdf`; `process_vision_job` / `VisionExtractJob` (`src/agentrag/graph/vision_jobs.py`); `PDFParser.extract_images(file_path, title)`.
- Produces: a CLI `python scripts/eval/vision_prod_augment.py [--limit N] [--doc-id UUID] [--dry-run]` that, per doc, deletes existing image segments (PG+ES) then runs `process_vision_job`. Prints per-doc `{doc, images, indexed}`.

- [ ] **Step 1: Write the augment script**

`scripts/eval/vision_prod_augment.py`:
```python
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
    pdf = os.path.join(PDF_DIR, f"{doc.id}.pdf")
    if not os.path.exists(pdf):
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
            pdf = os.path.join(PDF_DIR, f"{doc.id}.pdf")
            n = len(parser.extract_images(pdf, doc.title)) if os.path.exists(pdf) else 0
            print(f"  [{i}/{len(docs)}] {doc.title[:40]:40} images={n} (dry-run)")
            tot_img += n
            continue
        r = await augment_doc(doc, parser)
        tot_img += r["images"]; tot_idx += r["indexed"]
        print(f"  [{i}/{len(docs)}] {r['doc'][:40]:40} images={r['images']} indexed={r['indexed']} {r.get('skip','')}")
    print(f"TOTAL images={tot_img} indexed={tot_idx}")


asyncio.run(main())
```

- [ ] **Step 2: Dry-run against the LIVE index (read-only, counts only)**

```bash
cd /home/nguyenquocdung/work/AgentRag && source .env
VISION_PROVIDER=gemini VISION_MODEL=gemini-2.5-flash VISUAL_EMBEDDING_ENABLED=false \
  PYTHONPATH=. uv run python scripts/eval/vision_prod_augment.py --dry-run 2>&1 | tail -20
```
Expected: `docs=114` and a per-doc image count; `TOTAL images=` ~1000–1400. No writes (dry-run). Confirms PDF resolution + extraction work for real docs.

- [ ] **Step 3: Single-doc REAL run on the LIVE index, twice — idempotency proof**

Run one doc through the augment script twice and assert the image-segment count is stable (the delete-first logic must prevent doubling). This writes real image segments for one doc into prod — acceptable, it is part of the augmentation and re-runnable.

```bash
DOC=$(docker exec agentrag-postgres psql -U postgres -d rag -tAc "SELECT id FROM documents ORDER BY created_at DESC LIMIT 1")
echo "smoke doc=$DOC"
source .env
for run in 1 2; do
  VISION_PROVIDER=gemini VISION_MODEL=gemini-2.5-flash VISION_BASE_URL= VISION_MAX_RPM=10 VISION_DESCRIBE_BATCH=1 VISUAL_EMBEDDING_ENABLED=false \
    GEMINI_API_KEY="$GEMINI_API_KEY" PYTHONPATH=. uv run python scripts/eval/vision_prod_augment.py --doc-id "$DOC" 2>&1 | tail -2
  docker exec agentrag-postgres psql -U postgres -d rag -tAc "SELECT count(*) FROM segments WHERE document_id='$DOC' AND segment_type='image';"
done
```
Expected: both runs print the SAME image-segment count for that doc (delete-first makes re-runs stable, not doubling). This is the idempotency proof.

- [ ] **Step 4: Verify no text segments were touched by the smoke**

```bash
docker exec agentrag-postgres psql -U postgres -d rag -tAc "SELECT segment_type, count(*) FROM segments GROUP BY segment_type ORDER BY 1;"
```
Expected: an `image` row now exists for that one doc; `text`/`table` counts unchanged from before.

- [ ] **Step 5: Commit the augment script**

```bash
docker exec agentrag-postgres psql -U postgres -c "DROP DATABASE IF EXISTS rag_scratch;"
git add scripts/eval/vision_prod_augment.py
git commit -m "feat(vision): prod image-augment driver for slice C (delete-first, additive)"
```

---

### Task 3: Phase 2b — snapshot, full-corpus augmentation, verify, rollback rehearsal

**Files:** none new (runs Task 2's script over all 114 docs).

**Interfaces:**
- Consumes: `scripts/eval/vision_prod_augment.py`.
- Produces: `segment_type=image` segments for all docs in the LIVE `agentrag_segments` index; a backup index for rollback.

- [ ] **Step 1: Snapshot the live index (rollback safety net)**

```bash
cd /home/nguyenquocdung/work/AgentRag
curl -s -X POST "localhost:9200/agentrag_segments/_clone/agentrag_segments_backup_20260719" \
  -H 'Content-Type: application/json' >/dev/null 2>&1 || \
curl -s -X POST "localhost:9200/_reindex" -H 'Content-Type: application/json' -d '{"source":{"index":"agentrag_segments"},"dest":{"index":"agentrag_segments_backup_20260719"}}' | python3 -c "import sys,json;d=json.load(sys.stdin);print('reindexed', d.get('total'))"
# clone needs the source read-only; if it errors, the _reindex fallback runs.
curl -s "localhost:9200/_cat/indices?h=index,docs.count" | grep agentrag_segments
```
Expected: `agentrag_segments_backup_20260719` exists with docs.count == the live count (3359).

- [ ] **Step 2: Record the pre-run baseline counts**

```bash
docker exec agentrag-postgres psql -U postgres -d rag -tAc "SELECT segment_type, count(*) FROM segments GROUP BY segment_type ORDER BY 1;"
curl -s "localhost:9200/agentrag_segments/_count" -H 'Content-Type: application/json' -d '{"query":{"term":{"segment_type":"image"}}}' | python3 -c "import sys,json;print('image segments BEFORE:', json.load(sys.stdin)['count'])"
```
Expected: `image segments BEFORE: 0`.

- [ ] **Step 3: Run the full-corpus augmentation (long — ~2–2.5h at RPM 10)**

```bash
source .env
VISION_PROVIDER=gemini VISION_MODEL=gemini-2.5-flash VISION_BASE_URL= VISION_MAX_RPM=10 VISION_MAX_CONCURRENCY=4 \
VISION_DESCRIBE_BATCH=1 VISION_PER_IMAGE_RETRIES=3 VISUAL_EMBEDDING_ENABLED=false \
GEMINI_API_KEY="$GEMINI_API_KEY" PYTHONPATH=. \
  nohup uv run python scripts/eval/vision_prod_augment.py > /tmp/vis_augment.log 2>&1 &
echo "pid=$!"
```
Poll `/tmp/vis_augment.log` for the final `TOTAL images=<N> indexed=<M>`. Expected: N ≈ 1000–1400, M > 0, no unhandled traceback. Transient 429s are retried by the worker; a doc that fully fails logs + continues.

- [ ] **Step 4: Verify the augmentation landed + text untouched**

```bash
curl -s "localhost:9200/agentrag_segments/_count" -H 'Content-Type: application/json' -d '{"query":{"term":{"segment_type":"image"}}}' | python3 -c "import sys,json;print('image segments AFTER:', json.load(sys.stdin)['count'])"
docker exec agentrag-postgres psql -U postgres -d rag -tAc "SELECT segment_type, count(*) FROM segments GROUP BY segment_type ORDER BY 1;"
# spot-check a few captions are real medical text
curl -s "localhost:9200/agentrag_segments/_search" -H 'Content-Type: application/json' -d '{"size":3,"query":{"term":{"segment_type":"image"}},"_source":["content","document_title"]}' | python3 -c "import sys,json;[print('CAP:',h['_source']['content'][:120]) for h in json.load(sys.stdin)['hits']['hits']]"
```
Expected: `image segments AFTER:` matches the run's `indexed` total (> 0); `text`/`table` counts unchanged vs Step 2; captions are real text.

- [ ] **Step 5: Rehearse rollback on the BACKUP (do not run on live unless reverting)**

Document the exact rollback command (rehearse against the backup index to confirm the query shape):
```bash
# ROLLBACK (only if reverting Slice C): delete every image segment from the live index + PG.
# Rehearse the ES query against the backup so we know it targets only image segments:
curl -s "localhost:9200/agentrag_segments_backup_20260719/_count" -H 'Content-Type: application/json' -d '{"query":{"term":{"segment_type":"image"}}}' | python3 -c "import sys,json;print('backup image segments (should be 0 — snapshot pre-dates augment):', json.load(sys.stdin)['count'])"
# Actual rollback (NOT run now):
#   curl -X POST "localhost:9200/agentrag_segments/_delete_by_query?conflicts=proceed&refresh=true" -H 'Content-Type: application/json' -d '{"query":{"term":{"segment_type":"image"}}}'
#   docker exec agentrag-postgres psql -U postgres -d rag -c "DELETE FROM segments WHERE segment_type='image';"
```
Expected: backup shows `0` image segments (confirms the snapshot is a clean pre-augment restore point + the delete-by-query targets only images).

- [ ] **Step 6: Commit a short runbook note**

Append a "Phase 2 executed" note (date, image-segment count, backup index name, rollback commands) to `docs/eval/vision_caption_quality_gemini_2026-07-19.md` and commit:
```bash
git add docs/eval/vision_caption_quality_gemini_2026-07-19.md
git commit -m "docs(eval): vision slice C phase 2 — prod augmentation runbook + counts"
```

---

### Task 4: Phase 3 — build the synthesized image-dependent eval-Q set

**Files:**
- Create: `scripts/eval/build_vision_evalset.py`
- Output: `data/eval/vision_evalset_2026-07-19.jsonl`

**Interfaces:**
- Consumes: `agentrag_segments` image segments (Phase 2); `GEMINI_API_KEY`; `LLMGateway` (text model for the dependency filter); `score_correctness`.
- Produces: jsonl rows `{id, question, reference_answer, gold_contexts, lang, source, image_path}` matching `load_local_jsonl` (`benchmark_datasets.py:120`), containing ONLY image-dependent questions.

- [ ] **Step 1: Write the eval-set builder**

`scripts/eval/build_vision_evalset.py`:
```python
"""Slice C Phase 3 — synthesize an image-DEPENDENT eval-Q set.

For a stratified sample of augmented image segments:
  1. gemini (vision) drafts a Q + gold answer that requires READING THE IMAGE.
  2. dependency filter: give the project's TEXT model the image's neighbouring
     text (same document, nearby positions) WITHOUT the image + the question; if it
     answers correctly (score_correctness.mean >= 0.6) the Q is text-answerable ->
     DISCARD. Keep only Qs the text model fails -> those measure vision's lift.

Model diversity (anti-circularity): gemini generates, deepseek answers the filter,
score_correctness (task=eval_judge) judges.
"""
import asyncio, base64, json, os, sys
from collections import defaultdict
sys.path.insert(0, ".")
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI
from src.agentrag.config import settings
from src.agentrag.services.llm_gateway import LLMGateway
from src.agentrag.eval.correctness_judge import score_correctness

N_IMAGES = 60          # candidates to draft from (kept set will be smaller after filter)
KEEP_TARGET = 40
TEXT_ANSWERABLE_THRESHOLD = 0.6
OUT = "data/eval/vision_evalset_2026-07-19.jsonl"

GEN_SYS = (
  "You write ONE exam question that can ONLY be answered by LOOKING AT this medical image "
  "(a finding, count, structure, modality, or label visible in the image), NOT from general "
  "knowledge or surrounding prose. Return JSON: {\"question\": \"...\", \"answer\": \"<concise gold>\"}."
)


def _b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


async def _neighbour_text(es, title, position):
    # nearby text segments from the same document = the 'surrounding text' a reader has.
    res = await es.search(index=settings.ELASTICSEARCH_INDEX_NAME, size=6,
        query={"bool": {"must": [{"term": {"document_title.keyword": title}}],
                        "must_not": [{"term": {"segment_type": "image"}}]}},
        _source=["content"])
    return "\n".join(h["_source"].get("content", "") for h in res["hits"]["hits"])[:4000]


async def main():
    es = AsyncElasticsearch(settings.ELASTICSEARCH_URL)
    res = await es.search(index=settings.ELASTICSEARCH_INDEX_NAME, size=500,
        query={"term": {"segment_type": "image"}},
        _source=["content", "metadata", "document_title", "position"])
    hits = res["hits"]["hits"]
    # stratify across docs
    by_doc = defaultdict(list)
    for h in hits:
        by_doc[h["_source"].get("document_title")].append(h)
    docs = sorted(by_doc); sample = []
    i = 0
    while len(sample) < N_IMAGES and any(by_doc[d] for d in docs):
        d = docs[i % len(docs)]
        if by_doc[d]:
            sample.append(by_doc[d].pop(0))
        i += 1

    gem = AsyncOpenAI(api_key=os.environ["GEMINI_API_KEY"],
                      base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    gateway = LLMGateway()
    kept = []
    for h in sample:
        if len(kept) >= KEEP_TARGET:
            break
        s = h["_source"]; meta = s.get("metadata") or {}
        path = meta.get("image_path"); title = s.get("document_title")
        if not path or not os.path.exists(path):
            continue
        # 1. draft Q + gold from the image
        try:
            r = await gem.chat.completions.create(
                model="gemini-2.5-flash", temperature=0.2, max_tokens=2048,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": GEN_SYS},
                          {"role": "user", "content": [
                              {"type": "text", "text": "Draft the question now."},
                              {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(path)}"}}]}])
            j = json.loads(r.choices[0].message.content or "{}")
            q, gold = j.get("question", "").strip(), j.get("answer", "").strip()
        except Exception:
            continue
        if not q or not gold:
            continue
        # 2. dependency filter — text model, no image
        ctx = await _neighbour_text(es, title, s.get("position"))
        try:
            # json_response returns (payload_dict, latency_ms)
            payload, _ = await gateway.json_response(
                system_prompt="Answer the question from the provided context only. Return JSON {\"answer\":\"...\"}.",
                user_prompt=f"CONTEXT:\n{ctx}\n\nQUESTION: {q}", task="eval_judge2")
            txt_answer = (payload or {}).get("answer", "") if isinstance(payload, dict) else str(payload)
        except Exception:
            txt_answer = ""
        e = await score_correctness(q, txt_answer, gold, ctx, gateway, task="eval_judge")
        if e.mean >= TEXT_ANSWERABLE_THRESHOLD:
            continue  # text-answerable → not image-dependent → discard
        kept.append({"id": f"vis-{len(kept)}", "question": q, "reference_answer": gold,
                     "gold_contexts": [s.get("content", "")], "lang": "vi",
                     "source": title, "image_path": path})
        print(f"  kept {len(kept)}/{KEEP_TARGET} (text_score={e.mean:.2f}) {title[:30]}")
    await es.close()
    with open(OUT, "w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"WROTE {len(kept)} image-dependent Qs -> {OUT}")


asyncio.run(main())
```

- [ ] **Step 2: Run the builder**

```bash
cd /home/nguyenquocdung/work/AgentRag && source .env
GEMINI_API_KEY="$GEMINI_API_KEY" PYTHONPATH=. uv run python scripts/eval/build_vision_evalset.py 2>&1 | tail -15
```
Expected: `WROTE <N> image-dependent Qs -> data/eval/vision_evalset_2026-07-19.jsonl` with N ≥ ~20. Each kept Q had text-only score < 0.6 (printed).

- [ ] **Step 3: Sanity-check the eval set**

```bash
wc -l data/eval/vision_evalset_2026-07-19.jsonl
python3 -c "import json;[print(json.loads(l)['question'][:80]) for l in open('data/eval/vision_evalset_2026-07-19.jsonl')][:5]"
```
Expected: N rows; questions reference visible image content.

- [ ] **Step 4: Commit the builder + eval set**

```bash
git add scripts/eval/build_vision_evalset.py data/eval/vision_evalset_2026-07-19.jsonl
git commit -m "feat(eval): synthesized image-dependent eval-Q set for slice C (text-answerable filtered)"
```

---

### Task 5: Phase 4 — e2e answer-time vision ON/OFF A/B + results doc

**Files:**
- Create: `scripts/eval/run_vision_e2e_ab.py`
- Output: `data/eval/vision_e2e_on.json`, `data/eval/vision_e2e_off.json`, `docs/eval/vision_e2e_2026-07-19.md`

**Interfaces:**
- Consumes: `data/eval/vision_evalset_2026-07-19.jsonl`; `load_local_jsonl`; `get_agent_service().chat`; `score_correctness`; `LLMGateway`.
- Produces: per-arm correctness json + a comparison doc. Arms are run as SEPARATE processes (settings is import-time singleton — `VISION_ANSWER_MODEL` must be set via env before import).

- [ ] **Step 1: Write the A/B runner (single-arm + compare modes)**

`scripts/eval/run_vision_e2e_ab.py`:
```python
"""Slice C Phase 4 — measure answer-time vision lift (ON vs OFF) on the image eval set.

Settings is an import-time singleton, so each arm runs as its own process with a
different VISION_ANSWER_MODEL env. Mode --arm runs one arm; mode --compare diffs two
per-arm json files into a markdown report.
"""
import argparse, asyncio, json, sys
sys.path.insert(0, ".")


async def _run_arm(eval_path, n, out_path):
    from src.agentrag.config import settings
    from src.agentrag.eval.benchmark_datasets import load_local_jsonl
    from src.agentrag.agent.factory import get_agent_service
    from src.agentrag.eval.correctness_judge import score_correctness
    from src.agentrag.services.llm_gateway import LLMGateway
    examples = load_local_jsonl(eval_path, n)
    agent, gateway = get_agent_service(), LLMGateway()
    rows = []
    for ex in examples:
        out = await agent.chat(question=ex.question, document_title=None,
                               conversation_id=f"vis-eval-{ex.id}")
        if out.get("timed_out"):
            rows.append({"id": ex.id, "score": None, "timed_out": True}); continue
        ans = out.get("answer", "") or ""
        e = await score_correctness(ex.question, ans, ex.reference_answer,
                                    "\n".join(ex.gold_contexts), gateway)
        rows.append({"id": ex.id, "question": ex.question, "answer": ans[:400],
                     "score": e.mean, "low_confidence": e.low_confidence})
        print(f"  {ex.id} score={e.mean:.2f}")
    scored = [r["score"] for r in rows if isinstance(r.get("score"), (int, float))]
    mean = sum(scored) / len(scored) if scored else 0.0
    report = {"vision_answer_model": settings.VISION_ANSWER_MODEL, "n": len(rows),
              "scored": len(scored), "mean_correctness": round(mean, 4), "rows": rows}
    json.dump(report, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"ARM vision_answer_model={settings.VISION_ANSWER_MODEL!r} MEAN={mean:.4f} -> {out_path}")


def _compare(on_path, off_path, out_md):
    on, off = json.load(open(on_path)), json.load(open(off_path))
    delta = on["mean_correctness"] - off["mean_correctness"]
    lines = ["# Vision Slice C — answer-time e2e A/B (2026-07-19)", "",
             f"Eval set: `data/eval/vision_evalset_2026-07-19.jsonl` (image-dependent Qs).", "",
             "| arm | VISION_ANSWER_MODEL | n | scored | mean correctness |",
             "|---|---|---|---|---|",
             f"| OFF | `{off['vision_answer_model']}` | {off['n']} | {off['scored']} | {off['mean_correctness']:.4f} |",
             f"| ON | `{on['vision_answer_model']}` | {on['n']} | {on['scored']} | {on['mean_correctness']:.4f} |",
             "", f"**Delta (ON − OFF) = {delta:+.4f}**", "",
             ("**Recommendation:** default-ON answer-time vision" if delta >= 0.03
              else "**Recommendation:** keep answer-time vision OFF (no material lift on this set)"), ""]
    open(out_md, "w").write("\n".join(lines))
    print(f"delta={delta:+.4f} -> {out_md}")


async def _amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["on", "off"])
    ap.add_argument("--eval", default="data/eval/vision_evalset_2026-07-19.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("ON_JSON", "OFF_JSON"))
    ap.add_argument("--out-md", default="docs/eval/vision_e2e_2026-07-19.md")
    args = ap.parse_args()
    if args.compare:
        _compare(args.compare[0], args.compare[1], args.out_md)
    else:
        await _run_arm(args.eval, args.n, args.out or f"data/eval/vision_e2e_{args.arm}.json")


asyncio.run(_amain())
```

- [ ] **Step 2: Run the OFF arm (answer-time vision disabled)**

```bash
cd /home/nguyenquocdung/work/AgentRag && source .env
VISION_ANSWER_MODEL="" GEMINI_API_KEY="$GEMINI_API_KEY" PYTHONPATH=. \
  uv run python scripts/eval/run_vision_e2e_ab.py --arm off --out data/eval/vision_e2e_off.json 2>&1 | tail -6
```
Expected: `ARM vision_answer_model='' MEAN=<x.xxxx> -> data/eval/vision_e2e_off.json`.

- [ ] **Step 3: Run the ON arm (answer-time vision = gemini)**

```bash
source .env
VISION_ANSWER_MODEL="gemini-2.5-flash" GEMINI_API_KEY="$GEMINI_API_KEY" PYTHONPATH=. \
  uv run python scripts/eval/run_vision_e2e_ab.py --arm on --out data/eval/vision_e2e_on.json 2>&1 | tail -6
```
Expected: `ARM vision_answer_model='gemini-2.5-flash' MEAN=<x.xxxx> -> data/eval/vision_e2e_on.json`.

- [ ] **Step 4: Build the comparison doc**

```bash
PYTHONPATH=. uv run python scripts/eval/run_vision_e2e_ab.py \
  --compare data/eval/vision_e2e_on.json data/eval/vision_e2e_off.json \
  --out-md docs/eval/vision_e2e_2026-07-19.md
cat docs/eval/vision_e2e_2026-07-19.md
```
Expected: a table with OFF vs ON mean correctness + `Delta (ON − OFF)` + a keep/default recommendation. Add 2–3 example win/loss rows (from the per-arm jsons) to the doc by hand.

- [ ] **Step 5: Commit Phase 4**

```bash
git add scripts/eval/run_vision_e2e_ab.py data/eval/vision_e2e_on.json data/eval/vision_e2e_off.json docs/eval/vision_e2e_2026-07-19.md
git commit -m "feat(eval): vision slice C phase 4 — answer-time ON/OFF e2e A/B + lift report"
```

---

## Post-plan

- Update issue #9 with the Phase-4 delta + keep/retire decision; close it if the feature is decided.
- The default-on flip of `VISION_ANSWER_MODEL` in committed config is a SEPARATE ops decision, informed by the Phase-4 delta — not part of this plan.
- Keep `agentrag_segments_backup_20260719` until the augmentation is accepted; drop it afterward (`curl -X DELETE localhost:9200/agentrag_segments_backup_20260719`).
