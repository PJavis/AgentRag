# Vision Slice B — local caption-quality assessment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether local `qwen2.5-vl:7b` produces medically-useful captions of this corpus's images (Gemini-judged) to gate the full-corpus re-ingest (Slice C).

**Architecture:** Non-destructive sandbox (scratch Postgres DB + parallel ES index) ingest of 5 deduped image-heavy PDFs with `VISION_PROVIDER=ollama` sync captioning; then an independent Gemini judge scores each caption against its image; aggregate → pre-registered verdict. Controller-run measurement (no prod code change); deliverables are a results doc + reusable scripts.

**Tech Stack:** Ollama (qwen2.5-vl:7b), the project's `ingest_folder` pipeline, Elasticsearch (parallel index), Gemini (`gemini-2.5-flash`) via the generativelanguage OpenAI-compat endpoint, Python/uv.

## Global Constraints

- Env overrides MUST be exported BEFORE the Python process starts (settings is a singleton read at first import): `POSTGRES_DB=rag_scratch`, `ELASTICSEARCH_INDEX_NAME=agentrag_segments_scratch`, `TAGGING_ENABLED=false`, `VISION_PROVIDER=ollama`, `VISION_MODEL=qwen2.5-vl:7b`, `VISION_INGEST_MODE=sync`. Leave `STRUCTMEM_INGEST_MODE=async` (do NOT set sync; do NOT `init_pool`).
- Non-destructive: touch ONLY `rag_scratch` DB + `agentrag_segments_scratch` index. Never the live `rag` / `agentrag_segments`.
- PHI: ingest captioning is 100% local (Ollama). Only the judge (Task 2) sends sampled images to Gemini — bounded (~≤40), one-time, assessment-only.
- Pre-registered Slice C verdict: **GO** iff mean caption score ≥ 3.5/5 AND hallucinated-finding rate < 15%; else **NO-GO**.
- Ollama dies on this WSL box — verify `curl -s localhost:11434/api/tags` before each ollama-dependent step; `nohup ollama serve &` to restart.

---

### Task 1: Sandbox ingest with local qwen2.5-vl captioning

**Files:**
- Create (scratch, gitignored/tmp): a 5-PDF folder; `/tmp/vis_ingest.py`

**Interfaces:**
- Produces: `agentrag_segments_scratch` ES index containing `segment_type=image` segments (`content`=caption, `extra_metadata.image_path`/`image_url`); prints image-segment count.

- [ ] **Step 1: Pull the vision model + verify ollama**

```bash
curl -s -m5 localhost:11434/api/tags >/dev/null || (nohup ollama serve >/tmp/ollama.log 2>&1 & sleep 5)
ollama pull qwen2.5-vl:7b            # ~6GB, one-time
ollama show qwen2.5-vl:7b | head -3  # confirm present
```
Expected: model listed (no "not found").

- [ ] **Step 2: Create the scratch DB + staging folder**

```bash
source .env
PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 -p 5433 -U postgres -c "DROP DATABASE IF EXISTS rag_scratch;" -c "CREATE DATABASE rag_scratch;"
mkdir -p /tmp/vis_docs && cd /home/nguyenquocdung/work/AgentRag
for f in 0c560778-9ed1-428a-ac4e-0c4c900f2e4e 28f3ad1c-faf0-4d51-9af4-da45d0b22069 534533a9-55eb-47ad-a762-b10435291892 1617bcff-9bf1-41de-aa8d-9b5fcfc5f78e 162d54a5-eeac-4454-8ecb-ffdfef710dec; do cp "data/originals/$f.pdf" /tmp/vis_docs/; done
ls /tmp/vis_docs | wc -l   # expect 5
```

- [ ] **Step 3: Write the sandbox ingest script**

`/tmp/vis_ingest.py`:
```python
import asyncio, sys
sys.path.insert(0, ".")
from src.agentrag.config import settings
from src.agentrag.database import engine, Base
from src.agentrag.ingestion.pipeline import ingest_folder

async def main():
    assert settings.POSTGRES_DB == "rag_scratch", f"NOT scratch DB: {settings.POSTGRES_DB}"
    assert settings.ELASTICSEARCH_INDEX_NAME == "agentrag_segments_scratch", settings.ELASTICSEARCH_INDEX_NAME
    assert settings.VISION_PROVIDER == "ollama" and settings.VISION_INGEST_MODE == "sync"
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)   # scratch schema (TAGGING off → no ontology_terms need)
    rep = await ingest_folder("/tmp/vis_docs")
    print("INGEST:", {k: rep.get(k) for k in ("status", "ingested", "total")} if isinstance(rep, dict) else rep)

asyncio.run(main())
```

- [ ] **Step 4: Run the ingest (env-prefixed; captioning is local)**

```bash
POSTGRES_DB=rag_scratch ELASTICSEARCH_INDEX_NAME=agentrag_segments_scratch TAGGING_ENABLED=false \
VISION_PROVIDER=ollama VISION_MODEL=qwen2.5-vl:7b VISION_INGEST_MODE=sync \
PYTHONPATH=. nohup uv run python /tmp/vis_ingest.py > /tmp/vis_ingest.log 2>&1 &
```
Then wait for completion (image captioning is slow — inline, one image at a time on the local GPU; 5 image-heavy docs may take 20–60+ min). Poll `/tmp/vis_ingest.log` for `INGEST:`.
Expected: `INGEST: {status: ..., ingested: 5, ...}`, no traceback. (Ollama may die mid-run → restart + re-run; ingest upserts by content_hash so re-running is safe.)

- [ ] **Step 5: Verify image caption segments landed + zero cloud calls**

```bash
uv run python -c "
import asyncio
from elasticsearch import AsyncElasticsearch
from src.agentrag.config import settings
async def main():
    es=AsyncElasticsearch(settings.ELASTICSEARCH_URL)
    n=(await es.count(index='agentrag_segments_scratch', query={'term':{'segment_type':'image'}}))['count']
    tot=(await es.count(index='agentrag_segments_scratch'))['count']
    print(f'image segments={n} total segments={tot}')
    res=await es.search(index='agentrag_segments_scratch', size=3, query={'term':{'segment_type':'image'}}, _source=['content','extra_metadata'])
    for h in res['hits']['hits']:
        s=h['_source']; print('CAPTION:', (s.get('content') or '')[:120], '| meta_keys:', list((s.get('extra_metadata') or {}).keys()))
    await es.close()
asyncio.run(main())
"
grep -ic "generativelanguage\|api.deepseek\|googleapis" /tmp/vis_ingest.log   # expect 0 — captioning was local only
```
Expected: `image segments=<N>` with N ≥ ~20; captions are real medical text (not `[image …]`); the cloud-call grep prints `0`. If N=0, STOP — captioning silently dropped (check `/tmp/vis_ingest.log` for vision errors / ollama down).

- [ ] **Step 6: Commit the ingest script (if worth keeping)**

```bash
mkdir -p scripts/eval && cp /tmp/vis_ingest.py scripts/eval/vision_sandbox_ingest.py
git add scripts/eval/vision_sandbox_ingest.py
git commit -m "feat(vision): sandbox ingest script for caption-quality assessment (slice B)"
```

---

### Task 2: Gemini caption-quality judge

**Files:**
- Create: `scripts/eval/judge_vision_captions.py`
- Output: `/tmp/vis_scores.json`

**Interfaces:**
- Consumes: `agentrag_segments_scratch` image segments (Task 1); `GEMINI_API_KEY`.
- Produces: `/tmp/vis_scores.json` = list of `{doc, image_path, caption, score:1-5, failure_mode, note}`; prints aggregate.

- [ ] **Step 1: Confirm Gemini vision reachable**

```bash
source .env
curl -s -m20 "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions" -H "Authorization: Bearer $GEMINI_API_KEY" -H 'Content-Type: application/json' -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":[{"type":"text","text":"OK"}]}],"max_tokens":5}' -o /dev/null -w "http=%{http_code}\n"
```
Expected: `http=200`.

- [ ] **Step 2: Write the judge script**

`scripts/eval/judge_vision_captions.py`:
```python
"""Judge sandbox image captions (qwen2.5-vl) against the actual images via Gemini.
Reads segment_type=image segments from agentrag_segments_scratch, sends each image +
its caption to gemini-2.5-flash for a 1-5 medical-accuracy score + failure mode."""
import asyncio, base64, json, os, sys
sys.path.insert(0, ".")
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI
from src.agentrag.config import settings

MAX_IMAGES = 40
JUDGE_SYS = (
  "You grade how accurately and usefully a CAPTION describes a MEDICAL image for retrieval. "
  "Return JSON: {\"score\": 1-5 (1=wrong/hallucinated, 5=accurate+specific), "
  "\"failure_mode\": one of [accurate, generic-uninformative, missed-key-finding, "
  "hallucinated-finding, ocr-of-text-slide, unreadable], \"note\": \"<short>\"}."
)

def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

async def main():
    es = AsyncElasticsearch(settings.ELASTICSEARCH_URL)
    res = await es.search(index="agentrag_segments_scratch", size=MAX_IMAGES,
                          query={"term": {"segment_type": "image"}},
                          _source=["content", "extra_metadata", "document_title"])
    await es.close()
    gem = AsyncOpenAI(api_key=os.environ["GEMINI_API_KEY"],
                      base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    out = []
    for h in res["hits"]["hits"]:
        s = h["_source"]; meta = s.get("extra_metadata") or s.get("metadata") or {}
        # image_path = local disk path; if only a serving image_url ("/images/..") is present,
        # map it under settings.IMAGE_STORAGE_DIR. Confirm the real field from Task 1 Step 5 output.
        path = meta.get("image_path")
        if not path and (meta.get("image_url") or "").startswith("/images/"):
            path = os.path.join(getattr(settings, "IMAGE_STORAGE_DIR", "data/images"), meta["image_url"][len("/images/"):])
        caption = s.get("content") or ""
        if not path or not os.path.exists(path):
            out.append({"doc": s.get("document_title"), "image_path": path, "caption": caption[:200],
                        "score": None, "failure_mode": "image-file-missing", "note": "no local image"})
            continue
        try:
            r = await gem.chat.completions.create(
                model="gemini-2.5-flash", temperature=0.0, max_tokens=200,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": JUDGE_SYS},
                          {"role": "user", "content": [
                              {"type": "text", "text": f"CAPTION: {caption[:1500]}"},
                              {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(path)}"}}]}])
            j = json.loads(r.choices[0].message.content)
        except Exception as e:
            j = {"score": None, "failure_mode": "judge-error", "note": str(e)[:120]}
        out.append({"doc": s.get("document_title"), "image_path": path,
                    "caption": caption[:200], **j})
    json.dump(out, open("/tmp/vis_scores.json", "w"), ensure_ascii=False, indent=2)
    scored = [o for o in out if isinstance(o.get("score"), (int, float))]
    from collections import Counter
    fm = Counter(o.get("failure_mode") for o in out)
    mean = sum(o["score"] for o in scored) / len(scored) if scored else 0
    halluc = sum(1 for o in out if o.get("failure_mode") == "hallucinated-finding") / (len(out) or 1)
    print(f"JUDGED={len(out)} SCORED={len(scored)} MEAN={mean:.2f} HALLUC_RATE={halluc:.2f}")
    print("FAILURE_MODES:", dict(fm))

asyncio.run(main())
```

- [ ] **Step 3: Run the judge**

```bash
source .env
ELASTICSEARCH_INDEX_NAME=agentrag_segments_scratch GEMINI_API_KEY="$GEMINI_API_KEY" PYTHONPATH=. \
  uv run python scripts/eval/judge_vision_captions.py
```
Expected: `JUDGED=<N> SCORED=<M> MEAN=<x.xx> HALLUC_RATE=<0.xx>` + failure-mode tally; `/tmp/vis_scores.json` written. Report the numbers.

- [ ] **Step 4: Commit the judge script**

```bash
git add scripts/eval/judge_vision_captions.py
git commit -m "feat(vision): gemini caption-quality judge for slice B assessment"
```

---

### Task 3: Results doc + Slice C verdict + sandbox teardown

**Files:**
- Create: `docs/eval/vision_caption_quality_2026-07-19.md`

- [ ] **Step 1: Write the results doc**

Create `docs/eval/vision_caption_quality_2026-07-19.md` with: the 5-doc subset (+ which are scanned vs text+embedded), image-segment count, MEAN + HALLUC_RATE + failure-mode tally from Task 2, a scanned-vs-embedded score split (from `/tmp/vis_scores.json`, grouped by doc), 3–5 example captions (good + bad), and the pre-registered verdict:
> **Slice C verdict:** GO iff MEAN ≥ 3.5 AND HALLUC_RATE < 0.15. Measured: MEAN=⟨…⟩, HALLUC=⟨…⟩ → **⟨GO | NO-GO⟩**. ⟨If NO-GO: which failure modes dominate + recommended alternative model.⟩

- [ ] **Step 2: Commit the results doc**

```bash
git add docs/eval/vision_caption_quality_2026-07-19.md
git commit -m "docs(eval): vision caption-quality assessment + Slice C verdict"
```

- [ ] **Step 3: Teardown the sandbox (non-destructive to prod)**

```bash
source .env
PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 -p 5433 -U postgres -c "DROP DATABASE IF EXISTS rag_scratch;"
curl -s -X DELETE "localhost:9200/agentrag_segments_scratch" >/dev/null && echo "dropped scratch index"
rm -rf /tmp/vis_docs
```
Expected: scratch DB + index gone; live `rag` / `agentrag_segments` untouched (verify: `curl -s localhost:9200/_cat/indices | grep agentrag_segments`).
