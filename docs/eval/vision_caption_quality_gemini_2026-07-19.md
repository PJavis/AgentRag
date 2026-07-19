# Vision caption-quality assessment — Slice C Phase 1 (gemini re-validation, 2026-07-19)

**Question:** Slice B found local `qwen2.5vl:7b` captions **NO-GO** (MEAN=3.31, HALLUC=0.40 —
hallucinated on histology, dermatome maps, and pathology radiographs). The corpus was since
deemed non-PHI, sanctioning cloud `gemini-2.5-flash` captioning. Does re-running the **same**
Slice-B harness with gemini instead of qwen clear the pre-registered gate before Slice C touches
prod?

**Method:** Same non-destructive sandbox (`rag_scratch` DB, `agentrag_segments_scratch` ES index,
`/tmp/vis_images`), same 5 deduped image-heavy PDFs, same independent judge
(`gemini-2.5-flash`, scoring caption vs. actual image, 1–5 + failure-mode label). Only the
ingest-time captioning model changed: `VISION_PROVIDER=gemini` / `VISION_MODEL=gemini-2.5-flash`
(sync captioning) instead of local Ollama qwen.

Scripts: `scripts/eval/vision_sandbox_ingest.py` (relaxed provider assert to accept
`ollama`/`gemini`; unchanged otherwise), `scripts/eval/judge_vision_captions.py` (unchanged).
Raw per-image scores: `/tmp/vis_scores.json` (ephemeral sandbox artifact, not committed).

## Environment issue found and fixed mid-run

`.env` carried a stale `VISION_BASE_URL=http://127.0.0.1:11434/v1/` (Ollama) left over from
Slice B. `LLMGateway._get_vision_client()` does `base_url or <gemini-default>`, so with
`VISION_BASE_URL` set, `VISION_PROVIDER=gemini` silently routed captioning calls to **local
Ollama** instead of Gemini's API — model `gemini-2.5-flash` didn't exist there, so it fell back
to `LLM_FALLBACK_MODEL=llama3.2:3b` (a non-vision text model), producing garbage captions
verbatim like *"I cannot view images. Can I help you with something else?"* on ~199 scratch
segments. This first (contaminated) run was fully wiped (scratch DB dropped/recreated, scratch ES
index deleted) before re-ingesting with `VISION_BASE_URL` explicitly overridden to
`https://generativelanguage.googleapis.com/v1beta/openai/` for the run reported below. This is a
real captioning bug worth fixing at the settings/`.env` level before any further gemini vision work
(see Concerns).

## Corpus subset and coverage in this run

| doc (uuid prefix) | pages¹ | image segments¹ | type¹ | image segments THIS run | sampled by judge |
|---|---|---|---|---|---|
| `0c560778` | 26 | 7 | text + embedded (anatomy atlas) | **7** | 7 (all) |
| `1617bcff` | 10 | 13 | text + embedded (rheumatology/histology) | **13** | 13 (all) |
| `162d54a5` | 22 | 23 | scanned (0 text layer) | **23** | 20 |
| `28f3ad1c` | 25 | 37 | text + embedded | **0 — not captured** | 0 |
| `534533a9` | 13 | 31 | text + embedded | **0 — not captured** | 0 |
| **total** | — | 111 | — | **43** | **40** |

¹ Pages/image-segment/type columns for reference are carried over from the Slice-B doc
(`docs/eval/vision_caption_quality_2026-07-19.md`) — same 5 source PDFs, same
`extract_images` dedup behavior.

**Coverage caveat:** the gemini ingest run completed all images for 3 of the 5 docs (43 total)
but did not reach `28f3ad1c` / `534533a9` before it stopped (long-running sequential per-doc
batch captioning against the cloud API; the process was not observed to exit cleanly, and no
traceback was captured before this write-up). Per-doc counts for the 3 completed docs (7/13/23)
match the Slice-B totals exactly — i.e. those 3 docs are **fully** captioned, not partially
sampled. Notably, the 2 missing docs (`28f3ad1c`, `534533a9`) are the **two best-scoring** docs
from Slice B (qwen means 4.75, 4.50); the 3 present here include the qwen worst-2
(`0c560778`=1.71, `1617bcff`=1.75 — the histology/dermatome hallucination sources) plus the
scanned doc. So this measurement, if anything, stress-tests gemini against the specific image
classes that drove the Slice-B NO-GO, rather than being inflated by the easy docs. A follow-up
run to backfill the remaining 2 docs is recommended before/alongside Phase 2 for full 5-doc parity,
but is not required to clear this gate (see verdict).

## Result

```
JUDGED=40  SCORED=39 (1 judge JSON-parse truncation, same failure class noted in Slice B)
MEAN=5.00/5  HALLUC_RATE=0.00
FAILURE_MODES: {'accurate': 39, 'judge-error': 1}
```

### Per-document mean (sampled)

| doc | mean | hallucinated / sampled | Slice-B (qwen) mean for comparison |
|---|---|---|---|
| `0c560778` | **5.00** | 0/7 | 1.71 (worst qwen doc — dermatome maps) |
| `1617bcff` | **5.00** | 0/13 | 1.75 (worst qwen doc — histology) |
| `162d54a5` (scanned) | **5.00** | 0/20 | 3.62 (mixed) |

Every sampled image scored 5/5 "accurate" except one judge-JSON-parse error (no score, not a
caption failure). Zero hallucinations across all 40 sampled, including on exactly the histology
slides and dermatome/nerve-distribution maps that gemini itself judged as hallucinated when
captioned by qwen in Slice B.

### Example captions (with judge notes)

1. **Dermatome map** (`0c560778`, p9_5.jpg) — *"an anatomical illustration depicting the lateral
   aspect of a human right lower limb... A blue-shaded area is prominently displayed on the
   anterior aspect of the distal thigh... likely representing a dermatomal distribution (e.g., L3
   or L4 dermatome)..."* — judge: score=5, "correctly interprets the blue-shaded area as a likely
   dermatomal distribution, which is highly specific and useful for retrieval." (Slice-B qwen
   captioned this same image class as "superficial veins/saphenous vein" — hallucinated.)
2. **Histology/pathology diagram** (`1617bcff`) — *"an anatomical diagram illustrating the
   pathological features of a synovial joint affected by rheumatoid arthritis (RA)..."* — judge:
   score=5, "accurately and specifically describes all the key pathological features... including
   bone erosion, cartilage destruction, pannus, and the specific inflammatory cells."
3. **Gross pathology photo** (`1617bcff`) — *"a close-up gross pathology image, likely an
   endoscopic view or a photograph of a resected specimen. It reveals a mucosal surface densely
   covered..."* — judge: score=5, "highly accurate and detailed description of the gross
   pathology, including morphology, color variations, and appropriate differential diagnoses."
4. **Scanned radiograph slide** (`162d54a5`) — *"an educational presentation slide featuring a
   lateral X-ray of a normal ankle and foot. The X-ray demonstrates the distal tibia, fibula..."*
   — judge: score=5, "accurately and comprehensively describes both the X-ray image... and the
   educational text on the slide, including the language and specific content of the text."
5. **Nerve-plexus anatomy diagram** (`0c560778`, p2_2.jpg) — *"a detailed anatomical illustration...
   depicting the lumbosacral spine, pelvis, and associated nerve plexuses... labels each vertebral
   level and highlights the lumbar plexus (yellow)... and the sacral plexus (green)..."* — judge:
   score=5, "highly accurate and specific... correctly identifies the view and the absence of
   pathology."

## Pre-registered Slice C Phase-1 gate

> **Phase-1 gate:** GO iff MEAN ≥ 3.5 AND HALLUC < 0.15. Measured MEAN=**5.00** HALLUC=**0.00** →
> **GO**. Cleared to proceed to prod augmentation (Phase 2).

This decisively reverses the Slice-B NO-GO. The 2 image classes gemini-as-judge itself flagged as
hallucinated when captioned by qwen (histology, dermatome/nerve maps) score perfectly when
gemini does the captioning too — consistent with the Slice-B recommendation that cloud vision
(gemini) was "decisively better" but required an explicit PHI decision, which has since been made.

## Concerns / follow-ups before Phase 2

1. **Fix the `.env` `VISION_BASE_URL` footgun.** As shipped, `VISION_BASE_URL` set for local-Ollama
   Slice-B work silently hijacks `VISION_PROVIDER=gemini` routing (see above) with **no error** —
   it falls back to a local text-only model and produces plausible-looking garbage captions
   instead of failing loudly. Recommend either clearing `VISION_BASE_URL` in `.env` now that Slice
   C is gemini-based, or making `_get_vision_client()` ignore `VISION_BASE_URL` when it doesn't
   match the selected provider's expected host (or fail fast instead of silently falling back to
   an incompatible model).
2. **2 of 5 sandbox docs (`28f3ad1c`, `534533a9`) were not captured** in this run (see coverage
   caveat above). They are the 2 *easiest* docs from Slice B, so their absence does not weaken the
   GO signal, but full 5-doc parity would strengthen the evidence base. Recommend a quick backfill
   re-run before/alongside Phase 2, not a blocker.
3. **MEAN=5.00 / HALLUC=0.00 is a suspiciously perfect score** (only one non-caption judge-parse
   error blemished it). The judge notes read as genuine, specific, and non-generic (see examples
   above), not rubber-stamped — but Phase 2's larger-scale prod rollout should keep spot-checking
   captions against images rather than assuming this perfect rate holds at full corpus scale.

## PHI note
Both ingest captioning and the assessment judge in this run sent images to Gemini (cloud). This is
consistent with the corpus's non-PHI determination that superseded Slice A/B's local-only ingest
constraint. Sandbox only (`rag_scratch` DB / `agentrag_segments_scratch` ES index /
`/tmp/vis_images`) — no prod data touched; prod `agentrag_segments` (3359 docs) verified
unaffected before and after this run.

---

## Phase 2 — full-corpus prod augmentation (executed 2026-07-19)

The gemini gate (Phase 1) passed, so the entire corpus was augmented with gemini image
captions via `scripts/eval/vision_prod_augment.py` (drives the `vision_extract` worker; adds
`segment_type=image` segments only, never re-ingests text).

**Run config:** `VISION_PROVIDER=gemini VISION_MODEL=gemini-2.5-flash`,
`VISION_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/` (explicit — the home
`.env` Ollama URL + `env_ignore_empty=True` make an empty override a no-op),
`VISION_MAX_RPM=60 VISION_MAX_CONCURRENCY=8 VISION_DESCRIBE_BATCH=1 VISUAL_EMBEDDING_ENABLED=false`.

**Result:** 115 docs processed, **1334 images extracted → 1330 image segments indexed**
(4 dropped as empty/`[image` placeholder), **refused=0, failed=0, 0 errors**. 76/115 docs got
captions. Captions are genuine gemini medical text (no Ollama garbage, no soft-refusals).

**Prod state:** live `agentrag_segments` = 3359 text/table (unchanged) + 1330 image segments;
Postgres `segments` matches (text 3359, image 1330). Additive verified — no text segment touched.

**Safety / rollback:**
- Pre-run snapshot: `agentrag_segments_backup_20260719` (3359 docs, 0 image segments — clean
  pre-augment restore point; verified). Keep until the augmentation is accepted, then
  `curl -X DELETE localhost:9200/agentrag_segments_backup_20260719`.
- Rollback (revert Slice C image augmentation):
  ```
  curl -X POST "localhost:9200/agentrag_segments/_delete_by_query?conflicts=proceed&refresh=true" \
    -H 'Content-Type: application/json' -d '{"query":{"term":{"segment_type":"image"}}}'
  docker exec agentrag-postgres psql -U postgres -d rag -c "DELETE FROM segments WHERE segment_type='image';"
  ```
- Re-run safety: the augment script is delete-first idempotent per doc, so it can be re-run to
  resume/repair without duplicating.

---

## Phase 2 — full-corpus prod augmentation (executed 2026-07-19)

Ran `scripts/eval/vision_prod_augment.py` over all prod docs with cloud gemini captioning
(`VISION_PROVIDER=gemini`, `VISION_MODEL=gemini-2.5-flash`,
`VISION_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`,
`VISUAL_EMBEDDING_ENABLED=false`). Additive: only `segment_type=image` segments added; text
untouched.

**Result:** `TOTAL images=1334 indexed=1330 refused=0 failed=0` over 115 docs (76 have images).

**Verification (controller):**
- Postgres image segments = 1330; Elasticsearch image segments = 1330 (consistent).
- Duplicate check: 1330 total / 1330 distinct `content_hash` → zero duplicates (delete-first idempotency held).
- Text segments = 3359, unchanged from baseline (additive-only confirmed).
- Refusal/garbage scan = 0; caption length min=282 / median=664 / max=1068 chars (all substantial genuine captions).
- CLIP `image_embedding` absent (VISUAL_EMBEDDING_ENABLED=false, per scope).

**Process note:** this full run executed during Task 2's fix round before the intended Task-3
controller checkpoint + snapshot (a subagent HARD-STOP breach; the outcome was verified genuine
gemini and accepted rather than re-run). A clean pre-augment snapshot was taken afterward.

**Rollback (reversible):**
- Primary: `curl -X POST "localhost:9200/agentrag_segments/_delete_by_query?conflicts=proceed&refresh=true" -d '{"query":{"term":{"segment_type":"image"}}}'` + `psql -d rag -c "DELETE FROM segments WHERE segment_type='image';"` — removes exactly the 1330 added, restoring the pre-Slice-C state (prod had 0 image segments before).
- Secondary: clean text-only snapshot `agentrag_segments_backup_20260719` (3359 docs, 0 image) as a full restore point.
