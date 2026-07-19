# Vision caption-quality assessment — Slice B (2026-07-19)

**Question:** Does local `qwen2.5vl:7b` (Ollama) produce medically-useful captions of this
corpus's images — good enough to justify the expensive full-corpus vision re-ingest (Slice C)?

**Method:** Non-destructive sandbox ingest of 5 deduped image-heavy PDFs with
`VISION_PROVIDER=ollama` / `VISION_MODEL=qwen2.5vl:7b`, sync captioning (100% local). Then an
independent judge (`gemini-2.5-flash`) scored each caption **against the actual image** (1–5
medical accuracy/usefulness + a failure-mode label). Sample = 40 image segments, stratified
round-robin across the 5 docs. Ingest made **zero cloud calls** (captioning local); only the
judge sent the bounded 40-image sample to Gemini (assessment-only, one-time).

Scripts: `scripts/eval/vision_sandbox_ingest.py`, `scripts/eval/judge_vision_captions.py`.
Raw per-image scores: `/tmp/vis_scores.json` (ephemeral sandbox artifact).

## Corpus subset

| doc (uuid prefix) | pages | image segments¹ | type | sampled |
|---|---|---|---|---|
| `0c560778` | 26 | 7 | text + embedded (anatomy atlas) | 7 (all) |
| `28f3ad1c` | 25 | 37 | text + embedded | 8 |
| `534533a9` | 13 | 31 | text + embedded | 8 |
| `1617bcff` | 10 | 13 | text + embedded (rheumatology/histology) | 9 |
| `162d54a5` | 22 | 23 | **scanned** (0 text layer) | 8 |
| **total** | — | **111** | — | **40** |

¹ `extract_images` byte-hash-dedups + filters sub-5KB icons, so ingested unique-image counts run
far below raw PDF xref counts (e.g. `0c560778` had ~161 raw xrefs → **7** unique images).

## Result

```
JUDGED=40  SCORED=39 (1 residual judge JSON-truncation)  MEAN=3.31/5  HALLUC_RATE=0.40
FAILURE_MODES: hallucinated-finding=16, accurate=21, missed-key-finding=1,
               generic-uninformative=1, judge-error=1
```

### Per-document mean (sampled)

| doc | mean | hallucinated / sampled | verdict signal |
|---|---|---|---|
| `28f3ad1c` | **4.75** | 1/8 | good |
| `534533a9` | **4.50** | 1/8 | good |
| `162d54a5` (scanned) | 3.62 | 2/8 | mixed |
| `1617bcff` | **1.75** | 6/9 | bad (histology) |
| `0c560778` | **1.71** | 6/7 | bad (dermatome maps) |

**Strongly bimodal by image *type*, not by scanned-vs-embedded.** The scanned doc (`162d54a5`)
did *not* score worst — qwen handled its plain radiographs adequately and only hallucinated on
pathology films (see below). The worst docs are text+embedded ones dominated by **histology
slides** and **color-coded dermatome/nerve maps**.

### Caveat: high scores are inflated by non-diagnostic images

Many `accurate=5` captions are qwen correctly flagging **non-diagnostic** decorative images
extracted from the PDFs — blank backgrounds, color gradients, institutional logos, a stock
smartphone render. Examples:
- *"a plain, uniform blue gradient with no discernible anatomical structures…"* (correct, 5)
- *"not a medical image but rather … the emblem of the University of Medicine and Pharmacy,
  Hanoi"* (correct, 5)
- *"an X-ray of the foot, lateral view … calcaneus, talus, first metatarsal"* (correct, 5)

So on the **clinically-meaningful** subset the effective quality is worse than 3.31 suggests:
qwen scores easy points on trivial images and fails on the diagnostic ones.

## Failure modes (qwen2.5vl:7b)

Confident, specific, **wrong** — the retrieval-poisoning case. Judge notes cite on-image
evidence qwen misread:

1. **Histology / H&E slides → invented tissue identity.** `1617bcff` synovial-biopsy slides
   captioned as *"gastrointestinal tract … mucosa, submucosa, goblet cells"* and *"cutaneous
   lesion/tumor"*. 6/9 hallucinated.
2. **Dermatome / nerve-distribution maps → read as vasculature/pressure/bone.** `0c560778`
   color-coded dermatome maps captioned as *"superficial veins / saphenous vein"*, *"pressure
   distribution"*, *"bones and soft tissues"*. 6/7 hallucinated.
3. **Pathology radiographs → wrong region + missed findings.** `162d54a5` scanned films:
   cervical spine called *lumbar*; a wrist X-ray called *lumbar spine + MRI*; *"no pathological
   findings"* stated over clear fractures/dislocations.
4. **Disease mislabel against on-image text.** An image labeled *"RA joint"* (rheumatoid)
   captioned as *osteoarthritis*.
5. **Vietnamese on-image text mistranslated.** *"Biểu mô màng hoạt dịch"* (synovial-membrane
   epithelium) rendered as *"exudate membrane morphology"* — weak Vietnamese OCR on a Vietnamese
   medical corpus.

## Pre-registered Slice C verdict

> **GO iff MEAN ≥ 3.5 AND hallucinated-finding rate < 0.15.**
> Measured: **MEAN = 3.31**, **HALLUC = 0.40** → **NO-GO.**

**Weighting-robustness check.** Stratified sampling ~equal-weights the 5 docs, over-weighting the
two small bad docs (7, 13 imgs) relative to their corpus share. A corpus-frequency-weighted mean
is ≈ **3.90** (would pass the mean gate) — **but the hallucination gate fails under every
weighting**: equal-weight 40%, corpus-frequency-weighted ≈ 26%, and even discounting the 2
mild score-3 "hallucinations" ≈ 35%. All ≫ 15%. **The NO-GO is driven by hallucination, not by
the mean, and does not depend on the sampling scheme.**

For a medical RAG this is the decisive axis: a fluent, confident caption that misidentifies
histology or a dermatome map gets embedded and retrieved **as fact**, actively poisoning
answers — worse than no caption at all.

## Recommendation (before spending Slice C)

Do **not** re-ingest the full corpus with `qwen2.5vl:7b`. Options, cheapest first:

1. **Gate vision ingest by image type.** qwen is adequate on plain radiographs/photos and
   reliably rejects non-diagnostic images. It is dangerous on histology, dermatome/schematic
   diagrams, and Vietnamese-annotated figures. A classifier (or a cheap first-pass "is this a
   diagnostic photo/radiograph?") that only captions the safe classes would salvage most value.
2. **Try a larger local VL model** — `qwen2.5vl:32b` / `:72b` (VRAM permitting on the 16 GB
   RTX 5060 Ti — 72b won't fit; 32b quantized might) — and re-run this exact harness before Slice C.
3. **Cloud vision for ingest** (`gemini-2.5-flash`, which judged accurately here) — decisively
   better, but breaks the PHI-local-ingest guarantee; only with an explicit PHI decision.
4. **Add an OCR path** for text-slide / annotated-figure images instead of free-form captioning.

**Re-run gate after any model/pipeline change:**
`scripts/eval/vision_sandbox_ingest.py` + `scripts/eval/judge_vision_captions.py` reproduce this
measurement end-to-end; the pre-registered thresholds above stay fixed.

## PHI note
Ingest captioning was 100% local (Ollama, 0 cloud calls verified in the ingest log). Only the
assessment judge sent the 40-image sample to Gemini — bounded, one-time, assessment-only, not a
production path. Consistent with the Slice-A mixed-provider decision.
