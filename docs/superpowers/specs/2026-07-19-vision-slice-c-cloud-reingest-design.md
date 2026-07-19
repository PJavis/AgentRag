# Vision Slice C — cloud-vision full-corpus re-ingest + synthesized image eval (2026-07-19)

## Why

Slices A+B (merged, PR #8) built answer-time vision routing and a caption-quality gate. The gate
returned **NO-GO for local `qwen2.5vl:7b`** (hallucinates on histology / dermatome maps /
pathology radiographs / Vietnamese on-image text). Decision (2026-07-19): the corpus is **not
PHI-sensitive**, so **cloud captioning is sanctioned**. Slice C turns vision on for real using
`gemini-2.5-flash` (which judged accurately in Slice B), lands captions in the retrieval index,
and **measures the answer-quality lift** end-to-end. Tracked as issue #9.

## Goal / success criteria

Vision captions live in production retrieval, with a measured e2e delta. Done when:

1. **Gate re-validated** with gemini captions on the Slice-B sandbox: **mean ≥ 3.5 AND
   hallucinated < 0.15** (same pre-registered thresholds). GO required to proceed.
2. **All 114 prod docs augmented** with `segment_type=image` segments (gemini captions) in the
   live `agentrag_segments` index — additive, text segments untouched, reversible.
3. **A synthesized image-dependent eval-Q set** (~30–50 Qs) exists at
   `data/eval/vision_evalset_*.jsonl`, each question demonstrably answerable from the image but
   NOT from surrounding text.
4. **An e2e measurement** reports answer correctness with answer-time vision ON vs OFF on that
   set — the number that justifies (or retires) the feature.

## Non-goals

- CLIP visual-similarity retrieval (`image_embedding` exists in the schema; leave it off unless
  free — caption-text retrieval is the scope).
- Flipping `VISION_PROVIDER` on by default in committed config (the re-ingest run uses env; a
  default flip is a separate ops decision).
- Human-authored gold answers (chosen: fully LLM-synthesized eval, with an automated
  image-dependency filter for teeth).

## Key discovery facts (drive the design)

- **ES `index_segments` assigns `_id = uuid.uuid4()`** (`elasticsearch_store.py:331`), NOT a
  content hash. A naive full re-ingest therefore **duplicates every existing text segment**
  (3,359 today). Phase 2 MUST be image-only augmentation, not a whole-corpus re-ingest.
- **A purpose-built async worker already does exactly this augmentation:**
  `src/agentrag/graph/vision_jobs.py` → `process_vision_job(VisionExtractJob(document_id, title,
  image_records=[{path,page,mime,url}]))`. It reads images from disk, batch-describes via the
  vision LLM, builds `segment_type=image` segments, and indexes them into ES (+ PG). It has a
  **token-bucket RPM cap** (`_RpmBucket`) and **transient-retry on 429 / RESOURCE_EXHAUSTED /
  503** — i.e. it is already gemini-rate-limit aware. Phase 2 drives THIS worker, it does not
  re-run the text pipeline.
- **Image records source:** `image_records` (`{path,page,mime,url}`) come from
  `PDFParser.extract_images(pdf_path, title)` (byte-dedups + filters sub-5KB icons; writes to
  `IMAGE_STORAGE_DIR/<slug(title)>/`). Corpus ≈ **1,334 images across 114 docs** (disk estimate).
- **Slice A answer-time vision** (`VISION_ANSWER_MODEL`) is on master, default OFF; Phase 4 flips
  it on via env (`gemini-2.5-flash`) to measure the lift.
- **Requires a paid/quota gemini key** — ~1,334 captions; tune `_RpmBucket` to the key's real
  RPM. Confirm the key before Phase 2.

## Design

### Phase 1 — Gate re-validation (cheap insurance, no prod impact)
Re-run the Slice-B harness on the same 5 deduped PDFs, `VISION_PROVIDER=gemini
VISION_MODEL=gemini-2.5-flash`, into the scratch DB + `agentrag_segments_scratch` index (same
non-destructive sandbox). Judge with `scripts/eval/judge_vision_captions.py` (unchanged
thresholds). Expected mean ≫ 3.5, halluc ≈ 0 (gemini was the accurate judge in Slice B). NO-GO →
STOP and reconsider model. Deliverable: a new
`docs/eval/vision_caption_quality_gemini_2026-07-19.md`.

### Phase 2 — Full-corpus image augmentation (additive, reversible)
1. **Snapshot** the live ES index first (ES snapshot API, or `_reindex` to a dated backup index)
   — the rollback safety net.
2. For each of the 114 prod docs (from the `Document` table): `extract_images(data/originals/<id>.pdf,
   title)` → build `VisionExtractJob` → `process_vision_job` with `VISION_PROVIDER=gemini`, RPM
   bucket tuned to the paid key. A controller script drives this sequentially/concurrently under
   the worker's own rate cap. Captioning is cloud (gemini); text/graph untouched.
3. **Idempotency / re-run safety:** verify whether `process_vision_job` dedups image segments on
   re-run (plan-time check); regardless, rollback = **one `delete_by_query {segment_type:image}`**
   on the live index restores the pre-Slice-C state exactly (there are zero image segments today).
4. Verify: image-segment count > 0 per doc, captions are real gemini text, retrieval still serves
   text normally.

### Phase 3 — Synthesized image-dependent eval-Q set (~30–50 Qs)
1. **Generate:** for a stratified sample of augmented images, gemini (vision) drafts a question +
   gold answer that requires *reading the image*.
2. **Image-dependency filter (the teeth):** give a **text-only** model the doc's surrounding text
   (the image's neighboring text segments) WITHOUT the image, and the question. If it answers
   correctly → **discard** (text-answerable, no vision lift to measure). Keep only questions the
   text-only model fails. This makes even a synthesized set discriminative.
3. **Circularity mitigation:** diversify models across roles — generation (gemini vision) vs the
   dependency-filter answerer (the project's text model, deepseek) vs the Phase-4 judge (gemini) —
   so no single model both writes and grades. Store `data/eval/vision_evalset_2026-07-19.jsonl`
   ({question, gold, doc, image_path, why_image_dependent}).

### Phase 4 — E2e measurement (the deliverable number)
Run the agent over the Phase-3 set twice: answer-time vision **ON** (`VISION_ANSWER_MODEL=
gemini-2.5-flash`) vs **OFF** (`""`, text-only fallback). Grade both with the existing LLM-judge
harness. Report correctness delta (ON − OFF) = vision's contribution, plus abstain/refusal
changes. Deliverable: `docs/eval/vision_e2e_2026-07-19.md` with the A/B, example wins/losses, and
a keep/retire recommendation for default-on answer-time vision.

## Architecture / isolation

- **Reused units (no change):** `PDFParser.extract_images`, `process_vision_job` / `VisionExtractJob`,
  `ElasticsearchStore`, `LLMGateway.vision_response_batch`, `scripts/eval/judge_vision_captions.py`,
  the LLM-judge eval harness.
- **New units (scripts, controller-run, not prod code):**
  - `scripts/eval/vision_prod_augment.py` — Phase 2 driver (snapshot check → per-doc extract →
    enqueue/run vision_extract → verify).
  - `scripts/eval/build_vision_evalset.py` — Phase 3 (generate → dependency-filter → write jsonl).
  - `scripts/eval/run_vision_e2e_ab.py` — Phase 4 (ON/OFF agent runs → judge → report).
- Each script has one clear purpose, reads config from env, and is independently runnable +
  re-runnable.

## Error handling / safety

- Phase 2 is additive; ES snapshot before; rollback = delete image segments. Live index stays up.
- Worker handles provider 429/503 with retry + RPM cap; a doc that fails captioning logs + is
  skipped (never corrupts text segments).
- Phase 1 gates Phase 2; a Phase-1 NO-GO stops the slice.

## PHI
Corpus deemed non-sensitive (2026-07-19 decision) → cloud captioning sanctioned for ingest. This
is the one deliberate departure from the earlier PHI-local-ingest posture; documented here and in
the re-ingest runbook. Answer-time vision (Slice A) remains opt-in.

## Testing / verification
- Phase 1: judge aggregate meets the fixed gate.
- Phase 2: per-doc image-segment count > 0; spot-check captions; text-segment count unchanged
  (no duplication); rollback query rehearsed on the scratch index first.
- Phase 3: every kept Q fails the text-only answerer (dependency filter asserted).
- Phase 4: ON/OFF both run to completion; delta reported with N and CIs where available.

## Deliverables
- `docs/eval/vision_caption_quality_gemini_2026-07-19.md` (Phase 1 gate).
- Augmented prod index (Phase 2) + snapshot + rehearsed rollback.
- `data/eval/vision_evalset_2026-07-19.jsonl` (Phase 3).
- `docs/eval/vision_e2e_2026-07-19.md` — the ON/OFF lift + keep/retire call (Phase 4).
- Reusable scripts: `vision_prod_augment.py`, `build_vision_evalset.py`, `run_vision_e2e_ab.py`.
