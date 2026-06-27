# Remove MinerU — design

**Date:** 2026-06-27
**Status:** approved (ready for implementation plan)

## Problem

MinerU was added to the PDF ingestion path on the assumption it "OCRs better." That
assumption was never validated against the corpus, and MinerU is the single heaviest
component in the stack: `mineru[all]` pulls **vLLM** (+ `mineru-vl-utils`, torch-heavy
deps), and its default `vlm-auto-engine` backend spins up an in-process vLLM serving
Qwen2-VL — which OOMs the home box (16 GB VRAM shared with ES/TEI/ollama; ~15.7 GB host
RAM). Per the project's standing rule (*every change must move a trustworthy real-corpus
number above its noise floor, or it doesn't ship*), MinerU never earned its cost.

The ingestion pipeline **already** has a complete OCR path that does not need MinerU:
per page, `PyMuPDF text → Tesseract → vision-LLM (qwen2.5-vl via ollama)`. The vision tier
is already loaded for the existing image feature, so it costs no new resources. MinerU's
only real edge over this path is structured table/formula extraction — unproven to matter
for this medical corpus.

## Goal

Remove MinerU entirely. PDF parsing relies on the existing tier path. No new dependencies.
Outcome: OOM risk gone, `vllm`/`mineru` dropped from the lockfile (large disk/install win),
simpler ingestion code.

## Non-goals

- No replacement parser (Docling/Marker) in this change. (Considered; user chose
  remove-only. A future evidence-based A/B can revisit if structured tables prove needed.)
- No change to the tier path behavior (PyMuPDF → Tesseract → vision-LLM stays as-is).
- No re-ingest of the corpus as part of this change (corpus is currently empty after the
  stack reset; re-ingest happens separately when needed).

## Post-state PDF flow (unchanged tiers, MinerU branch gone)

```
per page:
  1. PyMuPDF.get_text("text", sort=True)        # text layer
  2. if len(text) < PDF_OCR_MIN_TEXT_CHARS  →  Tesseract OCR (PDF_OCR_LANG)
  3. if still thin and PDF_OCR_VISION_FALLBACK →  vision-LLM (VISION_PROVIDER)
```

## Changes

### Delete files (3)
- `src/agentrag/ingestion/parsers/mineru_parser.py`
- `src/agentrag/ingestion/parsers/pptx_via_mineru.py` — orphaned; no caller in `src/`
  (PPTX already routes through MarkItDown; `INGEST_USE_MINERU_FOR_PPTX` was off + unwired).
- `tests/ingestion/test_mineru_parser.py`

### Edit code (3)
- `src/agentrag/ingestion/parsers/pdf_parser.py`
  - Remove the `if backend == "mineru" and ocr_enabled:` block (≈ lines 79–139), including
    the `mineru_parser` import, `thin_pages`/`thin_fraction`/`needs_mineru` logic, and the
    per-page MinerU merge/return.
  - Remove `"mineru"` from the two `source` summary tuples (`("text","ocr","vision","mineru")`
    → `("text","ocr","vision")`).
  - The `backend` local becomes unused for branching; keep reading the setting only if still
    referenced, otherwise drop the local. The per-page tier loop runs unconditionally.
- `src/agentrag/config.py`
  - Delete MinerU settings: `MINERU_BACKEND`, `MINERU_OUTPUT_DIR`,
    `PDF_MINERU_MIN_THIN_FRACTION`, `MINERU_LANG`, `MINERU_DEVICE`,
    `INGEST_USE_MINERU_FOR_PPTX`, plus the MinerU comment block.
  - **Fix duplicate field:** `PDF_PARSER_BACKEND` is declared twice (line 185
    `Literal["pymupdf","markitdown"] = "pymupdf"` — dead/shadowed; line 272 `str = "hybrid"`
    — effective). Delete the dead `Literal` declaration; keep a single
    `PDF_PARSER_BACKEND: str = "hybrid"`. Update its comment to list only the tier path
    (mineru no longer a value).
- `pyproject.toml`
  - Remove the `"mineru[all]>=3.1.14"` dependency, then regenerate the lockfile with
    `uv lock`. Confirm `vllm` and `mineru` no longer appear in `uv.lock`.

### Config
- `.env`
  - `PDF_PARSER_BACKEND=mineru` → `PDF_PARSER_BACKEND=hybrid`.
  - Delete `MINERU_BACKEND`, `MINERU_OUTPUT_DIR`, `MINERU_LANG`, `MINERU_DEVICE`,
    `INGEST_USE_MINERU_FOR_PPTX` lines and their MinerU comment block.

### Docs (light touch — update only the operational ones)
- `docs/README-full.md` (§5.8 PDF/OCR/MinerU and the parser/backend tables/tree),
  `src/agentrag/ingestion/README.md` (env table + parser inventory), `.env` comments,
  `scripts/install_system.sh:160` (drop the MinerU/vLLM CPU warning).
- Leave dated records untouched: `docs/eval/*`, `docs/superpowers/specs/2026-06-05-*`,
  `BAO_CAO_DU_AN.md`, `BAO_CAO_VITAL.md`, `docs/kien-truc-vital.md`, `README.md` historical
  mentions (update README.md feature bullet if trivial).

## Verification

1. `uv lock` succeeds; `grep -i 'name = "vllm"' uv.lock` and `'name = "mineru"'` return
   nothing.
2. `uv run pytest tests/ingestion -q` → green (after deleting the mineru test).
3. Config import smoke: `uv run python -c "from src.agentrag.config import settings;
   print(settings.PDF_PARSER_BACKEND)"` → `hybrid`, no AttributeError on removed settings.
4. Grep clean: no remaining `import`/attribute references to `mineru_parser`,
   `pptx_via_mineru`, `MINERU_`, or `INGEST_USE_MINERU_FOR_PPTX` in `src/` or `tests/`.
5. Parse smoke: run `PDFParser().parse()` on one scanned PDF (re-add a sample to
   `data/originals` or use any scanned PDF) → returns `page_data` with `source ∈
   {text, ocr, vision}`, no MinerU import/branch hit.

## Risk / tradeoff

- **Lost:** structured table-to-HTML / formula-to-LaTeX (MinerU's only edge — unproven for
  this corpus). Scanned tables degrade to vision-LLM-described text. Accepted by the user.
- **PPTX:** unchanged in practice — falls to MarkItDown, which was already the default.
- **Reversibility:** MinerU can be re-added later (dep + parser shim) if an evidence-based
  A/B shows table extraction is needed. This change is git-reversible.
