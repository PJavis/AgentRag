# Remove MinerU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove MinerU and its heavy deps (vLLM) from PDF ingestion; rely on the existing PyMuPDF → Tesseract → vision-LLM tier path.

**Architecture:** Delete the MinerU branch in `pdf_parser.py`, the two MinerU parser shims, the MinerU config settings (fixing a duplicate `PDF_PARSER_BACKEND` field), and the `mineru[all]` dependency. The per-page tier loop already in `pdf_parser.py` handles all PDFs unconditionally after removal. No new dependencies, no replacement parser.

**Tech Stack:** Python, pydantic-settings, PyMuPDF, Tesseract (pytesseract), uv, pytest.

## Global Constraints

- No new runtime dependencies introduced by this change.
- Tier path behavior (PyMuPDF → Tesseract → vision-LLM) must stay byte-for-byte unchanged.
- After completion: `vllm` and `mineru` must NOT appear in `uv.lock`.
- No re-ingest of the corpus is part of this change (corpus is empty post stack-reset).
- Leave dated records untouched: `docs/eval/*`, `docs/superpowers/specs/2026-06-05-*`, `BAO_CAO_DU_AN.md`, `BAO_CAO_VITAL.md`, `docs/kien-truc-vital.md`.
- Run all Python via `uv run`. Working dir: repo root `/home/nguyenquocdung/work/AgentRag`.

---

### Task 1: Remove the MinerU branch from PDFParser

**Files:**
- Modify: `src/agentrag/ingestion/parsers/pdf_parser.py` (remove lines ~71, ~79–139, and `"mineru"` in two source-summary tuples ~128 & ~186)

**Interfaces:**
- Consumes: nothing new.
- Produces: `PDFParser.parse(file_path) -> {"parsed_content": str, "pages": int, "page_data": [{"page_num": int, "text": str, "source": "text"|"ocr"|"vision"}]}` — same contract, minus any `source == "mineru"` rows.

- [ ] **Step 1: Confirm the exact block to remove**

Run: `grep -n 'backend\|mineru\|"text", "ocr", "vision"' src/agentrag/ingestion/parsers/pdf_parser.py`
Expected: shows line ~71 (`backend = ...`), the `if backend == "mineru"` block start ~84, and two summary tuples containing `"mineru"`.

- [ ] **Step 2: Delete the `backend` local**

Remove this line (≈ line 71):
```python
        backend = (settings.PDF_PARSER_BACKEND or "hybrid").lower()
```
Leave the `ocr_enabled`/`ocr_min`/`ocr_dpi`/`ocr_lang`/`vision_fallback`/`vision_threshold` locals (still used by the tier loop).

- [ ] **Step 3: Delete the entire MinerU branch**

Remove the whole block starting at the comment `# MinerU backend: only hand the WHOLE PDF...` through the line `logger.info("MinerU per-page failed; falling back to Tesseract+vision path")` (≈ lines 79–139). The next surviving line must be:
```python
        parts: list[str] = []
        page_data: list[dict[str, Any]] = []
```

- [ ] **Step 4: Drop `"mineru"` from the source-summary tuples**

There are two occurrences of:
```python
                        for s in ("text", "ocr", "vision", "mineru")
```
Change BOTH to:
```python
                        for s in ("text", "ocr", "vision")
```

- [ ] **Step 5: Verify no MinerU references remain in the file**

Run: `grep -ni 'mineru' src/agentrag/ingestion/parsers/pdf_parser.py`
Expected: no output.

- [ ] **Step 6: Import + parse smoke**

Run:
```bash
uv run python -c "from src.agentrag.ingestion.parsers.pdf_parser import PDFParser; print('ok')"
```
Expected: `ok` (no ImportError; note `mineru_parser` is only imported inside the now-deleted block).

- [ ] **Step 7: Commit**

```bash
git add src/agentrag/ingestion/parsers/pdf_parser.py
git commit -m "refactor(ingest): drop MinerU branch from PDFParser; rely on tier path"
```

---

### Task 2: Delete the MinerU parser shims and their test

**Files:**
- Delete: `src/agentrag/ingestion/parsers/mineru_parser.py`
- Delete: `src/agentrag/ingestion/parsers/pptx_via_mineru.py`
- Delete: `tests/ingestion/test_mineru_parser.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (pure removal).

- [ ] **Step 1: Confirm `pptx_via_mineru` has no live caller**

Run: `grep -rn 'pptx_via_mineru' src/ tests/ | grep -v 'parsers/pptx_via_mineru.py'`
Expected: no output (only the file itself references the name). If any caller appears, STOP and report — the design assumed it is orphaned.

- [ ] **Step 2: Delete the three files**

```bash
git rm src/agentrag/ingestion/parsers/mineru_parser.py \
       src/agentrag/ingestion/parsers/pptx_via_mineru.py \
       tests/ingestion/test_mineru_parser.py
```

- [ ] **Step 3: Verify no imports of the deleted modules remain**

Run: `grep -rn 'mineru_parser\|pptx_via_mineru' src/ tests/`
Expected: no output.

- [ ] **Step 4: Ingestion tests green**

Run: `uv run pytest tests/ingestion -q`
Expected: PASS (collection succeeds; no import errors from the deleted test).

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor(ingest): delete mineru_parser + pptx_via_mineru shims + test"
```

---

### Task 3: Remove MinerU settings and fix the duplicate PDF_PARSER_BACKEND field

**Files:**
- Modify: `src/agentrag/config.py` (delete dead `Literal` line ~185; delete MinerU settings block ~273–297; keep single `PDF_PARSER_BACKEND: str = "hybrid"`)

**Interfaces:**
- Consumes: nothing.
- Produces: `settings.PDF_PARSER_BACKEND: str` (single field, default `"hybrid"`); MinerU settings no longer exist on `settings`.

- [ ] **Step 1: Delete the dead/shadowed Literal declaration**

Remove this line (≈ line 185 — it is overridden by the later `str` declaration, so it is already dead):
```python
    PDF_PARSER_BACKEND: Literal["pymupdf", "markitdown"] = "pymupdf"
```
Also remove its preceding comment line:
```python
    # PDF parser backend: pymupdf (page-aware, recommended) or markitdown (legacy)
```

- [ ] **Step 2: Replace the surviving PDF_PARSER_BACKEND comment + delete MinerU settings**

Find the block (≈ lines 269–297) that starts with:
```python
    # PDF parser backend escalation when PyMuPDF text-layer is thin:
    #   hybrid (default) → Tesseract → vision LLM
    #   mineru           → MinerU (layout + OCR + formula + table) replaces tiers 2+3
    PDF_PARSER_BACKEND: str = "hybrid"
```
…through the `INGEST_USE_MINERU_FOR_PPTX: bool = False` line. Replace the ENTIRE block with just:
```python
    # PDF parser backend escalation when PyMuPDF text-layer is thin:
    #   hybrid (default) → Tesseract → vision LLM
    PDF_PARSER_BACKEND: str = "hybrid"
```
This deletes `MINERU_BACKEND`, `MINERU_OUTPUT_DIR`, `PDF_MINERU_MIN_THIN_FRACTION`, `MINERU_LANG`, `MINERU_DEVICE`, `INGEST_USE_MINERU_FOR_PPTX` and their comments.

- [ ] **Step 3: Verify no MinerU settings remain**

Run: `grep -ni 'mineru' src/agentrag/config.py`
Expected: no output.

- [ ] **Step 4: Verify `Literal` import still needed (avoid unused-import lint)**

Run: `grep -n 'Literal\[' src/agentrag/config.py`
Expected: still several hits (e.g. `VISION_PROVIDER`, `VISION_INGEST_MODE`). If ZERO hits, also remove `Literal` from the `from typing import` line. (Expected: hits remain → leave the import.)

- [ ] **Step 5: Config import smoke**

Run:
```bash
uv run python -c "from src.agentrag.config import settings; print(settings.PDF_PARSER_BACKEND); print(hasattr(settings,'MINERU_BACKEND'))"
```
Expected: `hybrid` then `False`.

- [ ] **Step 6: Full unit suite still green for config-dependent ingestion**

Run: `uv run pytest tests/ingestion -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentrag/config.py
git commit -m "refactor(config): remove MinerU settings; collapse duplicate PDF_PARSER_BACKEND"
```

---

### Task 4: Drop the mineru dependency and relock

**Files:**
- Modify: `pyproject.toml` (remove `"mineru[all]>=3.1.14"`)
- Modify: `uv.lock` (regenerated)

**Interfaces:**
- Consumes: nothing.
- Produces: lockfile without `vllm`/`mineru`.

- [ ] **Step 1: Remove the dependency line**

In `pyproject.toml`, delete the line:
```toml
    "mineru[all]>=3.1.14",
```

- [ ] **Step 2: Relock**

Run: `uv lock`
Expected: completes successfully; resolution drops mineru and its transitive vllm/mineru-vl-utils.

- [ ] **Step 3: Verify vllm + mineru gone from lockfile**

Run: `grep -nE 'name = "(vllm|mineru|mineru-vl-utils)"' uv.lock`
Expected: no output.

- [ ] **Step 4: Sync the environment**

Run: `uv sync`
Expected: completes; removes mineru/vllm from the venv.

- [ ] **Step 5: Import smoke after sync**

Run: `uv run python -c "import src.agentrag.ingestion.parsers.pdf_parser as p; print('ok')"`
Expected: `ok`.

- [ ] **Step 6: Ingestion suite green post-sync**

Run: `uv run pytest tests/ingestion -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(deps): remove mineru[all] (drops vLLM); rely on tier OCR path"
```

---

### Task 5: Clean .env and operational docs

**Files:**
- Modify: `.env` (PDF_PARSER_BACKEND value + delete MINERU_* lines)
- Modify: `docs/README-full.md` (§5.8 + parser/backend tables + file tree)
- Modify: `src/agentrag/ingestion/README.md` (env table + parser inventory)
- Modify: `scripts/install_system.sh` (line ~160 MinerU/vLLM warning)
- Modify: `README.md` (feature bullet, if it names MinerU)

**Interfaces:** none (config + docs only).

- [ ] **Step 1: Fix `.env`**

Set:
```
PDF_PARSER_BACKEND=hybrid
```
Delete the lines `MINERU_BACKEND=...`, `MINERU_OUTPUT_DIR=...`, `MINERU_LANG=...`, `MINERU_DEVICE=...`, `INGEST_USE_MINERU_FOR_PPTX=...` and the MinerU comment lines around them (the `#   mineru → MinerU ...` block).

- [ ] **Step 2: Update operational docs**

In `docs/README-full.md`, `src/agentrag/ingestion/README.md`: remove `mineru` as a `PDF_PARSER_BACKEND` value, delete the `MINERU_*` / `INGEST_USE_MINERU_FOR_PPTX` env rows, the "Install MinerU" / "Picking MINERU_BACKEND" sections, and the `mineru_parser.py` / `pptx_via_mineru.py` entries in any file-tree/inventory. In `scripts/install_system.sh` remove or reword the line that warns "MinerU + vLLM will run on CPU". In `README.md`, drop "MinerU (opt-in)" from the feature bullet if present.

- [ ] **Step 3: Verify no operational MinerU references remain**

Run:
```bash
grep -rni 'mineru' .env src/agentrag/ingestion/README.md docs/README-full.md scripts/install_system.sh README.md
```
Expected: no output.

- [ ] **Step 4: Config import smoke with the edited .env**

Run: `uv run python -c "from src.agentrag.config import settings; print(settings.PDF_PARSER_BACKEND)"`
Expected: `hybrid` (no validation error from a removed/extra env var).

- [ ] **Step 5: Commit**

```bash
git add .env src/agentrag/ingestion/README.md docs/README-full.md scripts/install_system.sh README.md
git commit -m "docs(ingest): drop MinerU from .env + operational docs"
```

---

### Task 6: Final verification sweep

**Files:** none (verification only).

- [ ] **Step 1: Repo-wide grep for live MinerU references (excluding dated records)**

Run:
```bash
grep -rniE 'mineru' --include='*.py' --include='*.toml' --include='*.env' src/ tests/ pyproject.toml .env
```
Expected: no output. (Dated `docs/` records are intentionally retained.)

- [ ] **Step 2: Lockfile clean**

Run: `grep -nE 'name = "(vllm|mineru)"' uv.lock`
Expected: no output.

- [ ] **Step 3: Full test suite (ingestion + config-touching)**

Run: `uv run pytest tests/ingestion -q`
Expected: PASS.

- [ ] **Step 4: Parse smoke on a real scanned PDF**

Place any scanned PDF at `/tmp/scan.pdf` (or copy one back into `data/originals`), then:
```bash
uv run python -c "
from src.agentrag.ingestion.parsers.pdf_parser import PDFParser
r = PDFParser().parse('/tmp/scan.pdf')
print('pages', r['pages'])
print('sources', sorted({d.get('source') for d in r['page_data']}))
"
```
Expected: prints a page count and `sources` ⊆ `{'text','ocr','vision'}` (no `mineru`), no exceptions.

- [ ] **Step 5: No commit (verification task).** If any step fails, open a follow-up task; do not mark the plan complete.
