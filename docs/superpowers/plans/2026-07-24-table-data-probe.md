# Table-Data Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Revised 2026-08-02 (rev 2)** — the corpus survey
(`docs/eval/table_probe_corpus_survey_2026-08-02.md`) invalidated four assumptions in
rev 1. Tasks 1–3 are now **done**; Tasks 4–9 are rewritten around the corrected
instrument. See the spec changelog for the full list.

**Revised 2026-09-06 (rev 3)** — a code review of the rev-2 instrument found the gate
was classifying `extract()` while arm B would have emitted `to_markdown()`, so the
corruption it blocks was invisible to it (77% of gate-passing detections carried an
invented `ColN` header). Tasks 1–3 were re-implemented and re-measured; Task 5's
insertion point moved after the OCR block; Task 6 is withdrawn (the renderer removes
the need); Task 9's runner is rewritten around the real `ingest_folder` /
`get_agent_service` signatures. Every number in this plan is re-measured against the
corpus as of that date.

**Goal:** produce a trustworthy go/no-go signal on whether preserving PDF table
structure at ingest improves answers.

**Architecture:** a pure-function library holds all testable logic (track selection,
target ranking, row validation, paired statistics). Thin script shells wrap it with
real I/O (ES/TEI pings, PyMuPDF, ingest, `agent.chat`). A throwaway
`PDF_PRESERVE_TABLES` flag adds a *gated* table-preserving arm to the PDF parser. The
A/B runner mirrors `scripts/eval/run_vision_e2e_ab.py`.

**Tech Stack:** Python, pytest, PyMuPDF (`fitz`), existing agentrag eval harness
(`load_local_jsonl`, `score_correctness`, `get_agent_service`), Elasticsearch, TEI.

## Global Constraints

- Probe is disposable measurement code — no production default-on behavior.
  `PDF_PRESERVE_TABLES` defaults `False`. `table_quality.py` and `table_probe_lib.py`
  are deleted with it.
- No new heavy deps: NO MinerU / docling / camelot / pdfplumber. PyMuPDF
  `page.find_tables()` only.
- **All corpus work runs over the 29 unique documents**, never the 116 files on disk.
- **Arm B only rewrites tables passing `is_safe_to_markdown`.** 77% of detections
  (131 of 171) are layout artifacts whose markdown is worse than the flat text it
  replaces.
- **Arm B renders via `table_quality.render_markdown`, never `table.to_markdown()`.**
  The gate judges `extract()` cells; emitting a different renderer's output means
  gating one representation and shipping another. PyMuPDF's renderer is itself the
  source of the invented `ColN` headers and mirrored cells the gate exists to block.
- **The table append runs after the OCR block in `PDFParser.parse`**, so arm B cannot
  change OCR/vision routing. Appending before it would make arm B lose pages that arm A
  recovers — a confound with nothing to do with tables.
- **Decision rule is a paired sign test**, not a mean delta. The rev-1 `+0.10` gate is
  withdrawn — at n=10 it fires on one question flipping.
- `INCONCLUSIVE` is a valid, reportable outcome. Never round it up to GO.
- Judge must be grounded: pass gold context to `score_correctness`.
- Do NOT reuse `prod_corpus_evalset_v3.jsonl` gold — flagged INVALID vs real corpus.
- Evalset row schema: `id`, `question`, `reference_answer`, `gold_contexts` (list),
  **plus `source_doc` and `source_page`** — gold matching keys on `(doc, page)` because
  the arms produce different chunk text for the same table. `EvalExample` and
  `load_local_jsonl` carry both fields through (added 2026-09-06); before that the
  loader silently dropped them and provenance matching was impossible.
- **Timeouts are excluded, never scored 0.0.** A timeout measures stack latency, and
  arm B's larger packed context times out more often — scoring it as a loss turns the
  sign test into a latency test.
- **Every evalset row carries `corpus_docs_sha`** from the survey, and the runner
  refuses to score on a mismatch. `eval/corpus_fingerprint.py` cannot be used here:
  it hashes `(document_title, segment_count)` and arm B changes segment counts by
  design, so it would flag a correct run. `corpus_docs_sha` hashes deduplicated
  document *content* — identical under both arms, and still catches a PDF added or
  removed between authoring the evalset and running it, which is what would make the
  `(doc, page)` gold point at the wrong page.
- **Every question that does not reach the comparison is counted in the report**
  (scored by one arm only, ineligible, timed out). §4 requires the accounting, and a
  silent intersection is survivorship bias in the direction that favours GO.
- Test command: `PYTHONPATH=. uv run pytest <path> -v`.

---

## Task 1: Table-quality gate + renderer — **DONE (2026-08-02, rev 3 2026-09-06)**

`src/agentrag/ingestion/parsers/table_quality.py` + `tests/ingestion/test_table_quality.py` (24 tests).

Classifies extracted cells as `real_data` / `nonnumeric` / `single_column` /
`layout_prose` / `layout_dup` / `degenerate`. Three entry points:

- `is_safe_to_markdown(rows)` — structural only; **arm B's gate**. A numberless
  comparison matrix passes; a mirrored prose box does not.
- `is_data_grid(rows)` — structure + numeric density; used to rank targets for
  question authoring, *not* to gate arm B (gating on numeric density would silently
  drop legitimate text tables).
- `render_markdown(rows, max_tokens=...)` — **what arm B emits.** Renders the same
  cells the gate judged, header from the document, blocks packed to the chunk budget.

Rev-3 corrections, each measured on the 29-document corpus:

| defect | effect | now |
|---|---|---|
| gate read `extract()`, arm B would emit `to_markdown()` | 69 of 90 gate-passing detections carried an invented `ColN` header | arm B renders from the judged cells; **0 of 40** |
| `max(len(row)) < 2` counted `None` placeholders as columns | one-column prose strips passed as "2-column tables" | `single_column` kind; 50 detections reclassified |
| mirroring tested whole-row only | PyMuPDF mirrors per column-pair; corpus-wide `layout_dup` was 1/171 | adjacent-pair test, guarded on cell length so `1|1|1` score columns stay data |
| `\d` anywhere made a cell numeric | sentences mentioning a year, and checklists whose only digits are the STT row counter, ranked as `real_data` probe targets | digit-density test + leading-ordinal-column exclusion; data grids 35 → 19 |

## Task 2: Corpus survey — **DONE (2026-08-02)**

`scripts/eval/table_probe_corpus_survey.py` + `tests/eval/test_table_probe_corpus_survey.py` (25 tests).
Output: `data/eval/table_probe_corpus_survey.json`,
`docs/eval/table_probe_corpus_survey_2026-09-06.md`
(supersedes the 2026-08-02 run).

Also emits `--unique-list` and `--dedupe-dir` (a symlink farm of the unique documents).
Spec §3 Step 0a says every later step consumes the unique-document list; before rev 3
only the integer count was emitted, so `--corpus <UNIQUE_DOCS_DIR>` pointed at a
directory nothing produced. Every PyMuPDF call is guarded and the document is always
closed, so one malformed page cannot discard a whole-corpus run.

Measured (re-run 2026-09-06 with the corrected classifier): 116 files → 29 unique
(87 redundant copies); 684 pages, 25% with no text layer; 171 detections → 19 data
grids across 5 docs; 131 of 171 (77%) unsafe to convert. Each data grid records
`est_tokens`, the measured size of its rendered markdown.

## Task 3: Pure helper lib — **DONE (2026-08-02)**

`scripts/eval/table_probe_lib.py` + `tests/eval/test_table_probe_lib.py` (39 tests).

`decide_track`, `arm_index_name`, `rank_targets` (tables, not docs), `validate_probe_row`
(requires `source_doc`/`source_page`), `paired_outcomes`, `sign_test_p`, `decide_paired`,
`mean_delta` (colour only).

Rev-3 corrections:

- `rank_targets` interleaves documents round-robin. The flat `(-cells, doc, page)` sort
  was document-blind and returned 8 of the top 10 from one file — worse than the 47%
  doc-level share the table-level unit was introduced to avoid, while the sign test
  assumes independent draws over the corpus. Now 3 of 10, measured. It also takes
  `max_tokens` and filters on a measured survey figure: cell count predicts
  rendered size badly (20 cells → 955 tokens, 104 cells → 429). It filters on the
  survey's `max_block_tokens` — the largest emitted BLOCK, not the whole-table total,
  because a big table is emitted as several under-budget blocks the chunker keeps whole.
- `paired_outcomes` names and counts every question that does not reach the comparison
  (`n_missing`, `n_ineligible`). Intersecting the arms silently drops exactly the
  questions arm B failed to score — bias in the direction that favours GO.
- `mean_delta` takes `eligible`, so the number printed beside the decision averages the
  population the decision was made on.
- `validate_probe_row` rejects `True`/`0`/negative pages (`isinstance(True, int)` is
  `True`).

> `arm_index_name` is kept for the report header only. Arms are isolated by **sequence**,
> not by index name: `get_agent_service()` takes no arguments and `ElasticsearchStore`
> reads `settings.ELASTICSEARCH_INDEX_NAME` at construction, so there is no supported
> way to point ingest and retrieval at a per-arm index.

Decision rule: **GO** iff B wins ≥2× as often as it loses AND sign-test p < 0.05;
**NO-GO** if B does not lead; **INCONCLUSIVE** otherwise. A regression test pins the
rev-1 flaw: n=10 with one question flipped yields exactly the old `+0.10` GO threshold
and the new rule returns INCONCLUSIVE.

---

### Task 4: Preflight track selector (`table_probe_preflight.py`)

Thin shell: ping real ES + TEI, check judge key, print the track via `decide_track`.

**Files:**
- Create: `scripts/eval/table_probe_preflight.py`
- Test: `tests/eval/test_table_probe_preflight.py`

**Interfaces:**
- Consumes: `decide_track` (Task 3).
- Produces: `preflight() -> tuple[str, str]`; CLI prints `TRACK=<track> REASON=<reason>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_table_probe_preflight.py
from unittest.mock import patch
from scripts.eval import table_probe_preflight as pf


def test_preflight_reports_full_when_all_up():
    with patch.object(pf, "_es_up", return_value=True), \
         patch.object(pf, "_tei_up", return_value=True), \
         patch.object(pf, "_judge_key", return_value="sk-x"):
        track, _ = pf.preflight()
    assert track == "full"


def test_preflight_reports_offline_when_es_down():
    with patch.object(pf, "_es_up", return_value=False), \
         patch.object(pf, "_tei_up", return_value=True), \
         patch.object(pf, "_judge_key", return_value="sk-x"):
        track, reason = pf.preflight()
    assert track == "offline" and "elasticsearch" in reason
```

- [ ] **Step 2: Run test to verify it fails** — `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/eval/table_probe_preflight.py
"""Preflight: decide FULL vs OFFLINE track for the table probe."""
from __future__ import annotations
import os
import sys

sys.path.insert(0, ".")
from scripts.eval.table_probe_lib import decide_track


def _es_up() -> bool:
    from src.agentrag.config import settings
    try:
        import httpx
        url = getattr(settings, "ELASTICSEARCH_URL", None) or "http://127.0.0.1:9200"
        return httpx.get(url, timeout=3.0).status_code < 500
    except Exception:
        return False


def _tei_up() -> bool:
    from src.agentrag.config import settings
    try:
        import httpx
        url = getattr(settings, "TEI_URL", None) or getattr(settings, "EMBEDDING_BASE_URL", None)
        if not url:
            return False
        return httpx.get(url.rstrip("/") + "/health", timeout=3.0).status_code < 500
    except Exception:
        return False


def _judge_key() -> str | None:
    return os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("GEMINI_API_KEY")


def preflight() -> tuple[str, str]:
    return decide_track(_es_up(), _tei_up(), _judge_key())


if __name__ == "__main__":
    track, reason = preflight()
    print(f"TRACK={track} REASON={reason}")
```

> VERIFY AT HOME: the real settings attribute names for the ES and TEI URLs. The
> `getattr(..., default)` form keeps this import-safe if the names differ.

- [ ] **Step 4: Run test to verify it passes** (2 tests)
- [ ] **Step 5: Smoke** — `PYTHONPATH=. uv run python scripts/eval/table_probe_preflight.py`
- [ ] **Step 6: Commit** — `feat(eval): table-probe preflight track selector`

---

### Task 5: `PDF_PRESERVE_TABLES` flag + **gated** parser arm

Add the throwaway flag and an additive, gated table-markdown branch. Arm B keeps the
flat text AND appends each *safe* detected table as GFM markdown.

**Files:**
- Modify: `src/agentrag/config.py`
- Modify: `src/agentrag/ingestion/parsers/pdf_parser.py`
- Test: `tests/ingestion/test_pdf_preserve_tables.py`

- [ ] **Step 1: Add the config flag**

After `RETRIEVAL_RERANK_ENABLED` (~line 113):

```python
    #: PROBE-ONLY (throwaway, 2026-07-24). When True, PDFParser appends each page's
    #: detected tables as GFM markdown, gated on table_quality.is_safe_to_markdown.
    #: Arm B of the table-data probe. Delete with table_quality.py. Default OFF.
    PDF_PRESERVE_TABLES: bool = False
```

- [ ] **Step 2: Write the failing test**

```python
# tests/ingestion/test_pdf_preserve_tables.py
from unittest.mock import MagicMock, patch
from src.agentrag.ingestion.parsers.pdf_parser import PDFParser, _append_table_markdown

GRID = [["STT", "Liều"], ["1", "10mg"], ["2", "20mg"]]
# One populated cell per row: PyMuPDF reports 2 columns, extract() fills one.
# `to_markdown()` is what mirrors the text and invents "Col2".
SINGLE_COL = [["1. ĐỊNH NGHĨA", None], ["Lo lắng là phản ứng", None], ["của cơ thể", None]]


def _page(tables):
    tabs = []
    for rows in tables:
        t = MagicMock()
        t.extract.return_value = rows
        tabs.append(t)
    page = MagicMock()
    page.find_tables.return_value = MagicMock(tables=tabs)
    return page


def test_appends_markdown_for_a_real_grid():
    out = _append_table_markdown(_page([GRID]), "flat text")
    assert out.startswith("flat text")
    assert "| STT | Liều |" in out


def test_never_calls_pymupdf_to_markdown():
    """The gate judges extract(); emitting to_markdown() would emit ungated text."""
    page = _page([GRID])
    _append_table_markdown(page, "flat text")
    for t in page.find_tables.return_value.tables:
        t.to_markdown.assert_not_called()


def test_skips_layout_artifacts():
    """Converting these would corrupt the page and poison the A/B."""
    assert _append_table_markdown(_page([SINGLE_COL]), "flat text") == "flat text"


def test_mixed_page_keeps_only_the_safe_table():
    out = _append_table_markdown(_page([GRID, SINGLE_COL]), "flat text")
    assert "| STT | Liều |" in out
    assert "ĐỊNH NGHĨA" not in out


def test_noop_when_no_tables():
    assert _append_table_markdown(_page([]), "flat text") == "flat text"


def test_survives_find_tables_error():
    page = MagicMock()
    page.find_tables.side_effect = RuntimeError("no table layer")
    assert _append_table_markdown(page, "flat text") == "flat text"


def test_survives_extract_error():
    t = MagicMock()
    t.extract.side_effect = RuntimeError("bad table")
    page = MagicMock()
    page.find_tables.return_value = MagicMock(tables=[t])
    assert _append_table_markdown(page, "flat text") == "flat text"


def test_table_append_does_not_change_ocr_routing(monkeypatch, tmp_path):
    """THE load-bearing test for the '25% blind spot' scope claim.

    A page with a thin text layer must take the OCR path in BOTH arms. If the
    table markdown is appended before the OCR length check, arm B pushes the
    page over PDF_OCR_MIN_TEXT_CHARS, skips the OCR/vision fallback arm A takes,
    and loses content for reasons that have nothing to do with tables.

    Note the two mechanics this test depends on, both verified against
    pdf_parser.py: `import fitz` is INSIDE `PDFParser.parse`, so the module has no
    `fitz` attribute to patch — the import must be intercepted in `sys.modules`.
    And `parse` raises FileNotFoundError before opening anything, so the path
    argument must exist on disk.
    """
    import sys

    from src.agentrag.config import settings

    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", True)
    monkeypatch.setattr(settings, "PDF_OCR_FALLBACK_ENABLED", True)
    monkeypatch.setattr(settings, "PDF_OCR_MIN_TEXT_CHARS", 50)
    monkeypatch.setattr(settings, "PDF_OCR_VISION_FALLBACK", False)
    monkeypatch.setattr(settings, "PDF_OCR_VISION_THRESHOLD", 1)

    thin_page = _page([GRID])                     # 30 chars: under the threshold
    thin_page.get_text.return_value = "x" * 30
    thin_page.get_pixmap.return_value.tobytes.return_value = b"png"
    doc = MagicMock()
    doc.__iter__.return_value = iter([thin_page])
    fake_fitz = MagicMock()
    fake_fitz.open.return_value = doc

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with patch.dict(sys.modules, {"fitz": fake_fitz}), \
         patch("src.agentrag.ingestion.parsers.pdf_parser._ocr_tesseract",
               return_value="recovered OCR text") as ocr:
        PDFParser().parse(str(pdf))

    ocr.assert_called_once()   # arm B still took the OCR path
```

- [ ] **Step 3: Run test to verify it fails** — `ImportError: cannot import name '_append_table_markdown'`

- [ ] **Step 4: Add the helper + branch in `pdf_parser.py`**

Module-level, above `class PDFParser` (after `_ocr_tesseract`, ~line 46):

```python
def _append_table_markdown(page, text: str) -> str:
    """Probe arm B: append *structurally sound* tables as GFM markdown.

    Additive — flat text is preserved and the markdown is appended, so retrieval
    sees both. Tables failing `is_safe_to_markdown` are skipped: on this corpus
    77% of find_tables() detections are layout boxes whose markdown (mirrored
    cells, invented headers, one row per visual line) is worse than the flat text
    it would replace. Converting them would make arm B lose for reasons unrelated
    to tables. Any detection error leaves the text unchanged.

    Rendering goes through `render_markdown`, never `table.to_markdown()`: the
    gate judges the `extract()` cells, so the emitted text must come from those
    same cells. `to_markdown()` invents `ColN` headers and mirrors a single
    populated cell across columns — corruption the gate cannot see because it
    does not exist until that renderer runs. Blocks are packed to
    `SEARCH_CHUNK_MAX_TOKENS` and each repeats the header, so the chunker never
    has to cut inside a row.
    """
    from src.agentrag.config import settings
    from src.agentrag.ingestion.parsers.table_quality import render_markdown

    try:
        tables = list(page.find_tables().tables)
    except Exception:
        return text

    blocks = []
    for table in tables:
        try:
            md = render_markdown(
                table.extract(), max_tokens=settings.SEARCH_CHUNK_MAX_TOKENS
            )
        except Exception:
            continue
        if md.strip():
            blocks.append(md.strip())

    if not blocks:
        return text
    return text + "\n\n" + "\n\n".join(blocks)
```

In `PDFParser.parse`, **after the OCR/vision block closes** (after the final
`stripped = text.strip()` inside it, ~line 112) and before `if not stripped:`:

```python
            if settings.PDF_PRESERVE_TABLES:
                text = _append_table_markdown(page, text)
```

> **Not** immediately after `text = page.get_text(...)` at line 83. Line 84's
> `stripped = text.strip()` is the sole input to the OCR gate at line 86
> (`if ocr_enabled and len(stripped) < ocr_min`, `PDF_OCR_MIN_TEXT_CHARS=50`).
> Appending pipes, a header and a `|---|` separator to a page holding ~30–45
> characters of text layer pushes it past 50: arm A takes the Tesseract/vision
> fallback and recovers the whole page, arm B skips it and keeps a sparse text
> layer plus a markdown stub. Arm B would lose content arm A has, for reasons
> unrelated to tables — and it would falsify the "arm B is byte-identical to arm
> A on the 25% of pages with no text layer" claim that the spec, the report and
> the report template all repeat. Appending after the OCR block leaves routing
> untouched: on an OCR page `find_tables()` finds nothing anyway.
> `test_table_append_does_not_change_ocr_routing` pins this.

- [ ] **Step 5: Run test to verify it passes** (8 tests)
- [ ] **Step 6: Verify default-off path untouched** — `PYTHONPATH=. uv run pytest tests/ingestion/ -v`
- [ ] **Step 7: Commit** — `feat(ingest): gated PDF_PRESERVE_TABLES probe arm (default off)`

---

### Task 6: Chunk-window fit — **CLOSED IN THE RENDERER (2026-09-06)**

**Rev 2 planned a table-atomic chunking branch here. It is withdrawn.** The
branch as drafted could not run and its test could not pass:

- `if settings.PDF_PRESERVE_TABLES and _is_markdown_table(paragraph)` was to go
  into `hybrid_chunker.py`, which imports only `hashlib`, `re`, `typing` and
  `tiktoken`. There is no `settings` import, so every chunking call — for every
  document type, flag on or off — would raise `NameError`.
- The paired test built `HybridChunker(max_tokens=64, split_on_paragraphs=True)`
  and never set `PDF_PRESERVE_TABLES` (default `False`), so it exercised the
  flag-off path while asserting flag-on behaviour.
- It asserted `content.count("|---|") == 1` against a fixture whose separator row
  is the 4-column `|---|---|---|---|`, whose non-overlapping count is 2 —
  unsatisfiable for any table wider than two columns. Under TDD that pushes the
  executor to edit correct chunker behaviour to satisfy a counting bug.

The underlying need is real: `SEARCH_CHUNK_MAX_TOKENS=512`, and a table with no
blank lines arrives at `_chunk_section_by_paragraph` as one oversized paragraph,
which `hybrid_chunker.py:138` (under the branch at :133) hands to
`_chunk_section_by_tokens` — a blind token window that cuts mid-row.

`render_markdown` removes the need instead of special-casing the chunker. It
packs rows into blocks that fit the token budget, separates blocks with a blank
line, and repeats the header on each. Every block is therefore an ordinary
paragraph that already fits the window, so the chunker never reaches the
oversized branch and never cuts inside a row — and a block that *is* retrieved
alone still names its columns.

Measured on the 29-document corpus at `max_tokens=512`: 40 gate-passing tables →
55 emitted blocks, **0 over budget**, worst block 500 tokens. A fixed row count
could not do this — tokens per cell range 4.1–47.8 (median 9.4), so the
`ROWS_PER_BLOCK=8` fallback gives blocks of 53–1090 tokens and 7 of the 40
tables (13% of emitted blocks) overflow the window.

**No production chunker change ships.** That was the strongest argument against
rev 2's version: it put a probe-scoped branch on the hot path for every ingested
document, and the two ways to make its test pass were to monkeypatch a
process-wide singleton inside a chunker unit test (making `tests/ingestion`
order-dependent, since the A/B runner mutates the same singleton) or to flip the
default to `True` — turning a throwaway, default-off probe into default-on
production chunking.

- [x] Covered by `tests/ingestion/test_table_quality.py::test_render_packs_blocks_to_a_token_budget`
      and `::test_render_emits_an_oversized_single_row_whole`.

---

### Task 7: Evalset validator + hand-authored questions

> **BLOCKING PRECONDITION (2026-09-06).** Do not author questions until the power
> question in `docs/eval/table_probe_power_analysis_2026-09-06.md` is settled by a
> human decision. Summary: the unit of analysis is the **table**, not the question, so
> authoring more questions per table does **not** increase power; the corpus supports
> 27 structured candidate tables; the shipped rule cannot return GO below 6 discordant
> tables; and 14 of the 27 tables sit in one document. As written the probe can only
> detect arm B helping ~42% of tables, and cannot reach GO at all if outcomes cluster
> by document. The recommendation is to run it as estimation rather than as a gate.
>
> Also note the eligible set is the survey's `candidate_tables` (40 gate-passing, 27
> structured), **not** `data_grids` (19) — `is_data_grid` ranks, it has never gated.

**Files:**
- Create: `scripts/eval/validate_table_probe_evalset.py`
- Create: `data/eval/table_probe_evalset.jsonl` (hand-authored)
- Test: `tests/eval/test_validate_table_probe_evalset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_validate_table_probe_evalset.py
import json
from scripts.eval.validate_table_probe_evalset import validate_file

ROW = {"id": "tbl-01", "question": "Giá trị ô 'làm thành thạo' của bước 1.4?",
       "reference_answer": "0", "gold_contexts": ["| 1.4 | ... | 1 | 1 | 1 | 0 |"],
       "source_doc": "2f499de4.pdf", "source_page": 8}


def test_validate_file_passes_clean(tmp_path):
    p = tmp_path / "ok.jsonl"
    p.write_text(json.dumps(ROW) + "\n")
    assert validate_file(str(p)) == []


def test_validate_file_reports_missing_provenance(tmp_path):
    p = tmp_path / "bad.jsonl"
    row = {k: v for k, v in ROW.items() if k != "source_page"}
    p.write_text(json.dumps(row) + "\n")
    errs = validate_file(str(p))
    assert errs and errs[0].startswith("line 1") and "source_page" in errs[0]


def test_validate_file_reports_bad_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not json\n")
    assert "invalid JSON" in validate_file(str(p))[0]
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Write the validator**

```python
# scripts/eval/validate_table_probe_evalset.py
"""Validate the hand-authored table probe evalset against the row schema."""
from __future__ import annotations
import argparse
import json
import sys

sys.path.insert(0, ".")
from scripts.eval.table_probe_lib import validate_probe_row


def validate_file(path: str) -> list[str]:
    errors: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {i}: invalid JSON — {exc}")
                continue
            for e in validate_probe_row(row):
                errors.append(f"line {i}: {e}")
    return errors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/eval/table_probe_evalset.jsonl")
    args = ap.parse_args()
    errs = validate_file(args.path)
    if errs:
        print("\n".join(errs))
        sys.exit(1)
    print("OK — evalset valid")
```

- [ ] **Step 4: Run test to verify it passes** (3 tests)

- [ ] **Step 5: List the target tables**

```bash
PYTHONPATH=. uv run python -c "
import json; from scripts.eval.table_probe_lib import rank_targets
s=json.load(open('data/eval/table_probe_corpus_survey.json'))
for t in rank_targets(s): print(t)
"
```

- [ ] **Step 6: Author the evalset (manual, at home)**

Write 1–2 questions per target grid whose answer requires row-column alignment. One
row per line:

```json
{"id": "tbl-01", "question": "<answer lives in a specific cell>", "reference_answer": "<exact cell value>", "gold_contexts": ["<the table span holding the answer>"], "source_doc": "<file name>", "source_page": 8}
```

Rules:
- Answer must NOT be inferable from prose alone — it must require the table.
- `source_page` is the PDF page the grid is on. Required — gold matching keys on it.
- **Author as many as you can sustain.** n is the binding constraint on whether this
  probe can conclude anything — but across TABLES, not within them. Questions on one
  table are not independent draws (see the blocking precondition above); with 27
  structured candidate tables, ~2-3 questions each puts a real GO
  in reach; 10 questions cannot clear the sign test even if arm B wins every one it
  is scored on (10W/0L → p=0.002 does clear it, but 6W/1L does not).
- Do NOT copy rows from `prod_corpus_evalset_v3.jsonl` (INVALID vs real corpus).

- [ ] **Step 7: Validate** — `PYTHONPATH=. uv run python scripts/eval/validate_table_probe_evalset.py`
- [ ] **Step 8: Commit** — `feat(eval): table probe evalset + schema validator`

---

### Task 8: Chunk-integrity precondition (blocking)

Before scoring, confirm each gold row survives arm-B chunking with its header —
i.e. the chunker did not cut inside a rendered block. Questions whose row was cut
are **excluded from the comparison and reported as excluded**: they would measure
chunking, not tables.

Note this is a *check*, not a fix: `render_markdown` already emits blocks that fit
`SEARCH_CHUNK_MAX_TOKENS` (measured: 0 of 55 blocks over budget on the real
corpus), so a non-trivial exclusion count means something upstream regressed.

**Files:**
- Create: `scripts/eval/table_probe_chunk_integrity.py`
- Test: `tests/eval/test_table_probe_chunk_integrity.py`

**Interfaces:** `row_intact(chunks: list[str], gold_row: str) -> bool`;
`check_evalset(chunks_by_doc, evalset_rows) -> dict` →
`{"eligible": [...], "excluded": [...], "missing_provenance": [...]}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_table_probe_chunk_integrity.py
from scripts.eval.table_probe_chunk_integrity import check_evalset, row_intact

# `gold_contexts[0]` for a probe row is the single gold ROW, not the whole table:
# render_markdown deliberately splits a large table into header-repeating blocks,
# so "every row in one chunk" is the wrong question. The right one is "is this
# row still sitting under its header".
BLOCK = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
GOLD_ROW = "| 3 | 4 |"


def test_intact_when_a_chunk_holds_the_row_with_its_header():
    assert row_intact(["prose", BLOCK], GOLD_ROW)


def test_not_intact_when_the_row_was_cut_away_from_its_header():
    assert not row_intact(["| a | b |\n|---|---|\n| 1 | 2 |", GOLD_ROW], GOLD_ROW)


def test_not_intact_when_the_row_is_absent_entirely():
    assert not row_intact(["| a | b |\n|---|---|\n| 1 | 2 |"], GOLD_ROW)


def test_check_evalset_partitions_rows():
    rows = [{"id": "q1", "source_doc": "d.pdf", "source_page": 1, "gold_contexts": [GOLD_ROW]},
            {"id": "q2", "source_doc": "d.pdf", "source_page": 2, "gold_contexts": [GOLD_ROW]}]
    chunks = {("d.pdf", 1): [BLOCK], ("d.pdf", 2): ["| a | b |\n|---|---|", GOLD_ROW]}
    out = check_evalset(chunks, rows)
    assert out["eligible"] == ["q1"]
    assert out["excluded"] == ["q2"]


def test_a_row_missing_provenance_is_excluded_and_named():
    rows = [{"id": "q3", "source_doc": "", "source_page": None, "gold_contexts": [GOLD_ROW]}]
    out = check_evalset({}, rows)
    assert out["excluded"] == ["q3"] and out["missing_provenance"] == ["q3"]
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement**

```python
# scripts/eval/table_probe_chunk_integrity.py
"""Blocking precondition: did the gold table survive arm-B chunking intact?

A table cut mid-row cannot show arm B's advantage no matter how good the
extraction was, so such a question measures chunking, not tables. Those
questions are excluded from the paired comparison and counted in the report.

"Intact" is a property of a BLOCK, not of the whole table. `render_markdown`
deliberately splits a large table into header-repeating blocks sized to
`SEARCH_CHUNK_MAX_TOKENS`; a block boundary is a designed, alignment-preserving
split, and demanding that every row of a 60-row grid land in one 512-token chunk
would exclude every large table by construction — the outcome the exclusion
count exists to warn about.
"""
from __future__ import annotations


def _rows(table_md: str) -> list[str]:
    return [ln.strip() for ln in table_md.splitlines() if ln.strip().startswith("|")]


def _is_separator(line: str) -> bool:
    return set(line.replace("|", "").replace(" ", "")) <= {"-", ":"} and "-" in line


def row_intact(chunks: list[str], gold_row: str) -> bool:
    """True if one chunk holds the gold row AND the header naming its columns."""
    gold_row = gold_row.strip()
    if not gold_row.startswith("|"):
        return False
    for chunk in chunks:
        if gold_row not in chunk:
            continue
        lines = _rows(chunk)
        # The header survives with the row: block rendering repeats it, so a
        # chunk carrying a data row carries its column names unless the chunker
        # cut inside the block.
        if any(_is_separator(ln) for ln in lines) and lines[0] == gold_row.strip():
            continue  # row is itself the header line — no data context
        if any(_is_separator(ln) for ln in lines):
            return True
    return False


def check_evalset(chunks_by_doc: dict, rows: list[dict]) -> dict:
    """Partition evalset rows into eligible / excluded on chunk integrity.

    `rows` are the raw evalset dicts; `(source_doc, source_page)` is the key.
    Both fields are required by `validate_probe_row` and are carried through
    `EvalExample`, so the runner can build this without re-parsing the JSONL.
    """
    eligible, excluded, missing_provenance = [], [], []
    for row in rows:
        if not row.get("source_doc") or not row.get("source_page"):
            missing_provenance.append(row["id"])
            excluded.append(row["id"])
            continue
        chunks = chunks_by_doc.get((row["source_doc"], row["source_page"]), [])
        gold = (row.get("gold_contexts") or [""])[0]
        (eligible if row_intact(chunks, gold) else excluded).append(row["id"])
    return {
        "eligible": eligible,
        "excluded": excluded,
        "missing_provenance": missing_provenance,
    }
```

- [ ] **Step 4: Run test to verify it passes** (5 tests)

- [ ] **Step 5: Run against the real arm-B index (at home)**

Pull arm B's chunks keyed by `(source_doc, page)` and run `check_evalset`. If a
substantial share is excluded, `render_markdown`'s block packing is not reaching
the chunker (wrong `max_tokens`, or the flag off during ingest) — fix that before
scoring. Do not proceed with a shrunken eligible set, and do not let an empty
eligible list through: `paired_outcomes` treats `eligible=None` as "compare
everything", so an empty list must be passed as an empty *set*, never as `None`.

- [ ] **Step 6: Commit** — `feat(eval): table probe chunk-integrity precondition`

---

### Task 9: Paired A/B runner (`run_table_probe_ab.py`)

Ingest the **29 unique** docs twice (arm A flag-off, arm B flag-on) into isolated
indices, score every question under both arms, compare **per question** over the
eligible set. Modeled on `scripts/eval/run_vision_e2e_ab.py`.

**Files:**
- Create: `scripts/eval/run_table_probe_ab.py`
- Test: `tests/eval/test_run_table_probe_ab.py`
- Output: `data/eval/table_probe_arm_{a,b}.json`, `docs/eval/table_probe_<date>.md`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_run_table_probe_ab.py
import json
from scripts.eval.run_table_probe_ab import build_report


def _write(tmp_path, name, arm, scores, timeouts=()):
    p = tmp_path / name
    p.write_text(json.dumps({"arm": arm, "scores": scores,
                             "timeouts": list(timeouts)}))
    return str(p)


def test_report_declares_go_on_a_strong_paired_win(tmp_path):
    a = _write(tmp_path, "a.json", "a", {f"q{i}": 0.0 for i in range(9)})
    b = _write(tmp_path, "b.json", "b", {f"q{i}": 1.0 for i in range(9)})
    md = build_report(a, b, track="full",
                      eligible=[f"q{i}" for i in range(9)], excluded=[])
    # `"GO" in md` would also pass on **NO-GO**. Match the bolded verdict.
    assert "**GO**" in md and "9W/0L" in md


def test_report_declares_inconclusive_on_the_old_ten_question_gate(tmp_path):
    """One question flipping used to be a +0.10 GO. It must now read INCONCLUSIVE."""
    a = _write(tmp_path, "a.json", "a", {f"q{i}": 0.0 for i in range(10)})
    b = _write(tmp_path, "b.json", "b", {**{f"q{i}": 0.0 for i in range(10)}, "q0": 1.0})
    md = build_report(a, b, track="full",
                      eligible=[f"q{i}" for i in range(10)], excluded=[])
    assert "INCONCLUSIVE" in md


def test_report_states_exclusions_and_blind_spot(tmp_path):
    a = _write(tmp_path, "a.json", "a", {"q1": 0.0})
    b = _write(tmp_path, "b.json", "b", {"q1": 1.0})
    md = build_report(a, b, track="full", eligible=["q1"], excluded=["q2", "q3"])
    # A bare `"2" in md` is satisfied by the hardcoded "25%" scope prose, so it
    # is true for every input. Match the row itself.
    assert "| excluded (gold row cut by chunking) | 2 |" in md
    assert "ocr" in md.lower()  # 25% blind spot must be stated


def test_an_empty_eligible_list_does_not_silently_compare_everything(tmp_path):
    """`set(x) if x else None` turned "nothing survived" into "compare all"."""
    a = _write(tmp_path, "a.json", "a", {"q1": 0.0, "q2": 0.0})
    b = _write(tmp_path, "b.json", "b", {"q1": 1.0, "q2": 1.0})
    md = build_report(a, b, track="full", eligible=[], excluded=["q1", "q2"])
    assert "0W/0L" in md
    assert "INCONCLUSIVE" in md


def test_timeouts_are_reported_and_never_scored_zero(tmp_path):
    a = _write(tmp_path, "a.json", "a", {"q1": 1.0, "q2": 1.0})
    b = _write(tmp_path, "b.json", "b", {"q1": 1.0}, timeouts=["q2"])
    md = build_report(a, b, track="full", eligible=["q1", "q2"], excluded=[])
    assert "agent timeouts (never scored) | 1" in md
    assert "dropped (scored by only one arm) | 1" in md
    assert "0W/0L" in md   # q2 is absent, not a loss for B


def test_an_offline_report_labels_itself_as_weaker_evidence(tmp_path):
    """plan/spec both require the label; build_report had no track at all."""
    a = _write(tmp_path, "a.json", "a", {"q1": 0.0})
    b = _write(tmp_path, "b.json", "b", {"q1": 0.0})
    md = build_report(a, b, track="offline", eligible=["q1"], excluded=[])
    assert "OFFLINE" in md
    assert "does NOT close the table question" in md
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Write the runner**

```python
# scripts/eval/run_table_probe_ab.py
"""Table-data probe A/B — arm A (flat) vs arm B (gated PDF_PRESERVE_TABLES).

  PYTHONPATH=. uv run python scripts/eval/run_table_probe_ab.py --arm a --corpus <DIR>
  PYTHONPATH=. uv run python scripts/eval/run_table_probe_ab.py --arm b --corpus <DIR>
  PYTHONPATH=. uv run python scripts/eval/run_table_probe_ab.py --track full \
      --compare data/eval/table_probe_arm_a.json data/eval/table_probe_arm_b.json

<DIR> is the deduplicated corpus produced by
`table_probe_corpus_survey.py --dedupe-dir`, never data/originals.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.run_ablation import wipe_corpus_db, wipe_corpus_indices
from src.agentrag.common.build_info import build_info, format_build_banner
from scripts.eval.table_probe_lib import (
    corpus_matches, decide_paired, mean_delta, paired_outcomes,
)

EVAL_PATH = "data/eval/table_probe_evalset.jsonl"
SURVEY_PATH = "data/eval/table_probe_corpus_survey.json"


def check_corpus() -> None:
    """Refuse to run when the evalset was authored against a different corpus.

    The (doc, page) gold is only valid against the corpus snapshot it was written
    on; gold pointing at a page that moved scores 0.00 and reads as a real
    result. Must be called BEFORE the wipes — after them there is nothing left to
    protect.
    """
    survey = json.load(open(SURVEY_PATH))
    with open(EVAL_PATH, encoding="utf-8") as fh:
        stamp = json.loads(fh.readline() or "{}").get("corpus_docs_sha")
    ok, why = corpus_matches(stamp, survey.get("corpus_docs_sha"))
    print(f"corpus check: {why}")
    if not ok:
        raise SystemExit("corpus mismatch — re-author the evalset or re-run the survey")


async def _run_arm(arm: str, corpus: str, out_path: str) -> None:
    from src.agentrag.agent.factory import get_agent_service
    from src.agentrag.config import settings
    from src.agentrag.eval.benchmark_datasets import load_local_jsonl
    from src.agentrag.eval.correctness_judge import score_correctness
    from src.agentrag.ingestion.pipeline import ingest_folder
    from src.agentrag.services.llm_gateway import LLMGateway

    settings.PDF_PRESERVE_TABLES = (arm == "b")
    # Stamp the running code into the arm result. On a `-f` compose path the
    # ./src bind mount is not applied, so a stale image produces a complete,
    # plausible arm with no error; the only defence is that every artefact
    # carries the identity of the code that made it.
    print(format_build_banner())
    # Async ingest returns before anything is indexed; force sync.
    settings.STRUCTMEM_INGEST_MODE = "sync"
    print(f"ARM={arm} PDF_PRESERVE_TABLES={settings.PDF_PRESERVE_TABLES}")

    # Arm isolation is by SEQUENCE, not by index name. `get_agent_service()`
    # takes no arguments and `ElasticsearchStore.__init__` reads
    # settings.ELASTICSEARCH_INDEX_NAME at construction, so there is no
    # supported way to point ingest and retrieval at a per-arm index. The wipe
    # runs in `__main__` BEFORE asyncio.run (see below) — it cannot run here.
    await ingest_folder(corpus)          # MUST be awaited: it is `async def`

    # 2) Score each probe question against this arm.
    examples = load_local_jsonl(EVAL_PATH, n=1000)
    agent, gateway = get_agent_service(), LLMGateway()
    scores, rows, timeouts = {}, [], []
    for ex in examples:
        out = await agent.chat(question=ex.question, document_title=None,
                               conversation_id=f"tbl-{arm}-{ex.id}")
        if out.get("timed_out"):
            # NOT scored 0.0. A timeout says the stack was slow, not that the
            # answer was wrong, and arm B — longer table text, larger packed
            # context — times out more often. Scoring it 0.0 turns the sign test
            # into a latency test. run_vision_e2e_ab.py:36-39 does the same.
            timeouts.append(ex.id)
            rows.append({"id": ex.id, "score": None, "timed_out": True, "answer": ""})
            print(f"  {ex.id} TIMEOUT (excluded)")
            continue
        ans = out.get("answer") or ""
        e = await score_correctness(ex.question, ans, ex.reference_answer,
                                    "\n".join(ex.gold_contexts), gateway)
        scores[ex.id] = e.mean
        rows.append({"id": ex.id, "score": e.mean, "timed_out": False,
                     "answer": ans[:300],
                     "source_doc": ex.source_doc, "source_page": ex.source_page})
        print(f"  {ex.id} score={e.mean}")

    json.dump({"arm": arm, "build": build_info(), "scores": scores,
               "rows": rows, "timeouts": timeouts},
              open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"ARM={arm} scored={len(scores)}/{len(rows)} timeouts={len(timeouts)} "
          f"-> {out_path}")


def build_report(a_path: str, b_path: str, track: str,
                 eligible=None, excluded=None) -> str:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    a_s, b_s = a["scores"], b["scores"]
    excluded = excluded or []
    # `if eligible` would treat an EMPTY eligible list as "no filter" and compare
    # every question — including every one the integrity gate threw out. That is
    # the opposite of what an empty list means.
    elig = set(eligible) if eligible is not None else None
    outcomes = paired_outcomes(a_s, b_s, eligible=elig)
    d = decide_paired(outcomes)
    delta = mean_delta(a_s, b_s, eligible=elig)   # same population as the decision
    # Union, not sum: a slow question is slow under BOTH arms, and every other
    # row in the table counts questions, so summing would roughly double it.
    n_timeout = len(set(a.get("timeouts", [])) | set(b.get("timeouts", [])))

    offline = track == "offline"
    verdict = {
        "GO": "GO -> write a follow-up spec for a real table-aware build.",
        "NO-GO": "NO-GO -> keep flat-text ingest; record the number so this is "
                 "not re-litigated on a hunch.",
        "INCONCLUSIVE": "INCONCLUSIVE -> do NOT build. Either author more "
                        "questions and re-run, or close it explicitly. Do not "
                        "read this as a GO.",
    }[d["decision"]]
    if offline:
        verdict = (
            "**OFFLINE track — retrieval only, no answer-gen, no judge.** "
            + verdict
            + " An OFFLINE result is weaker evidence than a FULL one and is "
              "biased against arm B (pipe-table markdown may embed worse with "
              "e5 than prose even when it carries more information). An OFFLINE "
              "NO-GO does NOT close the table question."
        )

    return "\n".join([
        "# Table-Data Probe — paired A/B",
        "",
        f"Track: **{track.upper()}**. Eval set: `{EVAL_PATH}`. "
        "Arms: A = flat text, B = gated `PDF_PRESERVE_TABLES`.",
        "",
        f"Arm A built by: `{a.get('build', {}).get('image_git_sha', 'unstamped')}` "
        f"source `{a.get('build', {}).get('running_source_sha', '?')}`; "
        f"arm B: `{b.get('build', {}).get('image_git_sha', 'unstamped')}` "
        f"source `{b.get('build', {}).get('running_source_sha', '?')}`."
        + ("  **The two arms did not run the same source — this comparison is "
           "not valid.**"
           if a.get("build", {}).get("running_source_sha")
           != b.get("build", {}).get("running_source_sha") else ""),
        "",
        "| outcome | n |",
        "|---|---|",
        f"| B better | {outcomes['n_wins']} |",
        f"| A better | {outcomes['n_losses']} |",
        f"| tie | {outcomes['n_ties']} |",
        f"| **compared** | **{outcomes['n_compared']}** |",
        f"| excluded (gold row cut by chunking) | {len(excluded)} |",
        f"| excluded (ineligible, scored by both arms) | {outcomes['n_ineligible']} |",
        f"| dropped (scored by only one arm) | {outcomes['n_missing']} |",
        f"| agent timeouts (never scored) | {n_timeout} |",
        "",
        "The last three rows overlap: a question that timed out in one arm is "
        "counted both as a timeout and as dropped. Only **compared** is disjoint "
        "from them.",
        "",
        f"**{d['decision']}** — {d['reason']} "
        f"({outcomes['n_wins']}W/{outcomes['n_losses']}L, sign test p={d['p_value']})",
        "",
        f"Mean delta (B-A) over the compared set = {delta} — colour only; at this "
        "n it is inside measured A/B noise (~0.3) and does not drive the decision.",
        "",
        "**Scope limits.** 25% of corpus pages have no text layer and take the "
        "OCR/vision path, where `find_tables()` finds nothing and arm B is "
        "byte-identical to arm A — scanned tables are outside what this probe can "
        "measure. The table append runs *after* the OCR block precisely so arm B "
        "cannot change that routing. Arm B only converts detections passing "
        "`is_safe_to_markdown` (40 of 171, 23%). Corpus deduplicated to 29 unique "
        "documents before ingest.",
        "",
        verdict,
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["a", "b"])
    ap.add_argument("--corpus", help="deduplicated probe corpus (required with --arm)")
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    # No default and no global `required=True`: `--track` is only read on the
    # --compare path, and making it globally required breaks the two --arm
    # commands in this file's own usage block (argparse exits 2 before any code
    # runs). Defaulting it to "full" is worse than omitting it — an OFFLINE run
    # would label itself FULL — so --compare demands it explicitly below.
    ap.add_argument("--track", choices=["full", "offline"],
                    help="must match what table_probe_preflight.py printed; "
                         "required with --compare")
    ap.add_argument("--eligible", help="json file from table_probe_chunk_integrity")
    args = ap.parse_args()

    if args.compare:
        if not args.track:
            ap.error("--track full|offline is required with --compare — the "
                     "report cannot infer it, and mislabelling an OFFLINE run "
                     "as FULL is worse than not labelling it")
        elig = exc = None
        if args.eligible:
            gate = json.load(open(args.eligible))
            elig, exc = gate["eligible"], gate["excluded"]
        md = build_report(*args.compare, track=args.track,
                          eligible=elig, excluded=exc)
        out = f"docs/eval/table_probe_{date.today().isoformat()}.md"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(md, encoding="utf-8")
        print(md)
        print(f"\n-> {out}")
    elif args.arm:
        if not args.corpus:
            ap.error("--corpus is required with --arm")
        # Validate the corpus BEFORE anything destructive: the wipes below
        # delete four ES indices and the Postgres corpus, so a guard placed after
        # them cannot protect what it is guarding.
        check_corpus()
        # Wipe BEFORE asyncio.run, from sync context. `wipe_corpus_db` is
        # implemented as `asyncio.run(_wipe())` inside a bare `except Exception`
        # that only prints, so calling it from inside a coroutine raises
        # "asyncio.run() cannot be called from a running event loop", swallows
        # it, and returns cleanly — Postgres survives, every document re-ingests
        # as "skipped", `index_segments` never runs, and the arm is scored
        # against an EMPTY index. run_ablation.py calls it from sync `main()`
        # for exactly this reason.
        #
        # The PG wipe is what makes the re-ingest real: `save_document_and_segments`
        # dedupes on (source_id, content_hash). Setting
        # `settings.UPLOAD_DEDUPE_BY_HASH = False` does NOT help here — its only
        # reader is the HTTP upload router, never `ingest_folder`.
        wipe_corpus_indices()
        wipe_corpus_db()
        asyncio.run(_run_arm(args.arm, args.corpus,
                             f"data/eval/table_probe_arm_{args.arm}.json"))
    else:
        ap.error("pass --arm a|b --corpus DIR, or --compare A_JSON B_JSON")
```

> **Run the arms strictly in sequence**: `--arm a` to completion, then `--arm b`.
> Each run wipes the corpus indices and Postgres rows before ingesting, so
> running them concurrently, or scoring after both ingests, compares an arm
> against the other arm's data.

- [ ] **Step 4: Run test to verify it passes** (6 tests)

- [ ] **Step 5: Full A/B run (at home)**

```bash
# 0. dedup (writes the unique-docs dir every later step consumes) + preflight
PYTHONPATH=. uv run python scripts/eval/table_probe_corpus_survey.py \
    --corpus data/originals \
    --json data/eval/table_probe_corpus_survey.json \
    --unique-list data/eval/table_probe_unique_docs.txt \
    --dedupe-dir data/eval/table_probe_corpus
PYTHONPATH=. uv run python scripts/eval/table_probe_preflight.py   # note the TRACK

# 1. both arms, STRICTLY IN SEQUENCE (each wipes ES + Postgres, then ingests)
PYTHONPATH=. uv run python scripts/eval/run_table_probe_ab.py --arm a \
    --corpus data/eval/table_probe_corpus
PYTHONPATH=. uv run python scripts/eval/run_table_probe_ab.py --arm b \
    --corpus data/eval/table_probe_corpus

# 2. chunk-integrity gate, then compare (pass the track preflight reported)
PYTHONPATH=. uv run python scripts/eval/table_probe_chunk_integrity.py > data/eval/table_probe_eligible.json
PYTHONPATH=. uv run python scripts/eval/run_table_probe_ab.py --track full \
    --compare data/eval/table_probe_arm_a.json data/eval/table_probe_arm_b.json \
    --eligible data/eval/table_probe_eligible.json
```

> `--corpus` must be the **dedupe dir**, not `data/originals`. The raw directory
> holds 116 files for 29 documents; ingesting 4–7 copies of each lets duplicate
> near-identical chunks occupy top-k and swamp the arm difference.

- [ ] **Step 6: Commit runner** — `feat(eval): paired table-probe A/B runner`
- [ ] **Step 7: Commit result** — `docs(eval): table-data probe result — <GO|NO-GO|INCONCLUSIVE>`

---

## OFFLINE-track variant (Task 9 fallback, when preflight = offline)

If preflight returns `offline`, skip answer-gen. Per probe question, retrieve top-k
from each arm's index and record whether the gold table chunk is present — **matched on
`(source_doc, source_page)`, never on chunk text**, because the arms produce different
text for the same table. Feed the per-question hit/miss into the same
`paired_outcomes` / `decide_paired` path; the decision rule is unchanged.

`k` must equal production `top_k` — read it from config, do not hard-code it:
`settings.RETRIEVAL_TOP_K` (10), with `settings.AGENT_TOOL_TOP_K` (30) as the pool the
agent's retrieval tool actually requests. Rev 1 listed "confirm production top_k" as an
open question; rev 2 deleted the question and wrote `8`, which matches neither.

Lower fidelity than FULL, and biased against arm B: pipe-table markdown may embed worse
with e5 than prose even when it carries more information. An OFFLINE NO-GO is therefore
weaker evidence than a FULL NO-GO and **must be labelled as such** — pass
`--track offline` so `build_report` emits the label. The report has no way to infer the
track on its own.

---

## Self-Review

**Spec coverage (rev 3):**
- §3 Step 0a dedup → Task 2 ✓ (done; emits the unique-docs list + dedupe dir)
- §3 Step 0b preflight → Task 4 ✓
- §3 Step 1 target tables → `rank_targets`, Task 3 ✓ (done) + Task 7 Step 5
- §3 Step 2 author evalset w/ `(doc,page)` → Task 7 ✓
- §3 Step 3 gated arm B → Task 5 ✓ (gate + renderer done in Task 1; append runs
  after the OCR block so routing is unchanged)
- §3 Step 4 chunk-integrity precondition → `render_markdown` block packing (Task 1,
  measured 0/55 blocks over budget) + Task 8 (check) ✓; Task 6 withdrawn
- §3 Step 5 paired A/B, FULL + OFFLINE → Task 9 + OFFLINE variant ✓ (`--track` is
  required; the report cannot infer it)
- §4 paired decision rule → `decide_paired`, Task 3 ✓ (done); surfaced in Task 9 report ✓
- §6 risks → gate + renderer (Task 1), block packing (Task 1), `(doc,page)` match
  (Tasks 3/8/9 + `EvalExample` provenance), dedup (Task 2), OCR blind spot stated in
  report *and* structurally prevented by the append position (Task 5/9) ✓

**Rev-1 flaws and where each is closed:**

| flaw | closed by |
|---|---|
| ungated `to_markdown()` corrupts 77% of detections | Task 1 gate + `render_markdown`, wired in Task 5 |
| `+0.10` mean gate = one question at n=10 | Task 3 `decide_paired` + regression test |
| chunker splits tables before retrieval sees them | `render_markdown` block packing (Task 1), Task 8 check; Task 6 withdrawn |
| gold matched on chunk text across arms | `source_page` required (Task 3/7), Task 8/9 |
| 4× duplicate corpus counted as 116 docs | Task 2 dedup; unique-docs-only constraint |
| doc-level selection concentrates on one file | `rank_targets` returns tables |

**Rev-2 flaws and where each is closed (found by code review 2026-09-06):**

| flaw | closed by |
|---|---|
| gate classified `extract()`, arm B would emit `to_markdown()` | `render_markdown` (Task 1); 0/40 invented headers, measured |
| `max(len(row))` counted `None` cells as columns | `single_column` kind (Task 1) |
| `\d`-anywhere numeric test promoted prose to `real_data` | digit-density test (Task 1) |
| table append before the OCR gate changed OCR routing | append moved after the OCR block (Task 5) + a test that pins it |
| Task 6 chunker branch `NameError`s; its test unsatisfiable | Task 6 withdrawn; renderer packs to the chunk budget |
| `ingest_folder(corpus, index=...)` — wrong kwarg, never awaited | awaited, no kwarg; arms isolated by sequence + wipe (Task 9) |
| `get_agent_service(index=...)` — takes no arguments | removed; documented why per-arm indices are not available |
| content-hash dedupe would no-op arm B's re-ingest | `UPLOAD_DEDUPE_BY_HASH=False` + `STRUCTMEM_INGEST_MODE=sync` + wipe (Task 9) |
| timeouts scored 0.0; the `e.mean is not None` guard was dead | timeouts excluded and counted (Task 9) |
| `paired_outcomes` silently intersected the arms | `n_missing` / `n_ineligible` reported (Task 3) |
| `eligible=set(x) if x else None` — empty list compared everything | `eligible is not None` (Task 9) + a test |
| `mean_delta` averaged a different population than the decision | `eligible` parameter (Task 3) |
| `build_report` had no track; an OFFLINE NO-GO read as unqualified | `--track` required, OFFLINE label emitted (Task 9) |
| `EvalExample` dropped `source_doc`/`source_page` | both carried through `load_local_jsonl` |
| survey emitted only the unique-doc *count*, never the list | `--unique-list` + `--dedupe-dir` (Task 2) |
| one malformed page aborted the whole survey; `doc.close()` leaked | every call guarded, `close()` in `finally` (Task 2) |
| `rank_targets` put 8 of 10 targets in one document | round-robin + `max_tokens` (Task 3) |
| `k=8` hard-coded as "production top_k" | read `settings.RETRIEVAL_TOP_K` |
| gate had no CI coverage (`tests/ingestion` ignored wholesale) | explicit `make test-fast` / CI step |

**Placeholder scan:** no TBD/TODO in requirements. One `VERIFY AT HOME` note remains,
on the ES/TEI settings attribute names in Task 4; it ships an import-safe `getattr`
default. Task 9's two rev-2 notes are resolved — the real signatures are inlined.

**Type consistency:** `arm` is `"a"|"b"` throughout; `scores` is `dict[qid, float]` in
both the arm output and `paired_outcomes` input (timed-out questions are absent from it,
never present as `0.0`); `source_page` is a positive `int` everywhere; `gold_contexts` is
a list throughout; `track` is `"full"|"offline"` from preflight to report.
