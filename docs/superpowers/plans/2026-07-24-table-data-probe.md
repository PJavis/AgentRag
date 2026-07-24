# Table-Data Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a disposable measurement probe that produces one go/no-go lift number for whether preserving PDF table structure at ingest improves answers.

**Architecture:** A small pure-function library holds all testable logic (track selection, doc ranking, arm naming, evalset validation, lift computation). Thin script shells wrap it with real I/O (ES/TEI pings, PyMuPDF, ingest, agent.chat). A throwaway `PDF_PRESERVE_TABLES` flag adds a table-preserving arm to the existing PDF parser. The A/B runner mirrors the existing `scripts/eval/run_vision_e2e_ab.py` two-arm pattern.

**Tech Stack:** Python, pytest, PyMuPDF (`fitz`), existing agentrag eval harness (`load_local_jsonl`, `score_correctness`, `get_agent_service`), Elasticsearch, TEI embedder.

## Global Constraints

- Probe is disposable measurement code — no production default-on behavior. `PDF_PRESERVE_TABLES` defaults `False`, copied verbatim from spec §3 Step 3.
- No new heavy deps: NO MinerU / docling / camelot / pdfplumber. Table preservation uses PyMuPDF built-in `page.find_tables()` only (spec §2 Non-Goals).
- Go/no-go gate = **+0.10** lift on the primary metric (spec §3 Step 5, §4).
- Evalset size = **8–12** hand-authored table-dependent questions (spec §3 Step 2).
- Judge must be grounded: pass gold context to `score_correctness` (spec §6).
- Do NOT reuse `prod_corpus_evalset_v3.jsonl` gold — flagged INVALID vs real corpus (spec §3 Step 2, §6).
- Evalset row schema matches `load_local_jsonl`: `id`, `question`, `reference_answer`, `gold_contexts` (list) — same shape as `data/eval/vision_evalset_2026-07-19.jsonl`.
- Test command: `PYTHONPATH=. uv run pytest <path> -v`.

---

### Task 1: Pure-function library (`table_probe_lib.py`)

All deterministic logic lives here so every downstream script has a tested core. No I/O.

**Files:**
- Create: `scripts/eval/table_probe_lib.py`
- Test: `tests/eval/test_table_probe_lib.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `decide_track(es_ok: bool, tei_ok: bool, judge_key: str | None) -> tuple[str, str]` → `("full"|"offline", reason)`
  - `rank_docs(counts: dict[str, int], top_n: int) -> list[tuple[str, int]]`
  - `arm_index_name(base: str, arm: str) -> str`
  - `validate_probe_row(row: dict) -> list[str]` → list of error strings (empty = valid)
  - `compute_lift(b_mean: float, a_mean: float, gate: float = 0.10) -> dict` → `{"lift": float, "gate": float, "decision": "GO"|"NO-GO"}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_table_probe_lib.py
import pytest
from scripts.eval.table_probe_lib import (
    decide_track, rank_docs, arm_index_name, validate_probe_row, compute_lift,
)


def test_decide_track_full_when_all_up():
    track, _ = decide_track(True, True, "sk-key")
    assert track == "full"


@pytest.mark.parametrize("es,tei,key", [
    (False, True, "k"), (True, False, "k"), (True, True, None), (True, True, ""),
])
def test_decide_track_offline_when_any_missing(es, tei, key):
    track, reason = decide_track(es, tei, key)
    assert track == "offline"
    assert reason  # non-empty explanation


def test_rank_docs_orders_by_count_desc_and_truncates():
    counts = {"a.pdf": 1, "b.pdf": 9, "c.pdf": 5}
    assert rank_docs(counts, 2) == [("b.pdf", 9), ("c.pdf", 5)]


def test_rank_docs_drops_zero_table_docs():
    assert rank_docs({"a.pdf": 0, "b.pdf": 3}, 5) == [("b.pdf", 3)]


def test_arm_index_name_isolates_arms():
    a, b = arm_index_name("probe", "a"), arm_index_name("probe", "b")
    assert a != b
    assert a == "probe_arm_a" and b == "probe_arm_b"


def test_arm_index_name_rejects_bad_arm():
    with pytest.raises(ValueError):
        arm_index_name("probe", "c")


def test_validate_probe_row_accepts_good_row():
    row = {"id": "q1", "question": "Q?", "reference_answer": "A",
           "gold_contexts": ["| col | val |"]}
    assert validate_probe_row(row) == []


def test_validate_probe_row_flags_missing_and_empty_gold():
    errs = validate_probe_row({"id": "q1", "question": "Q?", "gold_contexts": []})
    assert any("reference_answer" in e for e in errs)
    assert any("gold_contexts" in e for e in errs)


def test_compute_lift_go_at_threshold():
    r = compute_lift(0.80, 0.70)
    assert r["lift"] == pytest.approx(0.10)
    assert r["decision"] == "GO"


def test_compute_lift_no_go_below_threshold():
    assert compute_lift(0.75, 0.70)["decision"] == "NO-GO"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. uv run pytest tests/eval/test_table_probe_lib.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.eval.table_probe_lib'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/eval/table_probe_lib.py
"""Pure, I/O-free helpers for the table-data probe. Tested in isolation."""
from __future__ import annotations

_REQUIRED_ROW_FIELDS = ("id", "question", "reference_answer", "gold_contexts")


def decide_track(es_ok: bool, tei_ok: bool, judge_key: str | None) -> tuple[str, str]:
    """FULL only if ES + TEI up and a judge key is present; else OFFLINE."""
    missing = []
    if not es_ok:
        missing.append("elasticsearch")
    if not tei_ok:
        missing.append("tei")
    if not judge_key:
        missing.append("judge_key")
    if missing:
        return "offline", "missing: " + ", ".join(missing)
    return "full", "es+tei+judge_key all available"


def rank_docs(counts: dict[str, int], top_n: int) -> list[tuple[str, int]]:
    """Docs with >=1 table, sorted by table count desc, truncated to top_n."""
    ranked = sorted(
        ((doc, n) for doc, n in counts.items() if n > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return ranked[:top_n]


def arm_index_name(base: str, arm: str) -> str:
    """Distinct index name per arm so retrieval never mixes control/treatment."""
    if arm not in ("a", "b"):
        raise ValueError(f"arm must be 'a' or 'b', got {arm!r}")
    return f"{base}_arm_{arm}"


def validate_probe_row(row: dict) -> list[str]:
    """Return a list of schema errors; empty list means the row is valid."""
    errs: list[str] = []
    for field in _REQUIRED_ROW_FIELDS:
        if field not in row or row[field] in (None, "", []):
            errs.append(f"missing/empty field: {field}")
    gold = row.get("gold_contexts")
    if gold is not None and not isinstance(gold, list):
        errs.append("gold_contexts must be a list")
    return errs


def compute_lift(b_mean: float, a_mean: float, gate: float = 0.10) -> dict:
    """lift = arm B (table-preserved) minus arm A (flat); GO if lift >= gate."""
    lift = round(b_mean - a_mean, 4)
    return {"lift": lift, "gate": gate, "decision": "GO" if lift >= gate else "NO-GO"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/eval/test_table_probe_lib.py -v`
Expected: PASS (all 10 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/table_probe_lib.py tests/eval/test_table_probe_lib.py
git commit -m "feat(eval): pure helper lib for table-data probe"
```

---

### Task 2: Preflight track selector (`table_probe_preflight.py`)

Thin shell: ping real ES + TEI, check judge key, print the track via `decide_track`.

**Files:**
- Create: `scripts/eval/table_probe_preflight.py`
- Test: `tests/eval/test_table_probe_preflight.py`

**Interfaces:**
- Consumes: `decide_track` (Task 1).
- Produces: `preflight() -> tuple[str, str]` (track, reason); CLI prints `TRACK=<track> REASON=<reason>`.

- [ ] **Step 1: Write the failing test**

The pings are patched so the test never touches the network.

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

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/eval/test_table_probe_preflight.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.eval.table_probe_preflight'`

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

> NOTE — verify at home: the exact settings attribute names for the ES and TEI
> URLs (`ELASTICSEARCH_URL` / `TEI_URL` / `EMBEDDING_BASE_URL`). Open
> `src/agentrag/config.py` and `src/agentrag/ingestion/stores/elasticsearch_store.py`
> and adjust the `getattr` fallbacks to the real names. The `getattr(..., default)`
> form keeps this import-safe even if the names differ.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/eval/test_table_probe_preflight.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Manual smoke check**

Run: `PYTHONPATH=. uv run python scripts/eval/table_probe_preflight.py`
Expected: prints one line `TRACK=full REASON=...` or `TRACK=offline REASON=missing: ...`

- [ ] **Step 6: Commit**

```bash
git add scripts/eval/table_probe_preflight.py tests/eval/test_table_probe_preflight.py
git commit -m "feat(eval): table-probe preflight track selector"
```

---

### Task 3: Table-heavy doc finder (`table_probe_find_docs.py`)

Count tables per PDF via `page.find_tables()`, rank with `rank_docs`, print candidates.

**Files:**
- Create: `scripts/eval/table_probe_find_docs.py`
- Test: `tests/eval/test_table_probe_find_docs.py`

**Interfaces:**
- Consumes: `rank_docs` (Task 1).
- Produces: `count_pdf_tables(path: str) -> int`; `scan(corpus_dir: str, top_n: int) -> list[tuple[str, int]]`.

- [ ] **Step 1: Write the failing test**

`fitz` is mocked so no real PDF is needed. Each fake page returns a fake `find_tables()` result whose `.tables` list length is the table count.

```python
# tests/eval/test_table_probe_find_docs.py
from unittest.mock import MagicMock, patch
from scripts.eval import table_probe_find_docs as fd


def _fake_doc(tables_per_page):
    pages = []
    for n in tables_per_page:
        page = MagicMock()
        page.find_tables.return_value = MagicMock(tables=[object()] * n)
        pages.append(page)
    doc = MagicMock()
    doc.__iter__.return_value = iter(pages)
    return doc


def test_count_pdf_tables_sums_pages():
    with patch.object(fd, "_open", return_value=_fake_doc([2, 0, 3])):
        assert fd.count_pdf_tables("x.pdf") == 5


def test_count_pdf_tables_returns_zero_on_error():
    with patch.object(fd, "_open", side_effect=RuntimeError("bad pdf")):
        assert fd.count_pdf_tables("x.pdf") == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/eval/test_table_probe_find_docs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.eval.table_probe_find_docs'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/eval/table_probe_find_docs.py
"""Rank corpus PDFs by table density to pick probe docs."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.eval.table_probe_lib import rank_docs


def _open(path: str):
    import fitz  # PyMuPDF
    return fitz.open(path)


def count_pdf_tables(path: str) -> int:
    """Total detected tables across all pages; 0 if the file can't be read."""
    try:
        doc = _open(path)
    except Exception:
        return 0
    total = 0
    for page in doc:
        try:
            total += len(page.find_tables().tables)
        except Exception:
            continue
    return total


def scan(corpus_dir: str, top_n: int) -> list[tuple[str, int]]:
    counts = {
        str(p): count_pdf_tables(str(p))
        for p in Path(corpus_dir).rglob("*.pdf")
    }
    return rank_docs(counts, top_n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="dir of source PDFs")
    ap.add_argument("--top-n", type=int, default=8)
    args = ap.parse_args()
    for doc, n in scan(args.corpus, args.top_n):
        print(f"{n}\t{doc}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/eval/test_table_probe_find_docs.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Manual smoke check (at home, needs real corpus)**

Run: `PYTHONPATH=. uv run python scripts/eval/table_probe_find_docs.py --corpus <REAL_CORPUS_DIR> --top-n 8`
Expected: up to 8 lines `<count>\t<path>`, highest table count first. Pick 5–8 docs with numeric tables for the evalset.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval/table_probe_find_docs.py tests/eval/test_table_probe_find_docs.py
git commit -m "feat(eval): table-heavy PDF finder for probe"
```

---

### Task 4: `PDF_PRESERVE_TABLES` flag + parser arm

Add a throwaway, default-off flag and an additive table-markdown branch in the PDF parser. Arm B keeps the flat text AND appends each detected table as GFM markdown.

**Files:**
- Modify: `src/agentrag/config.py` (add flag near the retrieval/ingest flags block, ~line 113)
- Modify: `src/agentrag/ingestion/parsers/pdf_parser.py` (add helper + branch after line 83)
- Test: `tests/ingestion/test_pdf_preserve_tables.py`

**Interfaces:**
- Consumes: `settings.PDF_PRESERVE_TABLES`.
- Produces: module-level `_append_table_markdown(page, text: str) -> str` in `pdf_parser.py`.

- [ ] **Step 1: Add the config flag**

In `src/agentrag/config.py`, immediately after `RETRIEVAL_RERANK_ENABLED: bool = False` (line 113), add:

```python
    #: PROBE-ONLY (throwaway, 2026-07-24). When True, PDFParser appends each
    #: page's detected tables as GFM markdown (PyMuPDF find_tables().to_markdown())
    #: to the flattened page text. Arm B of the table-data probe. Default OFF.
    PDF_PRESERVE_TABLES: bool = False
```

- [ ] **Step 2: Write the failing test**

`fitz` isn't invoked — the test calls the pure helper with a mocked page whose `find_tables()` yields tables exposing `.to_markdown()`.

```python
# tests/ingestion/test_pdf_preserve_tables.py
from unittest.mock import MagicMock
from src.agentrag.ingestion.parsers.pdf_parser import _append_table_markdown


def _page_with_tables(markdowns):
    tabs = []
    for md in markdowns:
        t = MagicMock()
        t.to_markdown.return_value = md
        tabs.append(t)
    page = MagicMock()
    page.find_tables.return_value = MagicMock(tables=tabs)
    return page


def test_append_adds_table_markdown_after_text():
    page = _page_with_tables(["| a | b |\n| - | - |\n| 1 | 2 |"])
    out = _append_table_markdown(page, "flat page text")
    assert out.startswith("flat page text")
    assert "| a | b |" in out


def test_append_noop_when_no_tables():
    page = _page_with_tables([])
    assert _append_table_markdown(page, "flat page text") == "flat page text"


def test_append_survives_find_tables_error():
    page = MagicMock()
    page.find_tables.side_effect = RuntimeError("no table layer")
    assert _append_table_markdown(page, "flat page text") == "flat page text"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/ingestion/test_pdf_preserve_tables.py -v`
Expected: FAIL — `ImportError: cannot import name '_append_table_markdown'`

- [ ] **Step 4: Add the helper + branch in `pdf_parser.py`**

Add this module-level function above `class PDFParser` (after the `_ocr_tesseract` helper, ~line 46):

```python
def _append_table_markdown(page, text: str) -> str:
    """Probe arm B: append detected tables as GFM markdown to the page text.

    Additive — the flat text is preserved; tables are appended so retrieval
    sees both. Any detection error is swallowed (returns text unchanged).
    """
    try:
        tabs = page.find_tables().tables
    except Exception:
        return text
    blocks = []
    for t in tabs:
        try:
            md = t.to_markdown()
        except Exception:
            continue
        if md and md.strip():
            blocks.append(md.strip())
    if not blocks:
        return text
    return text + "\n\n" + "\n\n".join(blocks)
```

Then, in `PDFParser.parse`, immediately after line 83 (`text = page.get_text("text", sort=True)`), insert:

```python
            if settings.PDF_PRESERVE_TABLES:
                text = _append_table_markdown(page, text)
```

(The existing `stripped = text.strip()` on line 84 then picks up the augmented text, so tables flow into `parts`/`page_data` unchanged.)

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/ingestion/test_pdf_preserve_tables.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Verify the default-off path is untouched**

Run: `PYTHONPATH=. uv run pytest tests/ingestion/ -v`
Expected: PASS — existing PDF parser tests still green (flag defaults False → no behavior change).

- [ ] **Step 7: Commit**

```bash
git add src/agentrag/config.py src/agentrag/ingestion/parsers/pdf_parser.py tests/ingestion/test_pdf_preserve_tables.py
git commit -m "feat(ingest): PDF_PRESERVE_TABLES probe arm (default off)"
```

---

### Task 5: Evalset validator + hand-authored questions

The 8–12 questions are authored by hand against the docs picked in Task 3. This task ships a validator so the file can't drift from the schema, plus the authored file itself.

**Files:**
- Create: `scripts/eval/validate_table_probe_evalset.py`
- Create: `data/eval/table_probe_evalset.jsonl` (hand-authored)
- Test: `tests/eval/test_validate_table_probe_evalset.py`

**Interfaces:**
- Consumes: `validate_probe_row` (Task 1).
- Produces: `validate_file(path: str) -> list[str]` (all errors, prefixed by line number).

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_validate_table_probe_evalset.py
import json
from scripts.eval.validate_table_probe_evalset import validate_file


def test_validate_file_passes_clean(tmp_path):
    p = tmp_path / "ok.jsonl"
    p.write_text(json.dumps({
        "id": "q1", "question": "Dose of X for age 5?",
        "reference_answer": "10mg", "gold_contexts": ["| age | dose |\n| 5 | 10mg |"],
    }) + "\n")
    assert validate_file(str(p)) == []


def test_validate_file_reports_bad_row(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"id": "q1", "question": "Q?", "gold_contexts": []}) + "\n")
    errs = validate_file(str(p))
    assert errs and errs[0].startswith("line 1")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/eval/test_validate_table_probe_evalset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.eval.validate_table_probe_evalset'`

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

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/eval/test_validate_table_probe_evalset.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Author the evalset (manual, at home)**

For each doc chosen in Task 3, open the table and write 1–2 questions whose answer lives in a specific cell / needs row-column alignment. Target 8–12 rows total. One row per line, this exact schema:

```json
{"id": "tbl-01", "question": "<question whose answer is in a table cell>", "reference_answer": "<exact cell value>", "gold_contexts": ["<the table text / markdown that contains the answer>"], "source_doc": "<doc path from Task 3>"}
```

Rules:
- Answer must NOT be inferable from prose alone — it must require the table.
- Prefer numeric medical tables (doses, lab ranges, comparisons).
- `gold_contexts` is a list; put the table span that holds the answer.
- Do NOT copy rows from `prod_corpus_evalset_v3.jsonl` (INVALID vs real corpus).

Save to `data/eval/table_probe_evalset.jsonl`.

- [ ] **Step 6: Validate the authored file**

Run: `PYTHONPATH=. uv run python scripts/eval/validate_table_probe_evalset.py`
Expected: `OK — evalset valid`. Also confirm row count is 8–12:
Run: `wc -l data/eval/table_probe_evalset.jsonl`

- [ ] **Step 7: Commit**

```bash
git add scripts/eval/validate_table_probe_evalset.py tests/eval/test_validate_table_probe_evalset.py data/eval/table_probe_evalset.jsonl
git commit -m "feat(eval): table probe evalset + schema validator"
```

---

### Task 6: A/B runner (`run_table_probe_ab.py`)

Ingest the probe docs twice (arm A flag-off, arm B flag-on) into isolated indices, run each probe question against each arm, and `--compare` into a lift report. Modeled on `scripts/eval/run_vision_e2e_ab.py`.

**Files:**
- Create: `scripts/eval/run_table_probe_ab.py`
- Test: `tests/eval/test_run_table_probe_ab.py`
- Output: `data/eval/table_probe_arm_a.json`, `data/eval/table_probe_arm_b.json`, `docs/eval/table_probe_2026-07-24.md`

**Interfaces:**
- Consumes: `arm_index_name`, `compute_lift` (Task 1); `settings.PDF_PRESERVE_TABLES` (Task 4); `load_local_jsonl`, `get_agent_service`, `score_correctness`, `LLMGateway` (existing harness, see `run_vision_e2e_ab.py`).
- Produces: `build_report(a_path, b_path) -> str` (markdown).

- [ ] **Step 1: Write the failing test**

Only the pure `build_report` is unit-tested (the live arm run is verified manually — it needs ES/TEI/agent).

```python
# tests/eval/test_run_table_probe_ab.py
import json
from scripts.eval.run_table_probe_ab import build_report


def test_build_report_has_lift_and_decision(tmp_path):
    a = tmp_path / "a.json"; b = tmp_path / "b.json"
    a.write_text(json.dumps({"arm": "a", "n": 10, "mean_correctness": 0.60}))
    b.write_text(json.dumps({"arm": "b", "n": 10, "mean_correctness": 0.74}))
    md = build_report(str(a), str(b))
    assert "0.14" in md            # lift
    assert "GO" in md              # decision (>= +0.10)
    assert "directional" in md.lower()  # small-n caveat present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/eval/test_run_table_probe_ab.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.eval.run_table_probe_ab'`

- [ ] **Step 3: Write the runner**

```python
# scripts/eval/run_table_probe_ab.py
"""Table-data probe A/B — arm A (flat) vs arm B (PDF_PRESERVE_TABLES).

  # arm A: ingest flat + score
  PYTHONPATH=. uv run python scripts/eval/run_table_probe_ab.py --arm a --corpus <DIR>
  # arm B: ingest table-preserved + score
  PYTHONPATH=. uv run python scripts/eval/run_table_probe_ab.py --arm b --corpus <DIR>
  # compare
  PYTHONPATH=. uv run python scripts/eval/run_table_probe_ab.py \
      --compare data/eval/table_probe_arm_a.json data/eval/table_probe_arm_b.json
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys

sys.path.insert(0, ".")
from scripts.eval.table_probe_lib import arm_index_name, compute_lift

EVAL_PATH = "data/eval/table_probe_evalset.jsonl"
BASE_INDEX = "table_probe"


async def _run_arm(arm: str, corpus: str, out_path: str) -> None:
    from src.agentrag.config import settings
    from src.agentrag.eval.benchmark_datasets import load_local_jsonl
    from src.agentrag.agent.factory import get_agent_service
    from src.agentrag.eval.correctness_judge import score_correctness
    from src.agentrag.services.llm_gateway import LLMGateway

    # Arm B preserves tables; arm A is the flat control. Set the call-time singleton.
    settings.PDF_PRESERVE_TABLES = (arm == "b")
    index = arm_index_name(BASE_INDEX, arm)
    print(f"ARM={arm} PDF_PRESERVE_TABLES={settings.PDF_PRESERVE_TABLES} index={index}")

    # 1) Ingest the probe docs into this arm's isolated index.
    #    VERIFY AT HOME: exact ingest entrypoint + how to target an index name.
    #    Explore reported src/agentrag/ingestion/pipeline.py:59 `ingest_folder(...)`.
    from src.agentrag.ingestion.pipeline import ingest_folder
    ingest_folder(corpus, index=index)  # adjust kwarg to the real signature

    # 2) Score each probe question against this arm.
    examples = load_local_jsonl(EVAL_PATH, n=100)
    agent, gateway = get_agent_service(index=index), LLMGateway()  # adjust to real selector
    rows = []
    for ex in examples:
        out = await agent.chat(question=ex.question, document_title=None,
                               conversation_id=f"tbl-{arm}-{ex.id}")
        ans = (out.get("answer") or "") if not out.get("timed_out") else ""
        e = await score_correctness(ex.question, ans, ex.reference_answer,
                                    "\n".join(ex.gold_contexts), gateway)
        rows.append({"id": ex.id, "score": e.mean, "answer": ans[:300]})
        print(f"  {ex.id} score={e.mean if e.mean is not None else 'NA'}")

    scored = [r["score"] for r in rows if isinstance(r["score"], (int, float))]
    mean = round(sum(scored) / len(scored), 4) if scored else 0.0
    json.dump({"arm": arm, "index": index, "n": len(rows), "scored": len(scored),
               "mean_correctness": mean, "rows": rows},
              open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"ARM={arm} MEAN={mean} scored={len(scored)}/{len(rows)} -> {out_path}")


def build_report(a_path: str, b_path: str) -> str:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    r = compute_lift(b["mean_correctness"], a["mean_correctness"])
    return "\n".join([
        "# Table-Data Probe — A/B (2026-07-24)",
        "",
        f"Eval set: `{EVAL_PATH}` (n={a.get('n')}). **Directional only — small n.**",
        "",
        "| arm | PDF_PRESERVE_TABLES | n | mean correctness |",
        "|---|---|---|---|",
        f"| A (control) | off | {a.get('n')} | {a['mean_correctness']} |",
        f"| B (tables)  | on  | {b.get('n')} | {b['mean_correctness']} |",
        "",
        f"**Lift (B−A) = {r['lift']}**, gate = +{r['gate']} → **{r['decision']}**",
        "",
        ("GO → write a follow-up spec for a real table-aware build."
         if r["decision"] == "GO"
         else "NO-GO → keep flat-text ingest; do not build on a hunch."),
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["a", "b"])
    ap.add_argument("--corpus", help="dir of probe PDFs (required with --arm)")
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    args = ap.parse_args()
    if args.compare:
        md = build_report(*args.compare)
        out = "docs/eval/table_probe_2026-07-24.md"
        open(out, "w").write(md)
        print(md)
        print(f"\n-> {out}")
    elif args.arm:
        if not args.corpus:
            ap.error("--corpus is required with --arm")
        asyncio.run(_run_arm(args.arm, args.corpus,
                             f"data/eval/table_probe_arm_{args.arm}.json"))
    else:
        ap.error("pass --arm a|b --corpus DIR, or --compare A_JSON B_JSON")
```

> NOTE — verify at home before running arms: (1) the real `ingest_folder`
> signature and how to direct it at a named index (Explore reported it at
> `pipeline.py:59`); (2) how `get_agent_service` selects which index it queries.
> These two I/O seams are the only unverified calls — the pure `build_report`
> path and everything in Tasks 1–5 is tested. If `ingest_folder` can't target a
> custom index, fall back to ingesting arms sequentially into the default index
> and running each arm's scoring immediately after its own ingest (no isolation
> flag needed, but never ingest both arms before scoring).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/eval/test_run_table_probe_ab.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Full A/B run (at home, FULL track only)**

```bash
PYTHONPATH=. uv run python scripts/eval/run_table_probe_ab.py --arm a --corpus <PROBE_DOCS_DIR>
PYTHONPATH=. uv run python scripts/eval/run_table_probe_ab.py --arm b --corpus <PROBE_DOCS_DIR>
PYTHONPATH=. uv run python scripts/eval/run_table_probe_ab.py \
    --compare data/eval/table_probe_arm_a.json data/eval/table_probe_arm_b.json
```
Expected: prints the report table with `Lift (B−A) = <n>` and `GO`/`NO-GO`, writes `docs/eval/table_probe_2026-07-24.md`.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval/run_table_probe_ab.py tests/eval/test_run_table_probe_ab.py
git commit -m "feat(eval): table-data probe A/B runner + lift report"
```

- [ ] **Step 7: Commit the result report**

```bash
git add docs/eval/table_probe_2026-07-24.md data/eval/table_probe_arm_a.json data/eval/table_probe_arm_b.json
git commit -m "docs(eval): table-data probe A/B result — <GO|NO-GO>, lift=<n>"
```

---

## OFFLINE-track variant (Task 6 fallback, when preflight = offline)

If preflight returns `offline` (no ES/TEI/judge), skip answer-gen. Replace Task 6 Step 5 with a retrieval-recall + chunk-integrity comparison:
- For each arm's index, for each probe question, retrieve top-k and record whether the gold table chunk is in the results (recall@k).
- Record whether the gold table chunk survived chunking as one intact chunk (not split mid-row).
- Reuse `compute_lift` with recall@k as the metric instead of correctness; same +0.10 gate.

This variant is lower-fidelity (spec §3 Step 0). Implement it only if the stack is down at run time; the pure lib (Task 1) and gate logic are unchanged.

---

## Self-Review

**Spec coverage:**
- §3 Step 0 preflight → Task 2 ✓
- §3 Step 1 find docs → Task 3 ✓
- §3 Step 2 author evalset → Task 5 ✓
- §3 Step 3 two ingest variants (`PDF_PRESERVE_TABLES`) → Task 4 ✓
- §3 Step 4 run A/B (FULL + OFFLINE) → Task 6 + OFFLINE variant ✓
- §3 Step 5 go/no-go gate (+0.10) → `compute_lift` (Task 1), report (Task 6) ✓
- §4 metrics table → report (Task 6) ✓
- §5 deliverables 1–7 → Tasks 2,3,5,4,6,6,6 ✓
- §6 risks (grounded judge, corpus validity, arm isolation) → Task 6 grounded `score_correctness`, Task 5 no-v3 rule, Task 1 `arm_index_name` ✓

**Placeholder scan:** no TBD/TODO in requirements. Two `VERIFY AT HOME` notes are on genuine I/O seams (settings attr names, ingest signature) that can't be known without the running stack — each ships an import-safe default + a concrete fallback, not a blank.

**Type consistency:** `arm` is `"a"|"b"` everywhere; `compute_lift(b_mean, a_mean, gate=0.10)` called consistently; `mean_correctness` key used identically in `_run_arm` output and `build_report` input; `gold_contexts` treated as list throughout.
