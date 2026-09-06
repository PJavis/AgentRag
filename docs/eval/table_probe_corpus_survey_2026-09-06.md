# Table-Probe Corpus Survey — 2026-09-06 (corrected instrument)

**Status:** measured, reproducible. Supersedes `table_probe_corpus_survey_2026-08-02.md`.
**Reproduce:**
```bash
PYTHONPATH=. uv run python scripts/eval/table_probe_corpus_survey.py \
    --corpus data/originals \
    --json data/eval/table_probe_corpus_survey.json \
    --unique-list data/eval/table_probe_unique_docs.txt \
    --dedupe-dir data/eval/table_probe_corpus
```
Raw report: `data/eval/table_probe_corpus_survey.json`. 0 unreadable documents, 0
unreadable pages.

The 2026-08-02 survey answered "can this corpus support the probe?". This one re-runs
the same questions after a code review found the classifier that produced those answers
was measuring the wrong artifact. §1 (dedup) and §2 (text layer) are unchanged and are
restated here only so one document carries the whole picture.

## 0. What was wrong with the 2026-08-02 instrument

The gate classified `table.extract()`, but arm B was to emit `table.to_markdown()`.
Those are different artifacts, and every corruption the gate exists to block is created
by the second one:

```
extract()     -> ['Lo lắng là hiện tượng phản ứng của con người...', None]
to_markdown() -> |1. ĐỊNH NGHĨA|Col2|                       <- header invented here
                 |Lo lắng là hiện tượng...|Lo lắng là hiện tượng...|  <- mirrored here
```

A gate reading the first can never see either problem. Measured consequence on this
corpus: of the 90 detections the old gate passed, **69 (77%) produced an invented `ColN`
header** and 14 produced markdown that was ≥40% mirrored rows. The spec's claim that the
treatment "cannot corrupt the 47%" was false as shipped.

Three narrower defects compounded it:

| defect | measured effect |
|---|---|
| `max(len(row)) < 2` counted `None` placeholders as columns | 50 one-column prose strips passed as "2-column tables" |
| mirroring tested whole-row only; PyMuPDF mirrors per adjacent column pair | corpus-wide `layout_dup` was 1 of 171 |
| any cell containing a digit counted as numeric | prose mentioning a year ranked as a `real_data` probe target |

**Fix:** arm B renders with `table_quality.render_markdown`, which renders the same
`extract()` cells the gate judged — document's own header, no mirroring, pipes escaped.
`to_markdown()` is not called anywhere in the probe.

## 1. Corpus is 4× duplicated  *(unchanged)*

| metric | value |
|---|---|
| PDF files on disk | 116 |
| **unique documents** (sha256 of bytes) | **29** |
| redundant copies | 87 |

Every one of the 29 documents is duplicated. Group sizes: one ×7, two ×6, four ×5,
sixteen ×4, one ×3, five ×2 — 116 files in total, no singletons.

The survey now writes the deduplicated set to disk (`--unique-list`, `--dedupe-dir`),
because "all later steps consume the unique-document list" was a design requirement that
no artifact satisfied — only the integer 29 was emitted, while the A/B runner asks for a
`<UNIQUE_DOCS_DIR>` that did not exist.

## 2. A quarter of pages have no text layer  *(unchanged)*

| metric | value |
|---|---|
| pages (unique docs) | 684 |
| pages with a text layer (≥ `PDF_OCR_MIN_TEXT_CHARS`=50 chars) | 515 |
| **pages with no text layer** | **169 (25%)** |

Those pages take the OCR / vision-LLM path in `PDFParser.parse`
(`pdf_parser.py:86-111`), which returns free-form text. `page.find_tables()` finds
nothing there, so **arm B is byte-identical to arm A on 25% of pages**. One document
(`29e99aa0`, 107 pages) has a text layer on exactly 1 page.

This identity is now structural, not incidental: the table append runs *after* the OCR
block. Placed before it — as the plan originally specified — the appended pipes and
separator row would push a thin page past `PDF_OCR_MIN_TEXT_CHARS` and stop arm B taking
an OCR fallback that arm A takes, making arm B strictly worse on exactly the pages the
scope claim says are identical.

## 3. Most detected "tables" are still not tables — and fewer are safe than we thought

| kind | count | share | safe to render? |
|---|---|---|---|
| `degenerate` (<2 usable rows, or one column) | 60 | 35% | no |
| **`single_column`** (2+ columns detected, <2 rows fill two) | **50** | **29%** | no |
| **`real_data`** (numeric grid) | **19** | **11%** | yes |
| `layout_prose` (paragraph cells) | 20 | 12% | no |
| `nonnumeric` (real grid, no numbers) | 21 | 12% | yes |
| `layout_dup` (text mirrored across columns) | 1 | 1% | no |
| total detections | 171 | | **40 safe (23%)** |

`single_column` is the new kind and the largest single correction: PyMuPDF reports a
column count that includes placeholders it never fills. The exemplar the old report
quoted — `881e7c3a` p.134 — is one of these. It now classifies `single_column` and
renders to the empty string, where before it passed the gate and would have been emitted
with a `Col2` header.

**Old gate vs new gate, same corpus, same detections:**

| | old | new |
|---|---|---|
| detections passing the gate | 90 (53%) | 40 (23%) |
| …of those, emitting an invented `ColN` header | **69** | **0** |
| …of those, ≥40% mirrored markdown rows | 14 | **0** |

## 3b. The eligible set is 40 tables, of which 27 can carry a question

`real_data` is the **ranking** class, not the eligible set. Arm B rewrites every
gate-passing table, and `table_quality.py` states the reason: *"a text-only comparison
matrix is still a real table whose columns carry meaning."*

| | count |
|---|---|
| gate-passing tables (`candidate_tables`) | **40** |
| — with ≥3 structured rows AND ≥3 columns (`structured_candidates`) | **27** |
| — of those: `real_data` / `nonnumeric` | 16 / 11 |
| — too thin to carry an alignment question | 13 |

Those 27 span 7 documents, but **14 of them are in `2f499de4`**. That concentration is
the binding constraint on what the probe can conclude — see
`table_probe_power_analysis_2026-09-06.md`, which settles the unit of analysis (the
table, not the question) and shows the corpus cannot support a confirmatory go/no-go.

## 4. Where a real table exists, the hypothesis still holds

19 data grids across **5 unique docs** (was 35 across 6 — the difference is prose that
the digit-anywhere test had promoted). The original spec asked for "5–8 PDF docs with
substantive numeric tables"; the corpus supports 5, unevenly.

`2f499de4` p.8, arm A (flat text):

```
STT   Tên kỹ năng      Quan sát   hướng dẫn của   Làm đúng   Làm thành thạo
 1.4  Đặt thông tiểu nam/nữ lấy nước    1     1     1     0
```

The header is shattered across lines and `1 1 1 0` has lost its binding to the columns —
"is step 1.4 marked *làm thành thạo*?" is unanswerable from this. Arm B restores the
binding:

```
| STT | Tên kỹ năng | Chỉ tiêu |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  | Quan sát | Thực hành có hướng dẫn của GV | Làm đúng | Làm thành thạo |
| 1 | Kỹ thuật đặt ống thông tiểu nam/nữ lấy nước tiểu làm xét nghiệm |  |  |  |  |
```

**Known limit, stated rather than hidden:** this grid has a two-level header — row 0
carries a merged `Chỉ tiêu` spanning four sub-columns whose names are in row 2. The
renderer takes row 0 as the header, so the emitted header row is thin and the real column
names arrive as an ordinary data row. Row/column *binding* is restored (that is the
mechanism under test); header *naming* is not fully. Questions authored against this
document should not depend on the header line alone.

## 5. Chunk-window fit is solved in the renderer

`SEARCH_CHUNK_MAX_TOKENS=512`. A markdown table has no blank lines, so it reaches
`_chunk_section_by_paragraph` as one oversized paragraph and goes to
`_chunk_section_by_tokens` (`hybrid_chunker.py:138`, under the branch at :133) — a blind
token window that cuts mid-row.

`render_markdown` packs rows into header-repeating blocks sized to the token budget and
separates blocks with a blank line, so each block is an ordinary paragraph that already
fits the window.

| metric | value |
|---|---|
| gate-passing tables rendered | 40 |
| blocks emitted at `max_tokens=512` | 55 |
| **blocks over budget** | **0** |
| worst block | 500 tokens |

A fixed rows-per-block count cannot achieve this: measured tokens per cell over the 19
data grids range **4.1 – 47.8** (median 9.4), and running the module's own
`ROWS_PER_BLOCK=8` fallback over all 40 gate-passing tables gives blocks of **53 – 1090**
tokens, 13% of which overflow the window. It is also why cell count is a bad ranking key — 20 cells renders to 955
tokens while 104 cells renders to 429 — and why the survey records, per grid, both
`est_tokens` (whole table) and `max_block_tokens` (largest emitted block).
`rank_targets(max_tokens=...)` filters on **`max_block_tokens`**: filtering on the table
total would discard every large multi-row grid even though each of its blocks fits, which
is exactly the case where the two arms differ most.

**No production chunker change is needed or made.**

## Candidate documents for question authoring

| data grids | document | pages | text-layer pages |
|---|---|---|---|
| 9 | `2f499de4-1467-4cea-b1de-f42e854a5c6a.pdf` | 19 | 19 |
| 5 | `81273a0e-f851-4915-a4c8-885fb5b7afb0.pdf` | 27 | 27 |
| 3 | `0bc89e50-c8dc-433e-8c3d-e09b462a8690.pdf` | 20 | 20 |
| 1 | `29e99aa0-9d83-4108-a3af-c5df0a14fab8.pdf` | 107 | 1 |
| 1 | `5164c4af-ec00-46e3-8b08-52d8f3595d20.pdf` | 38 | 30 |

`881e7c3a` (198 pages) drops off the list entirely: all 5 of its former "data grids"
were prose boxes.

Per-grid page, dimensions and `est_tokens` are in the JSON under `per_doc[].data_grids`.
Whole-table rendered size spans 55 – 1310 tokens (median 314). Largest **block** spans
55 – 500 tokens: all 19 grids fit the 512-token window, so none is excluded by size.

**Target selection.** `rank_targets` interleaves documents round-robin. Ranking flat by
cell count returned 8 of the top 10 from `2f499de4` alone — a *worse* concentration than
the doc-level selection the table-level unit of analysis was introduced to avoid — while
the sign test's p-value assumes wins and losses are independent draws over the corpus.
Re-measured on this survey: the flat sort still puts **8 of the top 10** in `2f499de4`;
round-robin gives **3 of 10**.

## What this changes for the probe

1. **Dedup first, on disk.** `--dedupe-dir` output is what `--corpus` receives.
2. **Arm B renders `render_markdown`, never `to_markdown()`.** The gate must judge the
   artifact that ships.
3. **Append after the OCR block**, so the treatment cannot change OCR routing.
4. **Chunk integrity is a check, not a chunker feature.** Blocks fit by construction; a
   non-trivial exclusion count means something upstream regressed.
5. **The target set is smaller than the 2026-08-02 report implied**: 19 grids in 5 docs, every
   one of which fits the chunk window. n remains the binding constraint on this probe, and no
   ranking or gating change can manufacture more of it.
