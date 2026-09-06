# Table-Probe Corpus Survey — 2026-08-02

> **SUPERSEDED 2026-09-06 — see `table_probe_corpus_survey_2026-09-06.md`.**
>
> The table-kind counts in §3, §4 and the candidate-document table below were
> produced by a classifier with three defects, found by code review:
>
> - it counted PyMuPDF's `None` column placeholders as real columns, so
>   one-column prose strips were classified as 2-column tables;
> - it tested mirroring across a whole row only, while PyMuPDF mirrors per
>   adjacent column pair (hence the implausible `layout_dup` = 1/171 here);
> - it treated any cell containing a digit as numeric, so prose mentioning a
>   year was promoted to `real_data`.
>
> Corrected counts: **19** data grids across **5** docs (not 35 across 6), and
> **40** of 171 detections safe to convert (not 90). §1 and §2 — the dedup and
> text-layer numbers — are unaffected and still hold.
>
> The `to_markdown()` corruption documented in §3 is real and is the reason the
> probe no longer uses that renderer at all; §4's arm-B example below shows
> `to_markdown()` output and is retained only as the illustration of the problem.
> This file is kept as the record of what was measured on 2026-08-02.

**Status:** measured, reproducible
**Reproduce:**
```bash
PYTHONPATH=. uv run python scripts/eval/table_probe_corpus_survey.py \
    --corpus data/originals --json data/eval/table_probe_corpus_survey.json
```
Raw report: `data/eval/table_probe_corpus_survey.json`.

This survey was run *before* executing the table-data probe
(`docs/superpowers/specs/2026-07-24-table-data-probe-design.md`), to check whether
the corpus can support the measurement the probe intends to make. It cannot, as
originally specced. Four design changes follow from the numbers below; the spec and
plan have been revised accordingly.

## 1. Corpus is 4× duplicated

| metric | value |
|---|---|
| PDF files on disk | 116 |
| **unique documents** (sha256 of bytes) | **29** |
| redundant copies | 87 |

Largest duplicate groups: ×7, ×6, ×6, ×5, ×5, ×5, ×5, ×4.

This is not a table problem — it affects every measurement taken over this corpus.
Retrieval returns near-identical chunks from 4–7 copies of the same source, which
inflates top-k occupancy, distorts recall@k, and makes any doc-level ranking
(including this probe's doc selection) count the same document repeatedly.

**Consequence for the probe:** doc selection and every per-doc statistic must run over
the 29 unique documents, never the 116 files.

## 2. A quarter of pages have no text layer

| metric | value |
|---|---|
| pages (unique docs) | 684 |
| pages with a text layer (≥ `PDF_OCR_MIN_TEXT_CHARS`=50 chars) | 515 |
| **pages with no text layer** | **169 (25%)** |

Those pages take the OCR / vision-LLM path in `PDFParser.parse`
(`src/agentrag/ingestion/parsers/pdf_parser.py:86-111`), which returns free-form text.
`page.find_tables()` finds nothing there, so **arm B is byte-identical to arm A on 25%
of pages**. One document (`29e99aa0`, 107 pages) has a text layer on exactly 1 page.

Scanned dosage tables — the case where flattening plausibly hurts most — are precisely
the case this probe cannot see. That is a stated limit of the result, not a fixable
one within the no-new-deps constraint.

## 3. Most detected "tables" are not tables

`find_tables()` fires on any bordered layout box. Classified with
`src/agentrag/ingestion/parsers/table_quality.py`:

| kind | count | share | `to_markdown()` safe? |
|---|---|---|---|
| `degenerate` (<2 usable rows/cols) | 60 | 35% | no |
| `nonnumeric` (real grid, no numbers) | 55 | 32% | yes |
| **`real_data`** (numeric grid) | **35** | **20%** | yes |
| `layout_prose` (paragraph cells) | 20 | 12% | no |
| `layout_dup` (text mirrored across columns) | 1 | 1% | no |
| total detections | 171 | | 90 safe (53%) |

On the 47% that are not structurally grids, `to_markdown()` output is **worse** than the
flat text it would replace. Real example (`881e7c3a` p.134), a bordered prose box:

```
|1. ĐỊNH NGHĨA|Col2|                                        <- invented header
|---|---|
|Lo lắng là hiện tượng phản ứng...|Lo lắng là hiện tượng phản ứng...|   <- mirrored
```

Each *visual line* of the paragraph becomes a table row, and the cell is duplicated
across both columns.

**Consequence for the probe:** an ungated arm B measures
*(gain on real tables) − (damage on layout artifacts)*. A null result would be
uninterpretable — indistinguishable from "tables don't matter". Arm B must gate on
`is_safe_to_markdown`.

## 4. Where a real table exists, the hypothesis holds

35 data grids across **6 unique docs**, and 17 of the 35 are in one document
(`2f499de4`). The original spec asked for "5–8 PDF docs with substantive numeric
tables"; the corpus supports 6, unevenly.

On those tables the mechanism the spec hypothesises is real. `2f499de4` p.8, arm A:

```
STT   Tên kỹ năng      Quan sát   hướng dẫn của   Làm đúng   Làm thành thạo
 1.4  Đặt thông tiểu nam/nữ lấy nước    1     1     1     0
```

The header row is shattered across lines and `1 1 1 0` has lost its binding to the
columns — a question like "is step 1.4 marked *làm thành thạo*?" is unanswerable from
this text. Arm B's `to_markdown()` restores the binding:

```
|1.4|Đặt thông tiểu nam/nữ lấy nước<br>tiểu làm xét nghiệm|1|1|1|0|
```

So the probe is worth running — on a corrected instrument, over a much smaller target
set than originally scoped.

## Candidate documents for question authoring

| data grids | document | pages | text-layer pages |
|---|---|---|---|
| 17 | `2f499de4-1467-4cea-b1de-f42e854a5c6a.pdf` | 19 | 19 |
| 7 | `81273a0e-f851-4915-a4c8-885fb5b7afb0.pdf` | 27 | 27 |
| 5 | `881e7c3a-eb84-49ec-9156-a14413ea7bc6.pdf` | 198 | 197 |
| 3 | `0bc89e50-c8dc-433e-8c3d-e09b462a8690.pdf` | 20 | 20 |
| 2 | `5164c4af-ec00-46e3-8b08-52d8f3595d20.pdf` | 38 | 30 |
| 1 | `29e99aa0-9d83-4108-a3af-c5df0a14fab8.pdf` | 107 | 1 |

Per-grid page and dimension lists are in the JSON report under `per_doc[].data_grids`.

## Design changes this forced

1. **Dedup first.** All probe steps operate on the 29 unique documents.
2. **Gate arm B** on `is_safe_to_markdown` so the treatment only touches structurally
   sound grids (90 of 171 detections) and cannot damage the other 47%.
3. **Paired, targeted measurement.** Compare arms per question on the questions whose
   gold table is in the changed set, and report a paired sign test — not a mean delta
   against a +0.10 gate, which at n≈10 is one question flipping and sits well inside
   the ~0.3 A/B noise already measured on this stack.
4. **Chunk integrity is a precondition, not a supporting metric.**
   `SEARCH_CHUNK_MAX_TOKENS=512` and `hybrid_chunker.py:142` will split a 17×8 markdown
   table mid-row, destroying arm B's advantage before retrieval sees it. If the gold
   table does not survive chunking intact, the downstream comparison measures chunking,
   not tables.

Additionally, the OFFLINE track must match the gold table by `(doc, page)`, not by chunk
text: the two arms produce *different* chunk text for the same table, so text matching
would compare different objects across arms.
