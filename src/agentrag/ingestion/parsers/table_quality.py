"""Table-quality gate + deterministic markdown rendering for the table-data probe.

PROBE-SCOPED (2026-08-02) — delete together with `PDF_PRESERVE_TABLES`.

Why this exists: PyMuPDF's `page.find_tables()` fires on any bordered layout
box, not only on data grids. Measured on the real corpus (29 unique PDFs,
`docs/eval/table_probe_corpus_survey_2026-09-06.md`), only a minority of
detections are data grids. For the rest, converting to markdown emits output
that is *worse* than the flat text it would replace, so arm B would lose for
reasons unrelated to tables and a null probe result could not be read as
"tables don't matter". This module keeps arm B's edit confined to detections
that are structurally grid-like.

Rendering is ours, not PyMuPDF's (2026-09-06)
---------------------------------------------
Arm B must NOT call `table.to_markdown()`. Measured over the corpus, that
renderer is the actual source of the corruption this gate exists to block:

    |1. ĐỊNH NGHĨA|Col2|                    <- invented column header
    |Lo lắng là hiện tượng...|Lo lắng là hiện tượng...|   <- cell mirrored

`extract()` returns `['Lo lắng là hiện tượng...', None]` for that same box —
one populated cell — so a gate reading `extract()` can never see the mirroring
or the invented header, because `to_markdown()` invents both. Gating one
representation while emitting another is unmeasurable by construction.

`render_markdown` therefore renders the same `extract()` cells the gate judged:
no invented `ColN` headers, no mirroring, pipes escaped, and the header row
repeated on every emitted block (the trick `excel_parser._rows_to_markdown`
already uses for spreadsheets).

Blocks are sized by a *token budget*, not a fixed row count. Measured over the
corpus, tokens-per-cell ranges 4.1–47.8 (median 9.4), so `ROWS_PER_BLOCK=8` gives
blocks of 53–1090 tokens and 7 of 40 gate-passing tables (13% of emitted blocks)
overflow `SEARCH_CHUNK_MAX_TOKENS`. Packing to the budget takes that to zero. Packing to the budget instead means every block is a
blank-line-separated paragraph that already fits the chunk window, so the
chunker never has to cut inside a row and no table-aware chunking special case
is needed — each block carries its own header and stands alone.

Two distinct questions, deliberately separated:
  - `is_safe_to_markdown` — will rendering this do no harm? (structure only)
  - `is_data_grid`        — does the table carry aligned numbers worth asking
                            an eval question about? (structure + numeric density)

Arm B gates on the first. Doc ranking for question authoring uses the second.
Numeric density must NOT gate arm B: a text-only comparison matrix is still a
real table whose columns carry meaning.
"""

from __future__ import annotations

import re
from itertools import pairwise

_DIGIT = re.compile(r"\d")
_WS = re.compile(r"\s+")

#: A cell longer than this is a paragraph, not a cell.
MAX_CELL_CHARS = 80
#: Fraction of rows whose non-empty cells are all identical (mirrored text).
MAX_MIRRORED_ROW_FRAC = 0.4
#: Fraction of cells that are paragraph-length.
MAX_LONG_CELL_FRAC = 0.4
#: Below this fraction of number-bearing cells, a grid is not a data grid.
MIN_NUMERIC_CELL_FRAC = 0.15
#: A cell this short counts as numeric on a single digit ("1", "10mg", "0.5%").
#: Longer cells must be digit-dense, so a sentence mentioning a year does not
#: promote a prose strip to `real_data`.
SHORT_NUMERIC_CELL_CHARS = 12
#: Digit share (of non-space characters) that makes a long cell numeric.
MIN_CELL_DIGIT_DENSITY = 0.3
#: Rows that must carry >= 2 populated cells for real column structure to exist.
MIN_STRUCTURED_ROWS = 2
#: Consecutive ordinal cells needed before a column counts as a row counter.
MIN_ORDINAL_RUN = 3
#: Adjacent columns holding the same text only signal a mirrored layout box when
#: that text is substantial. A skills grid legitimately repeats "1" across score
#: columns, and treating that as mirroring would reject the probe's target class.
MIN_MIRRORED_CELL_CHARS = 20
#: Fallback data rows per block when no token budget is supplied.
ROWS_PER_BLOCK = 8
#: Chars per token used when `tiktoken` is unavailable. Measured against cl100k
#: on this corpus's Vietnamese table text: 1.76-2.1 chars/token (ASCII runs ~3.2).
#: Set BELOW the Vietnamese figure so the estimate over-counts and blocks come out
#: smaller, never larger — an under-count would emit a block over the chunk window,
#: which is the mid-row slicing this module exists to prevent.
FALLBACK_CHARS_PER_TOKEN = 1.5

#: Structurally sound kinds — rendering is safe for these.
SAFE_KINDS = ("real_data", "nonnumeric")


def _normalize(rows) -> list[list[str]]:
    """Strip cells, drop fully empty rows. `None` cells become empty strings."""
    out = []
    for row in rows or []:
        cells = [_WS.sub(" ", (c or "").replace("\n", " ")).strip() for c in row]
        if any(cells):
            out.append(cells)
    return out


def _populated(row: list[str]) -> list[str]:
    return [c for c in row if c]


def _is_numeric_cell(cell: str) -> bool:
    """True when the cell reads as a measurement, not prose that mentions a year."""
    if not _DIGIT.search(cell):
        return False
    if len(cell) <= SHORT_NUMERIC_CELL_CHARS:
        return True
    dense = [c for c in cell if not c.isspace()]
    if not dense:
        return False
    return sum(c.isdigit() for c in dense) / len(dense) >= MIN_CELL_DIGIT_DENSITY


def _mirrored_rows(norm: list[list[str]]) -> int:
    """Rows whose populated cells are all the same text, or that duplicate an
    adjacent column pair. PyMuPDF layout boxes mirror per column-pair, not only
    across the whole row, so the whole-row test alone under-counts badly.
    """
    count = 0
    for row in norm:
        pop = _populated(row)
        if len(pop) >= 2 and len(set(pop)) == 1:
            count += 1
            continue
        if any(
            len(row[i]) >= MIN_MIRRORED_CELL_CHARS and row[i] == row[i + 1]
            for i in range(len(row) - 1)
        ):
            count += 1
    return count


_ORDINAL_CELL = re.compile(r"^\d+(?:\.\d+)*\.?$")


def _ordinal_key(cell: str) -> tuple[int, ...]:
    return tuple(int(part) for part in cell.rstrip(".").split(".") if part)


def _ordinal_columns(norm: list[list[str]]) -> set[int]:
    """Column indices that are just a row counter (1, 2, 3 ... or 1.1, 1.2, 1.4).

    An index column is not data. Counting it as numeric promoted procedure
    checklists whose only digits are the STT counter to `real_data`, the class
    used to pick tables to author eval questions about — half the ranked targets
    then carried no measurement to ask a question about.

    Deliberately narrow: only the leading column, only when *every* data cell in
    it is bare numbering and the sequence never decreases. A dose or score column
    holding real numbers is not excluded, and the header row is skipped because a
    label like "STT" would otherwise break the run.
    """
    if len(norm) < 2:
        return set()
    body = norm[1:]
    for col in range(max(len(r) for r in norm)):
        cells = [row[col] for row in body if col < len(row) and row[col]]
        if len(cells) < MIN_ORDINAL_RUN:
            continue  # too short to call it a counter; try the next column
        if not all(_ORDINAL_CELL.match(c) for c in cells):
            return set()
        keys = [_ordinal_key(c) for c in cells]
        if all(b >= a for a, b in pairwise(keys)):
            return {col}
        return set()
    return set()


def classify_table(rows) -> str:
    """Classify extracted table cells (``list[list[str | None]]``).

    Returns one of:
      ``degenerate``     — fewer than 2 usable rows, or no column at all.
      ``layout_dup``     — same text mirrored across columns (a layout box).
      ``layout_prose``   — cells are paragraph-length (flowing text, not cells).
      ``single_column``  — 2+ columns detected but fewer than
                           ``MIN_STRUCTURED_ROWS`` rows actually populate two of
                           them, so there is no alignment to restore.
      ``nonnumeric``     — a real grid, but carries no numbers.
      ``real_data``      — a grid with aligned numeric content.
    """
    return _classify_norm(_normalize(rows))


def _classify_norm(norm: list[list[str]]) -> str:
    """`classify_table` on already-normalized rows. Keeps the gate and the
    renderer working from one normalization pass, so they cannot disagree."""
    if len(norm) < 2:
        return "degenerate"
    if max(len(r) for r in norm) < 2:
        return "degenerate"

    cells = [c for row in norm for c in row if c]
    if not cells:
        return "degenerate"

    if _mirrored_rows(norm) / len(norm) >= MAX_MIRRORED_ROW_FRAC:
        return "layout_dup"

    long_cells = sum(1 for c in cells if len(c) > MAX_CELL_CHARS)
    if long_cells / len(cells) >= MAX_LONG_CELL_FRAC:
        return "layout_prose"

    # `max(len(row))` counts DETECTED columns, including the empty ones PyMuPDF
    # emits as None. Real column structure needs rows that actually fill two.
    if sum(1 for row in norm if len(_populated(row)) >= 2) < MIN_STRUCTURED_ROWS:
        return "single_column"

    ordinal = _ordinal_columns(norm)
    scored = [
        c
        for row in norm
        for i, c in enumerate(row)
        if c and i not in ordinal
    ]
    if not scored:
        return "nonnumeric"
    numeric = sum(1 for c in scored if _is_numeric_cell(c))
    if numeric / len(scored) < MIN_NUMERIC_CELL_FRAC:
        return "nonnumeric"
    return "real_data"


def is_safe_to_markdown(rows) -> bool:
    """True when rendering this table will not corrupt the page text.

    This is arm B's gate. Structure only — a numberless grid still passes.
    Pair it with `render_markdown`, never with `table.to_markdown()`: the gate
    judges these cells, so the emitted text must come from these cells too.
    """
    return classify_table(rows) in SAFE_KINDS


def is_data_grid(rows) -> bool:
    """True for grids with aligned numeric content — the probe's target class.

    Used to rank docs for eval-question authoring, NOT to gate arm B.
    """
    return classify_table(rows) == "real_data"


def _escape(cell: str) -> str:
    return cell.replace("|", r"\|")


def estimate_tokens(text: str) -> int:
    """Token count for `text`, via tiktoken when present, else a safe estimate."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:  # noqa: BLE001 — tiktoken absent or its BPE fetch failed
        return int(len(text) / FALLBACK_CHARS_PER_TOKEN) + 1


def render_markdown(
    rows,
    max_tokens: int | None = None,
    rows_per_block: int = ROWS_PER_BLOCK,
) -> str:
    """Render gate-approved `extract()` cells as GFM. Empty string if unsafe.

    The first normalized row is the header — taken from the document, never
    invented. Every block repeats it, so a block can be retrieved on its own and
    still name its columns.

    `max_tokens` (pass `settings.SEARCH_CHUNK_MAX_TOKENS`) packs data rows into
    blocks that fit the chunk window. Blocks are separated by a blank line, so
    the paragraph-splitting chunker keeps each one whole and never cuts a row.
    Without it, blocks fall back to a fixed `rows_per_block`. A single row wider
    than the budget is still emitted whole — splitting it would destroy the row
    alignment this exists to preserve.
    """
    if rows_per_block < 1:
        raise ValueError(f"rows_per_block must be >= 1, got {rows_per_block}")

    norm = _normalize(rows)
    if _classify_norm(norm) not in SAFE_KINDS:
        return ""

    width = max(len(r) for r in norm)
    header = [_escape(c) for c in norm[0]] + [""] * (width - len(norm[0]))
    header_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    head = "\n".join(header_lines)

    data = norm[1:]
    if not data:
        return head

    lines = []
    for row in data:
        padded = [_escape(c) for c in row] + [""] * (width - len(row))
        lines.append("| " + " | ".join(padded[:width]) + " |")

    if max_tokens is None:
        batches = [
            lines[i : i + rows_per_block] for i in range(0, len(lines), rows_per_block)
        ]
    else:
        budget = max(max_tokens, 1)
        head_cost = estimate_tokens(head)
        batches, current, cost = [], [], head_cost
        for line in lines:
            line_cost = estimate_tokens(line)
            if current and cost + line_cost > budget:
                batches.append(current)
                current, cost = [], head_cost
            current.append(line)
            cost += line_cost
        if current:
            batches.append(current)

    return "\n\n".join("\n".join([head, *batch]) for batch in batches)
