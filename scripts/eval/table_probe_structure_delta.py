"""0a — structure-delta: measure what flattening actually costs, before authoring anything.

The table probe rests on one unmeasured premise: that
`page.get_text("text", sort=True)` destroys row/column alignment, and that arm B's
markdown restores it. Step 0a of
`docs/superpowers/plans/2026-09-06-table-probe-unblock.md` tests that premise for
hours of compute instead of the days of question authoring the full probe costs.

What is measured, per table, is **row adjacency**: for every row carrying at least
two populated cells, do those cells still appear together on one line, in document
order? Arm A is the page text the pipeline indexes today. Arm B is
`table_quality.render_markdown`.

Read the numbers in one direction only. Arm B scores 1.0 **by construction** —
`render_markdown` emits one row per line, so its score is a tautology and is
reported flagged as one. The informative number is **arm A**: if flattening already
keeps rows intact on this corpus, arm B has nothing to restore and the probe is
over. That is the kill condition, and it is the whole point of running 0a first.

Usage:
    PYTHONPATH=. uv run python scripts/eval/table_probe_structure_delta.py \
        --corpus data/originals \
        --survey data/eval/table_probe_corpus_survey.json \
        --json data/eval/table_probe_structure_delta.json \
        --report docs/eval/table_probe_structure_delta_<date>.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agentrag.ingestion.parsers.table_quality import (  # noqa: E402
    _normalize,
    _populated,
    estimate_tokens,
    is_safe_to_markdown,
    render_markdown,
)

#: Pool definition from the power analysis: a table thin in either dimension
#: cannot carry a row/column-alignment question, so it cannot inform this probe.
MIN_STRUCTURED_ROWS = 3
MIN_COLS = 3

_WS = re.compile(r"\s+")


def norm_cell(cell) -> str:
    """Whitespace-normalized cell text.

    `extract()` keeps the line breaks a cell had on the page; the flattened page
    text does not. Comparing them raw would score a wrapped cell as destroyed when
    nothing was destroyed.
    """
    if cell is None:
        return ""
    return _WS.sub(" ", str(cell)).strip()


def cells_on_one_line(text: str, cells) -> bool:
    """True when one line of `text` holds every populated cell, in order.

    Order matters: a line carrying the cells shuffled has lost the row binding as
    surely as a line break would. Fewer than two populated cells is never a win —
    a lone cell has no adjacency to preserve, and counting it would inflate arm A.
    """
    wanted = [norm_cell(c) for c in cells]
    wanted = [c for c in wanted if c]
    if len(wanted) < 2:
        return False
    for raw_line in (text or "").splitlines():
        line = norm_cell(raw_line)
        if not line:
            continue
        pos, ok = 0, True
        for cell in wanted:
            idx = line.find(cell, pos)
            if idx < 0:
                ok = False
                break
            pos = idx + len(cell)
        if ok:
            return True
    return False


def classify_row_failure(text: str, cells) -> str:
    """Why a row failed, so the report cannot overclaim the mechanism.

    The spec blames "reading-order flattening". Some rows fail for a duller
    reason: a cell wraps over several physical lines, so no single line could
    ever hold the row. Both cost the reader the row binding, but only one is the
    scrambling the spec describes — and saying so is the difference between a
    measurement and a talking point.
    """
    wanted = [c for c in (norm_cell(c) for c in cells) if c]
    if len(wanted) < 2:
        return "not_applicable"
    flat = norm_cell(text)
    missing = [c for c in wanted if c not in flat]
    if missing:
        # Distinguish "the words are gone" from "the cell itself was cut up and
        # interleaved with the neighbouring column". The second is the worst case
        # for retrieval and by far the most common here; calling it "absent" would
        # read as an extraction bug that is not happening.
        for cell in missing:
            tokens = [t for t in cell.split(" ") if t]
            if not all(t in flat for t in tokens):
                return "cell_text_absent"
        return "cell_fragmented_across_columns"
    pos = 0
    for cell in wanted:
        idx = flat.find(cell, pos)
        if idx < 0:
            return "out_of_document_order"
        pos = idx + len(cell)
    return "ordered_but_split_across_lines"


def row_adjacency(text: str, rows) -> tuple[int, int]:
    """(rows whose cells stayed together, rows that had cells to keep together)."""
    total = preserved = 0
    for row in rows:
        if len(_populated([norm_cell(c) for c in row])) < 2:
            continue
        total += 1
        if cells_on_one_line(text, row):
            preserved += 1
    return preserved, total


def table_scores(rows, page_text: str, max_tokens: int | None = None) -> dict:
    """Per-table arm A vs arm B row adjacency.

    A gate-failing table is marked, never scored: arm B does not rewrite it, so a
    delta would be fiction.
    """
    if not is_safe_to_markdown(rows):
        return {
            "gate_passed": False,
            "structured_rows": 0,
            "scored_rows": 0,
            "arm_a": None,
            "arm_b": None,
            "delta": None,
            "header_a": None,
            "arm_b_tautological": True,
        }

    norm = _normalize(rows)
    header, data = (norm[0] if norm else []), norm[1:]
    md = render_markdown(rows, max_tokens=max_tokens)

    failures: dict[str, int] = {}
    for row in data:
        if len(_populated([norm_cell(c) for c in row])) < 2:
            continue
        if not cells_on_one_line(page_text, row):
            reason = classify_row_failure(page_text, row)
            failures[reason] = failures.get(reason, 0) + 1

    a_kept, total = row_adjacency(page_text, data)
    b_kept, _ = row_adjacency(md, data)
    arm_a = (a_kept / total) if total else None
    arm_b = (b_kept / total) if total else None

    return {
        "gate_passed": True,
        # Pool membership uses the survey's definition — every row with two
        # populated cells, header included — so this scores exactly the 27 tables
        # the power analysis reasoned about. The header is then scored on its own
        # (`header_a`) and excluded from the row denominator, because arm B
        # repeats it into every block and scoring it as a data row would count
        # the renderer's own bookkeeping as a win.
        "structured_rows": row_adjacency(page_text, norm)[1],
        "scored_rows": total,
        "cols": max((len(r) for r in norm), default=0),
        "arm_a": arm_a,
        "arm_b": arm_b,
        "delta": None if arm_a is None else arm_b - arm_a,
        "header_a": cells_on_one_line(page_text, header),
        # Flagged, not hidden: render_markdown emits one row per line, so arm B
        # cannot score anything but 1.0. Only arm A carries information here.
        "arm_b_tautological": True,
        "arm_a_failures": failures,
        "rendered_tokens": estimate_tokens(md),
    }


def usable_candidate(cand: dict) -> bool:
    """The 27-table pool from the power analysis §3."""
    return (
        int(cand.get("structured_rows", 0)) >= MIN_STRUCTURED_ROWS
        and int(cand.get("cols", 0)) >= MIN_COLS
    )


def summarize(records: list[dict]) -> dict:
    """Aggregate, per document as well as overall.

    The per-document means exist because 52% of the pool sits in one file. An
    overall mean would be that document's opinion wearing the corpus's name.
    """
    scored = [r for r in records if r.get("gate_passed") and r.get("arm_a") is not None]
    per_doc: dict[str, dict] = {}
    for rec in scored:
        bucket = per_doc.setdefault(rec["doc"], {"tables": 0, "_a": [], "_d": []})
        bucket["tables"] += 1
        bucket["_a"].append(rec["arm_a"])
        bucket["_d"].append(rec["delta"])
    for bucket in per_doc.values():
        bucket["mean_arm_a"] = statistics.fmean(bucket.pop("_a"))
        bucket["mean_delta"] = statistics.fmean(bucket.pop("_d"))

    if not scored:
        return {
            "tables": 0,
            "docs": 0,
            "median_arm_a": None,
            "mean_arm_a": None,
            "mean_of_doc_means_arm_a": None,
            "median_delta": None,
            "tables_fully_intact_in_arm_a": 0,
            "tables_fully_destroyed_in_arm_a": 0,
            "headers_intact_in_arm_a": 0,
            "arm_a_failure_reasons": {},
            "per_doc": {},
        }

    reasons: dict[str, int] = {}
    for rec in scored:
        for reason, n in (rec.get("arm_a_failures") or {}).items():
            reasons[reason] = reasons.get(reason, 0) + n

    a_vals = [r["arm_a"] for r in scored]
    d_vals = [r["delta"] for r in scored]
    return {
        "tables": len(scored),
        "docs": len(per_doc),
        "median_arm_a": statistics.median(a_vals),
        "mean_arm_a": statistics.fmean(a_vals),
        "mean_of_doc_means_arm_a": statistics.fmean(
            [b["mean_arm_a"] for b in per_doc.values()]
        ),
        "median_delta": statistics.median(d_vals),
        "tables_fully_intact_in_arm_a": sum(1 for v in a_vals if v == 1.0),
        "tables_fully_destroyed_in_arm_a": sum(1 for v in a_vals if v == 0.0),
        "headers_intact_in_arm_a": sum(1 for r in scored if r.get("header_a")),
        "arm_a_failure_reasons": reasons,
        "per_doc": per_doc,
    }


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def format_report(summary: dict, records: list[dict] | None = None) -> str:
    """Markdown report. States the kill condition and every caveat that bounds it."""
    records = records or []
    lines = [
        "# Table Probe — 0a Structure Delta",
        "",
        "**What this measures:** for every row carrying two or more populated cells,",
        "do those cells still sit together on one line, in document order?",
        "Arm A is the page text the pipeline indexes today",
        '(`get_text("text", sort=True)`); arm B is `table_quality.render_markdown`.',
        "",
        "**Read arm A only.** Arm B scores 1.0 by construction — `render_markdown`",
        "emits one row per line — so its column is tautological and is printed only",
        "as a sanity check that the renderer did what it claims.",
        "",
        "## Result",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| tables scored | {summary['tables']} |",
        f"| documents | {summary['docs']} |",
        f"| **median arm-A row adjacency** | **{_pct(summary['median_arm_a'])}** |",
        f"| mean arm-A row adjacency | {_pct(summary['mean_arm_a'])} |",
        f"| mean of per-document means (unweights the big doc) | {_pct(summary['mean_of_doc_means_arm_a'])} |",
        f"| median delta (arm B − arm A) | {_pct(summary['median_delta'])} |",
        f"| tables fully intact under arm A | {summary['tables_fully_intact_in_arm_a']} |",
        f"| tables fully destroyed under arm A | {summary['tables_fully_destroyed_in_arm_a']} |",
        f"| headers still on one line under arm A | {summary['headers_intact_in_arm_a']} |",
        "",
        "## Why arm-A rows failed",
        "",
        "| reason | rows |",
        "|---|---|",
    ]
    for reason, n in sorted(
        (summary.get("arm_a_failure_reasons") or {}).items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"| `{reason}` | {n} |")
    lines += [
        "",
        "- `cell_fragmented_across_columns` — the cell's own words were cut apart and",
        "  interleaved with the neighbouring column. The worst case: neither the row",
        "  nor the cell survives, and a phrase query for the cell cannot match it.",
        "- `ordered_but_split_across_lines` — a wrapped cell. No single line could have",
        "  held that row; duller than the reading-order scrambling the spec describes,",
        "  but it costs the reader the row binding just the same.",
        "- `out_of_document_order` — the scrambling proper, as the spec named it.",
        "- `cell_text_absent` — the words are not in the page text at all.",
        "",
        "Arm B's markdown puts the whole cell back on one line in every one of these",
        "cases, so the delta is real regardless of which reason dominates. Which one",
        "does dominate still matters: it is the difference between a measurement and a",
        "talking point.",
        "",
        "## Decision rule for this step",
        "",
        "- **NO-GO for the probe** if arm A already preserves most rows: there is no",
        "  adjacency left for arm B to restore, and the premise the spec rests on is",
        "  false for this corpus. Stop before authoring a single question.",
        "- **Continue to 0b** (retrieval-only rank probe) if arm A is broadly damaged.",
        "  That is evidence the mechanism exists — never evidence that answers improve.",
        "",
        "## Caveats that bound any reading",
        "",
        "- 25% of corpus pages have no text layer; `find_tables()` is blind there, so",
        "  arm B ≡ arm A on those pages and this measurement never sees them.",
        "- Document clustering is unchanged by this step: the pool is concentrated in a",
        "  few documents, so n_eff is far below the table count. The per-document column",
        "  above is the honest view; the overall mean is not.",
        "- Row adjacency is a proxy for retrievability, not for answer correctness.",
        "  A table can survive flattening and still be answered badly, and vice versa.",
        "",
    ]

    if records:
        lines += [
            "## Per document",
            "",
            "| doc | tables | mean arm A | mean delta |",
            "|---|---|---|---|",
        ]
        for doc, bucket in sorted(
            summary["per_doc"].items(), key=lambda kv: -kv[1]["tables"]
        ):
            lines.append(
                f"| `{doc[:8]}` | {bucket['tables']} | "
                f"{_pct(bucket['mean_arm_a'])} | {_pct(bucket['mean_delta'])} |"
            )
        lines += [
            "",
            "## Per table",
            "",
            "| doc | page | rows scored | arm A | arm B | header intact |",
            "|---|---|---|---|---|---|",
        ]
        for rec in sorted(records, key=lambda r: (r["doc"], r["page"])):
            if not rec.get("gate_passed") or rec.get("arm_a") is None:
                continue
            lines.append(
                f"| `{rec['doc'][:8]}` | {rec['page']} | "
                f"{rec['scored_rows']} | {_pct(rec['arm_a'])} | "
                f"{_pct(rec['arm_b'])} | {'yes' if rec['header_a'] else 'no'} |"
            )
        lines.append("")

    return "\n".join(lines)


def _chunk_max_tokens() -> int | None:
    try:
        from src.agentrag.config import settings

        return int(settings.SEARCH_CHUNK_MAX_TOKENS)
    except Exception:  # noqa: BLE001 — settings unavailable; fall back to defaults
        return None


def scan_corpus(corpus_dir: str, survey_path: str) -> list[dict]:
    """Score every usable table across the survey's unique documents.

    Runs over `unique_documents_list`, never the files on disk: the corpus carries
    87 redundant copies, and scoring them would weight duplicated documents by
    however many times they were uploaded.
    """
    import fitz  # PyMuPDF

    survey = json.loads(Path(survey_path).read_text(encoding="utf-8"))
    docs = survey["unique_documents_list"]
    max_tokens = _chunk_max_tokens()

    records: list[dict] = []
    for rel in docs:
        path = Path(corpus_dir) / rel
        if not path.exists():
            records.append({"doc": rel, "page": None, "error": "missing"})
            continue
        try:
            doc = fitz.open(str(path))
        except Exception as exc:  # noqa: BLE001 — unreadable PDF
            records.append({"doc": rel, "page": None, "error": repr(exc)})
            continue
        with doc:
            for page_num, page in enumerate(doc, start=1):
                try:
                    page_text = page.get_text("text", sort=True)
                    tables = list(page.find_tables().tables)
                except Exception:  # noqa: BLE001 — no table layer on this page
                    continue
                for idx, table in enumerate(tables):
                    try:
                        rows = table.extract()
                    except Exception:  # noqa: BLE001 — a table PyMuPDF cannot read
                        continue
                    try:
                        scores = table_scores(rows, page_text, max_tokens=max_tokens)
                    except Exception:  # noqa: BLE001 — malformed cells
                        continue
                    if not scores["gate_passed"]:
                        continue
                    if not usable_candidate(scores):
                        continue
                    records.append(
                        {"doc": rel, "page": page_num, "table": idx, **scores}
                    )
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/originals")
    ap.add_argument("--survey", default="data/eval/table_probe_corpus_survey.json")
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--report", dest="report_out")
    args = ap.parse_args()

    records = scan_corpus(args.corpus, args.survey)
    summary = summarize(records)
    report = format_report(summary, records)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps({"summary": summary, "records": records},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
