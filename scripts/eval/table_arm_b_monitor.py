"""Arm-B rollout monitor — what gets measured on production traffic.

Pre-registered before the flag is turned on, per condition 1 of the ship
decision (`docs/eval/table_arm_b_rollout_2026-09-06.md`). It exists because
"measure it on production" is otherwise assertable but not measurable: in
production there is no `extract()` ground truth for an arbitrary user question,
which is what every number in 0c rested on.

Two things ARE decidable without ground truth, and both are the mechanism 0c
found rather than proxies invented to have something to plot:

1. **Abstention rate** on answers citing a table-bearing page. 0c measured
   5 → 2 abstentions across 19 lookups; if that was real, the live rate should
   fall. Uses the shipped `_UNCERTAINTY_MARKERS`, not a private copy.
2. **Lookup overrun rate** — a SHORT, lookup-shaped answer whose tokens are
   drawn from two or more distinct rows of the cited table. That is exactly the
   cell-boundary bleed 0c isolated (90% of arm A's surplus tokens came from
   other cells of the same table, against 0% for arm B), and it needs no gold
   answer to compute. Long answers are excluded: they may legitimately
   summarise several rows, and flagging them would manufacture a signal.

Attribution comes from the segment stamp added at ingest
(`pipeline.build_chunk_metadata`), resolved through each citation's
`content_hash`. An answer citing segments from both arms is reported as
`mixed` and never folded into A or B — mid-rollout that is the transition
itself, not evidence about either arm.

What this is NOT: it is not a controlled A/B. Documents ingested before and
after the flip differ in more than the flag (corpus growth, model drift,
question mix). Read it as a guardrail and a directional check, and see the
rollout doc for the honest name of what a one-armed rollout can conclude.

Usage:
    PYTHONPATH=. uv run python scripts/eval/table_arm_b_monitor.py \
        --from-db --since 2026-09-06 --json data/eval/arm_b_monitor.json
    PYTHONPATH=. uv run python scripts/eval/table_arm_b_monitor.py \
        --from-json data/eval/answers_dump.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval.table_probe_comprehension_ab import normalize_answer  # noqa: E402
from scripts.eval.table_probe_retrieval_ab import tokenize  # noqa: E402

#: An answer longer than this is not a cell lookup, so spanning several rows is
#: not evidence of a lost cell boundary.
LOOKUP_MAX_TOKENS = 60


def is_abstention(text: str) -> bool:
    """Reuses the shipped uncertainty markers, so this cannot drift from prod."""
    from src.agentrag.agent.service import _UNCERTAINTY_MARKERS

    low = (text or "").lower()
    return any(marker in low for marker in _UNCERTAINTY_MARKERS)


def _row_fingerprints(table_cells) -> list[set[str]]:
    """Tokens unique to each data row — tokens shared by rows identify nothing."""
    rows = [
        {tok for cell in row for tok in tokenize(normalize_answer(cell or ""))}
        for row in (table_cells[1:] if table_cells else [])
    ]
    seen: dict[str, int] = defaultdict(int)
    for row in rows:
        for tok in row:
            seen[tok] += 1
    header = (
        {tok for cell in table_cells[0] for tok in tokenize(normalize_answer(cell or ""))}
        if table_cells
        else set()
    )
    return [{tok for tok in row if seen[tok] == 1 and tok not in header} for row in rows]


def rows_touched(answer: str, table_cells) -> int:
    """How many distinct data rows the answer draws identifying tokens from."""
    answer_tokens = set(tokenize(normalize_answer(answer)))
    return sum(1 for row in _row_fingerprints(table_cells) if row & answer_tokens)


def is_lookup_overrun(answer: str, table_cells, max_tokens: int = LOOKUP_MAX_TOKENS) -> bool:
    """A short answer that pulls from two or more rows lost the cell boundary."""
    if len(tokenize(normalize_answer(answer))) > max_tokens:
        return False
    return rows_touched(answer, table_cells) >= 2


def arm_of_answer(citations, segments_by_hash) -> str:
    """'A' | 'B' | 'mixed' | 'unknown', from the ingest-time segment stamp."""
    stamps = set()
    for citation in citations or []:
        meta = segments_by_hash.get(citation.get("content_hash"))
        if not meta or "pdf_preserve_tables" not in meta:
            continue
        stamps.add(bool(meta["pdf_preserve_tables"]))
    if not stamps:
        return "unknown"
    if stamps == {True}:
        return "B"
    if stamps == {False}:
        return "A"
    return "mixed"


def summarize_by_arm(records: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        grouped[rec["arm"]].append(rec)

    out: dict[str, dict] = {}
    for arm, rows in grouped.items():
        latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        out[arm] = {
            "answers": len(rows),
            "abstention_rate": sum(1 for r in rows if r.get("abstained")) / len(rows),
            "overrun_rate": sum(1 for r in rows if r.get("overrun")) / len(rows),
            "median_latency_ms": statistics.median(latencies) if latencies else None,
        }
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_from_db(since: str | None = None) -> tuple[list[dict], dict]:
    """Assistant answers + the segment stamps their citations resolve to.

    The app's session factory is async (`AsyncSessionLocal`), so this drives it
    through `asyncio.run` rather than inventing a second, sync engine.
    """
    import asyncio

    from sqlalchemy import select

    from src.agentrag.database import AsyncSessionLocal
    from src.agentrag.database.models import ChatMessage, Segment

    async def go():
        async with AsyncSessionLocal() as session:
            stmt = select(ChatMessage).where(ChatMessage.role == "assistant")
            if since:
                stmt = stmt.where(ChatMessage.created_at >= since)
            rows = (await session.execute(stmt)).scalars().all()
            messages = [
                {
                    "content": m.content,
                    "citations": m.citations or [],
                    "timings_ms": m.timings_ms or {},
                    "created_at": str(m.created_at),
                }
                for m in rows
            ]
            hashes = {
                c.get("content_hash")
                for m in messages
                for c in m["citations"]
                if c.get("content_hash")
            }
            segments = {}
            if hashes:
                seg_rows = (
                    await session.execute(
                        select(Segment).where(Segment.content_hash.in_(hashes))
                    )
                ).scalars().all()
                segments = {s.content_hash: (s.extra_metadata or {}) for s in seg_rows}
            return messages, segments

    return asyncio.run(go())


def score_answers(messages: list[dict], segments: dict, tables_by_page) -> list[dict]:
    """One record per answer. `tables_by_page` maps (title, page) -> extract() cells."""
    records = []
    for msg in messages:
        citations = msg.get("citations") or []
        cells = None
        for citation in citations:
            title = citation.get("document_title")
            for page in citation_pages(citation):
                if (title, page) in tables_by_page:
                    cells = tables_by_page[(title, page)]
                    break
            if cells:
                break
        timings = msg.get("timings_ms") or {}
        records.append(
            {
                "arm": arm_of_answer(citations, segments),
                "abstained": is_abstention(msg.get("content", "")),
                "overrun": bool(cells) and is_lookup_overrun(msg.get("content", ""), cells),
                "cited_a_table": bool(cells),
                # graph_service writes {"total": <ms>}; the alternates are for
            # older rows that used a different key.
            "latency_ms": timings.get("total")
            or timings.get("total_ms")
            or timings.get("answer_ms"),
                "created_at": msg.get("created_at"),
            }
        )
    return records


def resolve_pdf(corpus_dir: str, document_title: str) -> Path | None:
    """Locate the source PDF for a cited document title.

    Ingest sets `Document.title` to the file's stem and `source_id` to the
    filename, so the citation's `document_title` maps straight onto the corpus.
    Returns None rather than guessing when it does not.
    """
    if not document_title:
        return None
    candidate = Path(corpus_dir) / f"{document_title}.pdf"
    return candidate if candidate.exists() else None


def citation_pages(citation: dict) -> list[int]:
    """Every page a citation covers.

    A chunk spanning pages is cited as a RANGE — production rows carry
    `page` values like `"22-24"`, not just integers — so a bare `int(page)`
    raises on real data. `page_start`/`page_end` are preferred when present.
    """
    start, end = citation.get("page_start"), citation.get("page_end")
    if isinstance(start, int):
        last = end if isinstance(end, int) else start
        return list(range(start, max(last, start) + 1))

    raw = citation.get("page") or citation.get("page_start")
    if isinstance(raw, int):
        return [raw]
    if not isinstance(raw, str):
        return []
    text = raw.strip().lstrip("p.").strip()
    if "-" in text:
        lo, _, hi = text.partition("-")
        try:
            lo_i, hi_i = int(lo), int(hi)
        except ValueError:
            return []
        return list(range(lo_i, max(hi_i, lo_i) + 1))
    try:
        return [int(text)]
    except ValueError:
        return []


def build_tables_for_citations(corpus_dir: str, messages: list[dict]) -> dict:
    """{(title, page): extract() cells} for only the pages actually cited.

    Built on demand: scanning the whole corpus to answer a handful of citations
    would cost minutes per run and most of it would be thrown away.
    """
    import fitz

    from src.agentrag.ingestion.parsers.table_quality import is_safe_to_markdown

    wanted: set[tuple[str, int]] = set()
    for msg in messages:
        for citation in msg.get("citations") or []:
            title = citation.get("document_title")
            if not title:
                continue
            for page in citation_pages(citation):
                wanted.add((title, page))

    by_doc: dict[str, set[int]] = defaultdict(set)
    for title, page in wanted:
        by_doc[title].add(page)

    out: dict[tuple[str, int], list] = {}
    for title, pages in by_doc.items():
        path = resolve_pdf(corpus_dir, title)
        if path is None:
            continue
        try:
            doc = fitz.open(str(path))
        except Exception:  # noqa: BLE001 — unreadable PDF
            continue
        with doc:
            for page_num in sorted(pages):
                if not (1 <= page_num <= doc.page_count):
                    continue
                try:
                    tables = list(doc[page_num - 1].find_tables().tables)
                except Exception:  # noqa: BLE001 — no table layer
                    continue
                for table in tables:
                    try:
                        cells = table.extract()
                        if is_safe_to_markdown(cells):
                            out[(title, page_num)] = cells
                            break
                    except Exception:  # noqa: BLE001 — unreadable table
                        continue
    return out


def format_report(summary: dict, table_citing: int, total: int) -> str:
    lines = [
        "# Arm-B Rollout Monitor",
        "",
        f"Answers examined: **{total}**, of which **{table_citing}** cite a page that",
        "carries a detected table (the only ones the overrun metric applies to).",
        "",
        "| arm | answers | abstention rate | lookup-overrun rate | median latency |",
        "|---|---|---|---|---|",
    ]
    for arm in ("A", "B", "mixed", "unknown"):
        row = summary.get(arm)
        if not row:
            continue
        latency = "—" if row["median_latency_ms"] is None else f"{row['median_latency_ms']:.0f} ms"
        lines.append(
            f"| {arm} | {row['answers']} | {row['abstention_rate']:.2f} | "
            f"{row['overrun_rate']:.2f} | {latency} |"
        )
    lines += [
        "",
        "**`mixed` is answers citing segments from both arms** — mid-rollout that is",
        "the transition, not evidence about either arm, so it is reported and never",
        "folded in. `unknown` is answers whose citations carry no ingest stamp",
        "(segments written before the stamp existed).",
        "",
        "**This is not a controlled A/B.** Documents ingested before and after the flip",
        "differ in more than the flag. Read this as a guardrail and a directional",
        "check; see `docs/eval/table_arm_b_rollout_2026-09-06.md` for what a one-armed",
        "rollout can and cannot conclude.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-db", action="store_true")
    src.add_argument("--from-json", help="dump of {messages: [...], segments: {...}}")
    ap.add_argument("--since", help="ISO date; only answers at or after it")
    ap.add_argument("--tables", help="JSON map of table cells by 'title|page'")
    ap.add_argument("--build-tables", metavar="CORPUS_DIR",
                    help="derive the table map from the cited pages of this corpus")
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--report", dest="report_out")
    args = ap.parse_args()

    if args.from_db:
        messages, segments = load_from_db(args.since)
    else:
        blob = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        messages, segments = blob["messages"], blob.get("segments", {})

    tables_by_page = {}
    if args.build_tables:
        tables_by_page = build_tables_for_citations(args.build_tables, messages)
        print(f"[monitor] {len(tables_by_page)} cited pages carry a gate-passing table",
              file=sys.stderr)
    if args.tables:
        raw = json.loads(Path(args.tables).read_text(encoding="utf-8"))
        tables_by_page = {(k.split("|")[0], int(k.split("|")[1])): v for k, v in raw.items()}

    records = score_answers(messages, segments, tables_by_page)
    summary = summarize_by_arm(records)
    report = format_report(
        summary, sum(1 for r in records if r["cited_a_table"]), len(records)
    )

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
