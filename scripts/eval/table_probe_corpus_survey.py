"""Survey the PDF corpus before running the table probe.

PROBE-SCOPED (2026-08-02). Answers three questions the probe design depends on,
none of which were known when the probe was specced:

  1. How many source PDFs are actually distinct? (byte-level dedup)
  2. How many pages even have a text layer? `find_tables()` is blind on the
     rest — those pages take the OCR/vision path, where arm B == arm A.
  3. Of the tables PyMuPDF detects, how many are real data grids versus layout
     boxes that rendering as markdown would corrupt? (Classified with
     `table_quality.classify_table`, the same gate arm B uses.)

Usage:
    PYTHONPATH=. uv run python scripts/eval/table_probe_corpus_survey.py \
        --corpus data/originals --json data/eval/table_probe_corpus_survey.json \
        --unique-list data/eval/table_probe_unique_docs.txt \
        --dedupe-dir data/eval/table_probe_corpus

`--dedupe-dir` materialises the deduplicated corpus as a directory of symlinks.
Spec §3 Step 0a says every later step consumes the unique-document list, and the
A/B runner is invoked with `--corpus <UNIQUE_DOCS_DIR>` — without this the only
directory that exists is the raw one with 87 redundant copies, both arms ingest
4-7 copies of every document, and duplicate chunks occupy top-k. That is the
exact failure Step 0a exists to prevent.

Exit code is always 0 — this is a measurement, not a gate. One malformed page in
one document must never discard a whole-corpus run, so every PyMuPDF call is
guarded and failures are reported in the `errors` list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentrag.common.build_info import build_info
from src.agentrag.ingestion.parsers.table_quality import (
    SAFE_KINDS,
    _normalize,
    _populated,
    classify_table,
    estimate_tokens,
    render_markdown,
)

#: Pages under this many characters take the OCR/vision path in PDFParser.parse,
#: where find_tables() sees nothing. Mirrors settings.PDF_OCR_MIN_TEXT_CHARS.
#: NOT read from `settings` at import time: `Settings()` has no defaults for
#: POSTGRES_USER/PASSWORD/DB, so importing this module without a populated .env
#: is a pytest *collection* error that aborts the whole session — in a test
#: module whose own docstring advertises it as pure.
TEXT_LAYER_MIN_CHARS_DEFAULT = 50
#: Rendered-table budget recorded per grid, mirroring SEARCH_CHUNK_MAX_TOKENS.
CHUNK_MAX_TOKENS_DEFAULT = 512


def _settings_int(name: str, default: int) -> int:
    """Read one int off `settings`, tolerating an unconfigured environment."""
    try:
        from src.agentrag.config import settings

        return int(getattr(settings, name, default))
    except Exception:  # noqa: BLE001 — no .env / no DB creds: use the default
        return default


TEXT_LAYER_MIN_CHARS = _settings_int(
    "PDF_OCR_MIN_TEXT_CHARS", TEXT_LAYER_MIN_CHARS_DEFAULT
)
CHUNK_MAX_TOKENS = _settings_int(
    "SEARCH_CHUNK_MAX_TOKENS", CHUNK_MAX_TOKENS_DEFAULT
)


def _open(path: str):
    import fitz  # PyMuPDF

    return fitz.open(path)


def unique_by_content(
    paths: list[Path], corpus_dir: Path | None = None
) -> tuple[list[Path], dict[str, list[str]]]:
    """Split paths into (first-seen unique, {sha256: [all members of that group]}).

    Files that cannot be read are collected under the reserved key
    ``"__unreadable__"`` so the caller can report them instead of letting them
    vanish into the redundant-copy arithmetic.

    Byte-level identity only. Two scans of the same source document produce
    different bytes and will not be collapsed here.

    Members are recorded as paths relative to `corpus_dir`, not bare names: the
    scan is `rglob`, so keying on `p.name` silently collapses same-named files
    living in different subdirectories. An unreadable file is skipped, not fatal
    — it would otherwise kill the run before a single document is surveyed.
    """
    groups: dict[str, list[str]] = {}
    unique: list[Path] = []
    unreadable: list[str] = []
    for p in sorted(paths):
        try:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError as exc:
            # Named, not silently dropped: a skipped file used to inflate
            # `redundant_copies` (files_on_disk - unique) and appear in the
            # summary as a duplicate that does not exist.
            unreadable.append(f"{p.name}: {exc}")
            continue
        if digest not in groups:
            groups[digest] = []
            unique.append(p)
        try:
            label = str(p.relative_to(corpus_dir)) if corpus_dir else p.name
        except ValueError:
            label = p.name
        groups[digest].append(label)
    if unreadable:
        groups["__unreadable__"] = unreadable
    return unique, groups


def survey_pdf(path: str) -> dict:
    """Per-document page and table-quality counts.

    Every PyMuPDF call is guarded and the document is always closed. A single
    page PyMuPDF cannot decode used to propagate out of here with the document
    still open and abort the whole 29-document run — discarding a ~16.5s survey
    and re-charging it on every fix attempt.
    """
    try:
        doc = _open(path)
    except Exception as exc:  # noqa: BLE001 — unreadable PDF: report, don't crash
        return {"doc": Path(path).name, "error": str(exc)}

    kinds: Counter = Counter()
    data_grids: list[dict] = []
    candidates: list[dict] = []
    page_errors: list[str] = []
    pages = 0
    text_layer_pages = 0

    # Iterating the document is the documented API and is what the mocks in
    # tests exercise. `next()` is guarded separately from the per-page work so a
    # broken iterator ends the document cleanly instead of spinning.
    try:
        try:
            pages_iter = enumerate(doc, start=1)
        except Exception as exc:  # noqa: BLE001 — document not iterable at all
            page_errors.append(f"document not iterable: {exc}")
            pages_iter = iter(())
        while True:
            try:
                page_num, page = next(pages_iter)
            except StopIteration:
                break
            except Exception as exc:  # noqa: BLE001 — iterator died; keep what we have
                page_errors.append(f"after page {pages}: {exc}")
                break
            pages = page_num
            try:
                _survey_page(page, page_num, kinds, data_grids, candidates)
                text = page.get_text("text", sort=True)
                if len(text.strip()) >= TEXT_LAYER_MIN_CHARS:
                    text_layer_pages += 1
            except Exception as exc:  # noqa: BLE001 — one bad page, not a bad run
                page_errors.append(f"page {page_num}: {exc}")
    finally:
        doc.close()

    out = {
        "doc": Path(path).name,
        "pages": pages,
        "text_layer_pages": text_layer_pages,
        "detected_tables": sum(kinds.values()),
        "kinds": dict(kinds),
        "data_grids": data_grids,
        "data_grid_count": len(data_grids),
        "candidate_tables": candidates,
        "candidate_count": len(candidates),
    }
    if page_errors:
        out["page_errors"] = page_errors
    return out


def _survey_page(
    page,
    page_num: int,
    kinds: Counter,
    data_grids: list[dict],
    candidates: list[dict] | None = None,
) -> None:
    """Classify every table on one page. Raises only on a page-level failure.

    `data_grids` collects `real_data` only — the RANKING class. `candidates`
    collects every gate-passing table, which is the ELIGIBLE set: arm B rewrites
    `nonnumeric` tables too, and a text-only comparison matrix carries column
    meaning just as a numeric grid does. Confusing the two undercounts the
    probe's usable pool by a third (19 vs 27 structured tables).
    """
    try:
        tables = list(page.find_tables().tables)
    except Exception:  # noqa: BLE001 — no table layer on this page
        return
    for table in tables:
        try:
            rows = table.extract()
        except Exception:  # noqa: BLE001 — a table PyMuPDF cannot read
            kinds["degenerate"] += 1
            continue
        try:
            kind = classify_table(rows)
        except Exception:  # noqa: BLE001 — malformed cells
            kinds["degenerate"] += 1
            continue
        kinds[kind] += 1
        if candidates is not None and kind in SAFE_KINDS:
            norm = _normalize(rows)
            md = render_markdown(rows, max_tokens=CHUNK_MAX_TOKENS)
            candidates.append(
                {
                    "page": page_num,
                    "kind": kind,
                    "rows": len(norm),
                    # Rows that actually populate two columns — the only ones a
                    # row/column-alignment question can be asked about.
                    "structured_rows": sum(
                        1 for r in norm if len(_populated(r)) >= 2
                    ),
                    "cols": max((len(r) for r in norm), default=0),
                    "est_tokens": estimate_tokens(md),
                    "max_block_tokens": max(
                        (estimate_tokens(b) for b in md.split("\n\n") if b),
                        default=0,
                    ),
                }
            )
        if kind != "real_data":
            continue
        try:
            rc, cc = table.row_count, table.col_count
        except Exception:  # noqa: BLE001 — dimensions unavailable
            rc = len(rows or [])
            cc = max((len(r) for r in rows or []), default=0)
        data_grids.append(
            {
                "page": page_num,
                "rows": rc,
                "cols": cc,
                # Measured, because cell count predicts rendered size badly.
                # `rank_targets(max_tokens=...)` filters on this.
                "est_tokens": estimate_tokens(
                    render_markdown(rows, max_tokens=CHUNK_MAX_TOKENS)
                ),
                # The number that decides whether a table survives chunking is
                # the LARGEST BLOCK, not the whole-table total: render_markdown
                # emits a big table as several under-budget blocks, every one of
                # which the chunker keeps whole. Filtering on the total would
                # discard exactly the large multi-row grids the probe is about.
                "max_block_tokens": max(
                    (
                        estimate_tokens(block)
                        for block in render_markdown(
                            rows, max_tokens=CHUNK_MAX_TOKENS
                        ).split("\n\n")
                        if block
                    ),
                    default=0,
                ),
            }
        )


def survey_corpus(corpus_dir: str) -> dict:
    root = Path(corpus_dir)
    all_pdfs = sorted(root.rglob("*.pdf"))
    unique, groups = unique_by_content(all_pdfs, root)
    unreadable = groups.pop("__unreadable__", [])
    readable = len(all_pdfs) - len(unreadable)

    docs = []
    for p in unique:
        try:
            docs.append(survey_pdf(str(p)))
        except Exception as exc:  # noqa: BLE001 — never lose the whole run
            docs.append({"doc": p.name, "error": str(exc)})
    ok = [d for d in docs if "error" not in d]

    kinds: Counter = Counter()
    for d in ok:
        kinds.update(d["kinds"])

    pages = sum(d["pages"] for d in ok)
    text_layer = sum(d["text_layer_pages"] for d in ok)
    return {
        # Which code produced these numbers. A stale image on a `-f` compose
        # path yields a complete, plausible report from the wrong classifier.
        "build": build_info(),
        "corpus_dir": corpus_dir,
        # Arm-independent corpus identity. `eval/corpus_fingerprint.py` hashes
        # (document_title, segment_count), which arm B changes BY DESIGN — it
        # adds table blocks — so that fingerprint cannot gate this probe. This
        # one hashes the deduplicated document CONTENT, so it is identical under
        # both arms and still catches "someone added or removed a PDF between
        # authoring the evalset and running it", the drift that would silently
        # invalidate the (doc, page) gold.
        "corpus_docs_sha": hashlib.sha1(
            "\n".join(sorted(digest for digest in groups)).encode("utf-8")
        ).hexdigest()[:12],
        "text_layer_min_chars": TEXT_LAYER_MIN_CHARS,
        "chunk_max_tokens": CHUNK_MAX_TOKENS,
        "files_on_disk": len(all_pdfs),
        "unreadable_files": unreadable,
        "unique_documents": len(unique),
        # The LIST, not just the count. Spec §3 Step 0a says every later step
        # consumes it; only the integer was ever emitted, so `--corpus
        # <UNIQUE_DOCS_DIR>` pointed at a directory nothing produced.
        "unique_documents_list": [str(p.relative_to(root)) for p in unique],
        "redundant_copies": readable - len(unique),
        "duplicate_groups": {
            members[0]: members for members in groups.values() if len(members) > 1
        },
        "pages": pages,
        "text_layer_pages": text_layer,
        "no_text_layer_pages": pages - text_layer,
        "detected_tables": sum(kinds.values()),
        "table_kinds": dict(kinds),
        "data_grids": sum(d["data_grid_count"] for d in ok),
        "docs_with_data_grids": sum(1 for d in ok if d["data_grid_count"] > 0),
        # The ELIGIBLE set: every gate-passing table, and of those the ones with
        # enough structure to carry a row/column-alignment question. See
        # docs/eval/table_probe_power_analysis_2026-09-06.md.
        "candidate_tables": sum(d.get("candidate_count", 0) for d in ok),
        "structured_candidates": sum(
            1
            for d in ok
            for c in d.get("candidate_tables", [])
            if c["structured_rows"] >= 3 and c["cols"] >= 3
        ),
        "errors": [d for d in docs if "error" in d],
        "page_errors": {
            d["doc"]: d["page_errors"] for d in ok if d.get("page_errors")
        },
        "per_doc": sorted(ok, key=lambda d: -d["data_grid_count"]),
    }


def _identity(path: Path) -> tuple[int, int] | None:
    """(st_dev, st_ino) of `path`, or None when it does not exist.

    Used instead of comparing Path objects because path comparison is
    case-SENSITIVE and `resolve()` does not case-normalise: on a case-insensitive
    filesystem (/mnt/c under WSL, default macOS APFS) `corpus/unique` and
    `Corpus/unique` compare as different directories while being the same one.
    Inode identity is also immune to bind mounts and Windows 8.3 short names.
    Deliberately does NOT walk up to an existing ancestor — a missing path has no
    identity, and substituting its parent's makes unrelated siblings look nested.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def _overlaps(dest: Path, root: Path) -> bool:
    """True when clearing `dest` could touch anything under `root`."""
    # Textual containment first: cheap, and works for paths that do not exist.
    if dest == root or root in dest.parents or dest in root.parents:
        return True

    root_id, dest_id = _identity(root), _identity(dest)
    if root_id is not None and root_id == dest_id:
        return True
    if root_id is not None and any(_identity(p) == root_id for p in dest.parents):
        return True  # dest lives inside the corpus under a different spelling
    # ...or the corpus lives inside dest under a different spelling.
    return dest_id is not None and any(
        _identity(p) == dest_id for p in root.parents
    )


def write_dedupe_dir(corpus_dir: str, unique_rel: list[str], out_dir: str) -> int:
    """Materialise the deduplicated corpus as symlinks. Returns the count.

    Three things this must never do, because every one of them destroys the
    operator's source corpus rather than just failing:

    - **Write into the corpus.** `--corpus` defaults to `data/originals`, and the
      two flags sit on adjacent lines in the usage block. Clearing `dest` when
      `dest` IS the corpus deletes every original and replaces it with dangling
      self-symlinks, exit code 0.
    - **Follow a symlink when writing.** `link.write_bytes()` through an existing
      symlink truncates its *target* — a file inside the corpus. Links are
      removed before being rewritten, and the copy fallback only ever runs on a
      path that does not exist.
    - **Collapse two distinct documents onto one name.** `rglob` finds nested
      files, and `unique_by_content` was fixed to keep their relative paths, so
      naming the link after `src.name` alone can collide. Nested paths are
      flattened with the separator preserved, and a residual collision is
      raised, not silently resolved.
    """
    root = Path(corpus_dir).resolve()
    dest = Path(out_dir).resolve()
    if _overlaps(dest, root):
        raise ValueError(
            f"--dedupe-dir ({dest}) must not be the corpus directory or contain "
            f"it / live inside it ({root}); that would delete the originals"
        )
    dest.mkdir(parents=True, exist_ok=True)

    # Only ever clear PDFs this function would itself have written. The dedupe
    # dir is given on the command line next to `--json data/eval/...`, so a
    # dropped path segment (`--dedupe-dir data/eval`) used to unlink every eval
    # artefact in that directory and exit 0. Refuse instead.
    foreign = sorted(
        e.name
        for e in dest.iterdir()
        if not e.is_dir() and e.suffix.lower() != ".pdf"
    )
    if foreign:
        raise ValueError(
            f"--dedupe-dir ({dest}) holds {len(foreign)} non-PDF file(s) — "
            f"{', '.join(foreign[:5])}. Refusing to clear a directory this tool "
            "does not own; point it at a dedicated empty directory."
        )
    for stale in dest.iterdir():
        if not stale.is_dir():
            stale.unlink()

    written = 0
    used: dict[str, str] = {}
    missing: list[str] = []
    for rel in unique_rel:
        src = root / rel
        if not src.exists():
            missing.append(rel)
            continue
        name = rel.replace("/", "__").replace("\\", "__")
        if name in used:
            raise ValueError(
                f"dedupe name collision: {used[name]!r} and {rel!r} both map to "
                f"{name!r}"
            )
        used[name] = rel
        link = dest / name
        if link.is_symlink() or link.exists():
            link.unlink()
        try:
            link.symlink_to(src)
        except OSError:  # no symlink support on this filesystem — copy instead
            link.write_bytes(src.read_bytes())
        written += 1

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} unique document(s) missing from {root}: "
            + ", ".join(missing[:5])
        )
    return written


def format_summary(rep: dict) -> str:
    pages = max(rep["pages"], 1)
    detected = max(rep["detected_tables"], 1)
    lines = [
        f"corpus: {rep['corpus_dir']}",
        f"  files on disk      : {rep['files_on_disk']}",
        (
            f"  unique documents   : {rep['unique_documents']}"
            f"  (redundant copies: {rep['redundant_copies']})"
        ),
        f"  pages              : {rep['pages']}",
        (
            f"  no text layer      : {rep['no_text_layer_pages']}"
            f" ({rep['no_text_layer_pages'] / pages * 100:.0f}%) — find_tables blind here"
        ),
        f"  detected tables    : {rep['detected_tables']}",
    ]
    for kind, n in sorted(rep["table_kinds"].items(), key=lambda kv: -kv[1]):
        lines.append(f"      {kind:<13} {n:>4}  ({n / detected * 100:.0f}%)")
    lines += [
        (
            f"  data grids         : {rep['data_grids']}"
            f" across {rep['docs_with_data_grids']} docs  (ranking class only)"
        ),
        (
            f"  ELIGIBLE tables    : {rep.get('candidate_tables', 0)} gate-passing,"
            f" {rep.get('structured_candidates', 0)} with >=3 structured rows and"
            " >=3 cols"
        ),
        "",
        "candidate docs for eval authoring (by data-grid count):",
    ]
    for d in rep["per_doc"][:10]:
        if not d["data_grid_count"]:
            break
        lines.append(
            f"  {d['data_grid_count']:>3} grids  {d['doc']}"
            f"  (pages={d['pages']}, text_layer={d['text_layer_pages']})"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/originals")
    ap.add_argument("--json", dest="json_out", help="write the full report here")
    ap.add_argument(
        "--unique-list", help="write the unique-document paths here, one per line"
    )
    ap.add_argument(
        "--dedupe-dir",
        help="materialise the deduplicated corpus here (symlinks); pass THIS to "
        "run_table_probe_ab.py --corpus",
    )
    args = ap.parse_args()

    report = survey_corpus(args.corpus)
    print(format_summary(report))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n-> {args.json_out}")
    if args.unique_list:
        Path(args.unique_list).parent.mkdir(parents=True, exist_ok=True)
        Path(args.unique_list).write_text(
            "\n".join(report["unique_documents_list"]) + "\n", encoding="utf-8"
        )
        print(f"-> {args.unique_list}")
    if args.dedupe_dir:
        n = write_dedupe_dir(
            args.corpus, report["unique_documents_list"], args.dedupe_dir
        )
        print(f"-> {args.dedupe_dir} ({n} unique documents)")
