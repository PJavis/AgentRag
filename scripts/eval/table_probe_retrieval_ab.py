"""0b — retrieval-only rank probe for the table-data probe.

Step 0b of `docs/superpowers/plans/2026-09-06-table-probe-unblock.md`. 0a showed
that flattening destroys row adjacency (median 17% of rows survive). That is a
statement about *text*. 0b asks the next question — the cheapest one that can
still kill the probe:

    does arm B actually make the gold row easier to RETRIEVE?

If it does not, the answer model never sees the row and no end-to-end gain is
possible, whatever a judge would say. That is a NO-GO bought for a day of compute
instead of the weeks the full probe costs.

Design choices that keep this honest:

- **Queries are generated mechanically** — `"<row label> <column header>"` from
  `extract()` — so nothing is hand-tuned toward either arm. They are therefore
  NOT user questions, and this step can kill the probe but never green-light a
  build on its own.
- **The gold chunk is defined arm-neutrally**: right page, and containing most of
  the gold row's *tokens* (bag of words), never a contiguous substring. Arm A
  shreds cells across lines; a substring-based gold would make arm A unable to
  have a gold chunk at all, which would rig the comparison at the definition.
- **Outcome is continuous** (reciprocal rank), not the binary the sign test
  consumes, so it carries magnitude and needs far less n for the same effect.
- **The interval resamples documents, not tables.** 52% of the pool sits in one
  file; per-table resampling would report false precision.
- **BM25 here is a simplified lexical baseline** (whitespace + lowercase), not
  Elasticsearch's Vietnamese analyzer. It is identical across arms, so the delta
  is fair, but the absolute numbers are not production retrieval.

Usage:
    PYTHONPATH=. uv run python scripts/eval/table_probe_retrieval_ab.py \
        --corpus data/originals \
        --survey data/eval/table_probe_corpus_survey.json \
        --json data/eval/table_probe_retrieval_ab.json \
        --report docs/eval/table_probe_retrieval_ab_<date>.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agentrag.ingestion.parsers.table_quality import (  # noqa: E402
    _normalize,
    _populated,
    is_safe_to_markdown,
)

#: Distinct tokens a gold row needs before it can be identified without a
#: page filter. Below this it is dropped from the comparison, never guessed at.
MIN_GOLD_TOKENS = 6
#: Share of a gold row's tokens a chunk must hold before it counts as gold.
GOLD_TOKEN_FRAC = 0.5
#: Standard RRF constant.
RRF_K = 60

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+", re.UNICODE)


def norm(text) -> str:
    if text is None:
        return ""
    return _WS.sub(" ", str(text)).strip()


def tokenize(text) -> list[str]:
    return _TOKEN.findall(norm(text).lower())


def pick_gold_row(rows) -> int | None:
    """Index of the first data row carrying three or more populated cells.

    Three, not two: a row/column-alignment question needs a label, a column and
    an answer. Deterministic, so nothing is cherry-picked.
    """
    normed = _normalize(rows)
    for idx, row in enumerate(normed):
        if idx == 0:
            continue
        if len(_populated([norm(c) for c in row])) >= 3:
            return idx
    return None


def build_query(rows, gold_idx: int) -> dict | None:
    """`"<row label> <column header>"` — mechanical, never hand-tuned.

    Returns None when the table cannot carry the question: no row label, or the
    target column has no header to name.
    """
    normed = _normalize(rows)
    if gold_idx <= 0 or gold_idx >= len(normed):
        return None
    header = [norm(c) for c in normed[0]]
    row = [norm(c) for c in normed[gold_idx]]

    label_col = next((i for i, c in enumerate(row) if c), None)
    if label_col is None:
        return None
    label = row[label_col]

    target = next(
        (
            i
            for i in range(len(row) - 1, -1, -1)
            if i != label_col and i < len(header) and row[i] and header[i]
        ),
        None,
    )
    if target is None:
        return None

    return {
        "query": f"{label} {header[target]}",
        "row_label": label,
        "column": header[target],
        "answer_cell": row[target],
        "gold_cells": [c for c in row if c],
    }


def gold_chunk_ids(chunks, gold_cells, min_frac: float = GOLD_TOKEN_FRAC,
                   min_unique_tokens: int = MIN_GOLD_TOKENS):
    """Chunks holding most of the gold row's tokens. Bag of words, no page filter.

    Two deliberate choices, both about not rigging the comparison:

    - **Bag of words, never a contiguous substring.** Arm A shreds cells across
      lines; a substring test would make arm A unable to *have* a gold chunk and
      would decide the result at the definition.
    - **No page filter.** The chunker labels a chunk by the page markers inside
      it, so a chunk carrying page 16's tail plus the page-17 marker is labelled
      page 17. Arm B's longer pages shift those boundaries more often, so a page
      filter silently deletes arm B's gold chunks — measured: it cost arm B a
      whole question on 5164c4af p16. Rows with fewer than `min_unique_tokens`
      distinct tokens are too generic to identify without the page and are
      dropped from the comparison instead, and counted.
    """
    unique = {t for cell in gold_cells for t in tokenize(cell)}
    if len(unique) < min_unique_tokens:
        return []
    need = max(1, math.ceil(len(unique) * min_frac))
    return [
        idx
        for idx, chunk in enumerate(chunks)
        if len(unique & set(tokenize(chunk.get("content", "")))) >= need
    ]


def row_is_identifiable(gold_cells, min_unique_tokens: int = MIN_GOLD_TOKENS) -> bool:
    """False when the row's own words cannot single it out inside its document."""
    return len({t for cell in gold_cells for t in tokenize(cell)}) >= min_unique_tokens


def rank_of(ranked_ids, gold_ids) -> int | None:
    """1-based rank of the best gold item, or None when none was retrieved."""
    gold = set(gold_ids)
    for pos, item in enumerate(ranked_ids, start=1):
        if item in gold:
            return pos
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if not rank else 1.0 / rank


def bm25_scores(docs_tokens, query_tokens, k1: float = 1.5, b: float = 0.75):
    """Okapi BM25 over pre-tokenized documents.

    Deliberately simple and identical across arms. Not Elasticsearch's analyzer —
    the deltas are fair, the absolute scores are not production retrieval.
    """
    n = len(docs_tokens)
    if not n:
        return []
    lengths = [len(d) for d in docs_tokens]
    avgdl = (sum(lengths) / n) or 1.0
    freqs = [defaultdict(int) for _ in docs_tokens]
    for bag, tokens in zip(freqs, docs_tokens):
        for tok in tokens:
            bag[tok] += 1
    df = defaultdict(int)
    for bag in freqs:
        for tok in bag:
            df[tok] += 1

    scores = [0.0] * n
    for term in set(query_tokens):
        if term not in df:
            continue
        idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        for i, bag in enumerate(freqs):
            tf = bag.get(term, 0)
            if not tf:
                continue
            denom = tf + k1 * (1 - b + b * lengths[i] / avgdl)
            scores[i] += idf * (tf * (k1 + 1)) / denom
    return scores


def rrf_fuse(rankings, k: int = RRF_K) -> list[int]:
    """Reciprocal-rank fusion of several ranked id lists."""
    score: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for pos, item in enumerate(ranking, start=1):
            score[item] += 1.0 / (k + pos)
    return [item for item, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def bootstrap_doc_clustered_ci(per_doc, iterations: int = 2000, seed: int = 0):
    """95% CI for the mean, resampling DOCUMENTS with replacement.

    The pool is concentrated — 52% of tables in one file — so tables within a
    document are not independent draws. Resampling documents keeps the interval
    honest about that; resampling tables would not.
    """
    docs = [vals for vals in per_doc.values() if vals]
    if not docs:
        return (None, None)
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        picked = [docs[rng.randrange(len(docs))] for _ in range(len(docs))]
        flat = [v for group in picked for v in group]
        if flat:
            means.append(statistics.fmean(flat))
    if not means:
        return (None, None)
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return (lo, hi)


# ---------------------------------------------------------------------------
# Corpus wiring
# ---------------------------------------------------------------------------


def _chunker():
    from src.agentrag.config import settings
    from src.agentrag.ingestion.chunkers.hybrid_chunker import HybridChunker

    return HybridChunker(
        max_tokens=settings.SEARCH_CHUNK_MAX_TOKENS,
        overlap_tokens=settings.SEARCH_CHUNK_OVERLAP_TOKENS,
        tokenizer_model=settings.CHUNK_TOKENIZER_MODEL,
        split_on_headings=True,
        split_on_paragraphs=settings.SEARCH_CHUNK_BY_PARAGRAPH,
    )


def build_arm(corpus_dir: str, docs: list[str], preserve_tables: bool) -> list[dict]:
    """Parse + chunk every unique document under one arm. Production code path."""
    from src.agentrag.config import settings
    from src.agentrag.ingestion.parsers.pdf_parser import PDFParser

    previous = settings.PDF_PRESERVE_TABLES
    previous_vision = settings.PDF_OCR_VISION_FALLBACK
    settings.PDF_PRESERVE_TABLES = preserve_tables
    # Vision OCR calls a remote model: non-deterministic, and it fails outright
    # when the endpoint is unreachable. Off for both arms, so the delta is clean.
    settings.PDF_OCR_VISION_FALLBACK = False
    chunker = _chunker()
    parser = PDFParser()
    out: list[dict] = []
    try:
        for rel in docs:
            path = Path(corpus_dir) / rel
            if not path.exists():
                continue
            try:
                parsed = parser.parse(str(path))
            except Exception as exc:  # noqa: BLE001 — unreadable PDF
                print(f"  ! parse failed {rel}: {exc!r}", file=sys.stderr)
                continue
            for chunk in chunker.chunk(parsed["parsed_content"], {"doc": rel}):
                out.append(
                    {
                        "doc": rel,
                        "content": chunk["content"],
                        "page_start": chunk.get("page_start"),
                        "page_end": chunk.get("page_end"),
                    }
                )
    finally:
        settings.PDF_PRESERVE_TABLES = previous
        settings.PDF_OCR_VISION_FALLBACK = previous_vision
    return out


def collect_questions(corpus_dir: str, docs: list[str]) -> list[dict]:
    """One mechanical query per usable table, over the same pool as 0a."""
    import fitz

    from scripts.eval.table_probe_structure_delta import usable_candidate

    questions: list[dict] = []
    for rel in docs:
        path = Path(corpus_dir) / rel
        if not path.exists():
            continue
        try:
            doc = fitz.open(str(path))
        except Exception:  # noqa: BLE001 — unreadable PDF
            continue
        with doc:
            for page_num, page in enumerate(doc, start=1):
                try:
                    tables = list(page.find_tables().tables)
                except Exception:  # noqa: BLE001 — no table layer
                    continue
                for t_idx, table in enumerate(tables):
                    try:
                        rows = table.extract()
                    except Exception:  # noqa: BLE001
                        continue
                    try:
                        if not is_safe_to_markdown(rows):
                            continue
                        normed = _normalize(rows)
                        structured = sum(
                            1 for r in normed if len(_populated([norm(c) for c in r])) >= 2
                        )
                        cols = max((len(r) for r in normed), default=0)
                        if not usable_candidate(
                            {"structured_rows": structured, "cols": cols}
                        ):
                            continue
                        gold_idx = pick_gold_row(rows)
                        if gold_idx is None:
                            questions.append(
                                {"doc": rel, "page": page_num, "table": t_idx,
                                 "skipped": "no row with 3 populated cells"}
                            )
                            continue
                        q = build_query(rows, gold_idx)
                        if q is None:
                            questions.append(
                                {"doc": rel, "page": page_num, "table": t_idx,
                                 "skipped": "no named column to ask about"}
                            )
                            continue
                    except Exception:  # noqa: BLE001 — malformed cells
                        continue
                    questions.append(
                        {"doc": rel, "page": page_num, "table": t_idx, **q}
                    )
    return questions


async def _embed_all(texts: list[str], batch: int = 32) -> list[list[float]]:
    from src.agentrag.services.embedding_service import EmbeddingService

    svc = EmbeddingService()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        out.extend(await svc.embed(texts[i : i + batch]))
        if (i // batch) % 20 == 0:
            print(f"    embedded {min(i + batch, len(texts))}/{len(texts)}",
                  file=sys.stderr)
    return out


def score_arm(chunks: list[dict], questions: list[dict], use_dense: bool,
              scope: str = "corpus") -> list[dict]:
    """Rank every query against one arm's chunk set. Returns per-question ranks."""
    import numpy as np

    docs_tokens = [tokenize(c["content"]) for c in chunks]
    by_doc: dict[str, list[int]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        by_doc[chunk["doc"]].append(idx)

    dense = None
    if use_dense:
        vectors = asyncio.run(_embed_all([c["content"] for c in chunks]))
        dense = np.asarray(vectors, dtype="float32")
        dense /= np.linalg.norm(dense, axis=1, keepdims=True) + 1e-9
        q_vecs = asyncio.run(_embed_all([q["query"] for q in questions]))
        q_dense = np.asarray(q_vecs, dtype="float32")
        q_dense /= np.linalg.norm(q_dense, axis=1, keepdims=True) + 1e-9

    results = []
    for q_idx, question in enumerate(questions):
        # Corpus-wide by default: production retrieval does not know which
        # document holds the answer, and restricting to one document would hide
        # exactly the competition arm B's extra blocks create.
        candidates = (
            by_doc.get(question["doc"], []) if scope == "doc" else list(range(len(chunks)))
        )
        gold_ids = [
            candidates[i]
            for i in gold_chunk_ids(
                [chunks[i] for i in candidates], gold_cells=question["gold_cells"]
            )
        ]

        q_tokens = tokenize(question["query"])
        lex = bm25_scores([docs_tokens[i] for i in candidates], q_tokens)
        lex_rank_ids = [
            candidates[i]
            for i in sorted(range(len(candidates)), key=lambda i: -lex[i])
        ]

        record = {
            "doc": question["doc"],
            "page": question["page"],
            "table": question["table"],
            "query": question["query"],
            "gold_chunks": len(gold_ids),
            "bm25_rank": rank_of(lex_rank_ids, gold_ids),
        }

        if use_dense:
            sims = dense[candidates] @ q_dense[q_idx]
            dense_rank_ids = [
                candidates[i]
                for i in sorted(range(len(candidates)), key=lambda i: -sims[i])
            ]
            record["dense_rank"] = rank_of(dense_rank_ids, gold_ids)
            record["rrf_rank"] = rank_of(
                rrf_fuse([lex_rank_ids, dense_rank_ids]), gold_ids
            )
        results.append(record)
    return results


def compare(arm_a: list[dict], arm_b: list[dict], retrievers) -> dict:
    """Per-retriever paired comparison on reciprocal rank."""
    from scipy.stats import wilcoxon

    out: dict = {"retrievers": {}, "questions": len(arm_a)}
    for name in retrievers:
        key = f"{name}_rank"
        pairs, per_doc = [], defaultdict(list)
        for a, b in zip(arm_a, arm_b):
            if key not in a or key not in b:
                continue
            ra, rb = reciprocal_rank(a[key]), reciprocal_rank(b[key])
            pairs.append({"a": a, "b": b, "ra": ra, "rb": rb})
            per_doc[a["doc"]].append(rb - ra)

        deltas = [p["rb"] - p["ra"] for p in pairs]
        wins = sum(1 for d in deltas if d > 1e-9)
        losses = sum(1 for d in deltas if d < -1e-9)
        ties = len(deltas) - wins - losses

        # Wilcoxon drops zero differences by definition; with none left there is
        # nothing to test and the honest answer is "no p-value", not "p = 1".
        p_value = None
        nonzero = [d for d in deltas if abs(d) > 1e-9]
        if nonzero:
            try:
                p_value = float(wilcoxon(nonzero).pvalue)
            except ValueError:
                p_value = None

        def _recall(arm_key: str, k: int = 10) -> float | None:
            if not pairs:
                return None
            hit = sum(
                1 for p in pairs
                if p[arm_key][key] is not None and p[arm_key][key] <= k
            )
            return hit / len(pairs)

        lo, hi = bootstrap_doc_clustered_ci(per_doc)
        out["retrievers"][name] = {
            "mrr_a": statistics.fmean([p["ra"] for p in pairs]) if pairs else None,
            "mrr_b": statistics.fmean([p["rb"] for p in pairs]) if pairs else None,
            "mean_delta": statistics.fmean(deltas) if deltas else None,
            "ci95_doc_clustered": [lo, hi],
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "wilcoxon_p": p_value,
            "recall_at_10_a": _recall("a"),
            "recall_at_10_b": _recall("b"),
            "not_retrieved_a": sum(1 for p in pairs if p["a"][key] is None),
            "not_retrieved_b": sum(1 for p in pairs if p["b"][key] is None),
            "docs": len(per_doc),
        }
    return out


def _fmt(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def format_report(summary: dict, skipped: int, generic: int = 0,
                  scope: str = "corpus") -> str:
    lines = [
        "# Table Probe — 0b Retrieval-Only Rank Probe",
        "",
        "**Question:** does arm B make the gold row easier to *retrieve*? If not, the",
        "answer model never sees the row and no end-to-end gain is possible — a NO-GO",
        "bought for a day of compute instead of the weeks the full probe costs.",
        "",
        "**Queries are mechanical** (`\"<row label> <column header>\"` straight out of",
        "`extract()`), so nothing is tuned toward either arm. They are therefore not",
        "user questions: this step can kill the probe, never green-light a build.",
        "",
        f"Questions scored: **{summary['questions']}** — ranked against the "
        f"{'whole corpus' if scope == 'corpus' else 'source document only'}.",
        f"Dropped: {skipped} tables with no askable column, {generic} rows too",
        "generic to identify by their own words.",
        "",
        "| retriever | MRR arm A | MRR arm B | mean ΔRR | 95% CI (doc-clustered) | recall@10 A→B | W/L/T | Wilcoxon p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in summary["retrievers"].items():
        ci = r["ci95_doc_clustered"]
        ci_txt = "—" if ci[0] is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        lines.append(
            f"| {name} | {_fmt(r['mrr_a'])} | {_fmt(r['mrr_b'])} | "
            f"{_fmt(r['mean_delta'])} | {ci_txt} | "
            f"{_fmt(r['recall_at_10_a'], 2)} → {_fmt(r['recall_at_10_b'], 2)} | "
            f"{r['wins']}/{r['losses']}/{r['ties']} | {_fmt(r['wilcoxon_p'])} |"
        )

    lines += [
        "",
        "## How to read this",
        "",
        "- **ΔRR > 0 means arm B retrieved the gold row higher.** The outcome is",
        "  continuous, so it carries magnitude the binary sign test throws away.",
        "- **The interval resamples documents, not tables.** 52% of the pool sits in",
        "  one file; per-table resampling would report precision that is not there.",
        "- **A ceiling bounds the lexical column.** The query is built from the row's",
        "  own words, and BM25 is bag-of-words — so a row shredded across lines still",
        "  matches every term. Arm A therefore starts near the top of the scale and",
        "  there is almost nothing left for arm B to add. Row adjacency, which 0a",
        "  showed collapsing to 17%, is simply not what lexical retrieval consumes.",
        "  **Dense is where the headroom is**, and dense is where to read the answer.",
        "- **BM25 here is a simplified lexical baseline** (whitespace + lowercase), not",
        "  Elasticsearch's Vietnamese analyzer. Identical across arms, so the delta is",
        "  fair; the absolute MRR is not production retrieval.",
        "- The gold chunk is bag-of-words overlap with the row, never a contiguous",
        "  substring — arm A shreds cells, and a substring gold would have defined arm",
        "  A out of the comparison instead of measuring it. No page filter either: the",
        "  chunker labels a chunk by the markers inside it, so a chunk holding one",
        "  page's tail and the next page's marker is labelled with the next page. Arm",
        "  B's longer pages shift those boundaries more often, so filtering on page",
        "  deleted arm B's gold chunks — measured, not hypothesised.",
        "- Vision OCR fallback is forced off in both arms: it calls a remote model, so",
        "  it is neither deterministic nor available offline.",
        "",
        "## Verdict",
        "",
    ]

    improved = [
        name
        for name, r in summary["retrievers"].items()
        if r["mean_delta"] is not None
        and r["mean_delta"] > 0
        and (r["ci95_doc_clustered"][0] or 0) > 0
    ]
    best_recall_a = max(
        (r["recall_at_10_a"] or 0) for r in summary["retrievers"].values()
    ) if summary["retrievers"] else 0.0

    if improved:
        lines += [
            f"Arm B raised the gold row for: **{', '.join(improved)}** "
            "(interval excludes zero).",
            "The retrieval mechanism exists. Continue to Road 1 — with the MDE and the",
            "n_eff≈7 clustering bound stated on the same page, because this step",
            "repairs neither.",
        ]
    elif best_recall_a >= 0.9:
        lines += [
            "**No retriever shows arm B raising the gold row** — every interval spans",
            "zero.",
            "",
            "But the kill condition this step was written with does **not** fire, and",
            "saying it did would be wrong. That condition was *\"arm B does not improve",
            "retrieval, therefore the answer model never sees the row\"*. Its premise is",
            f"false here: arm A already retrieves the gold row at recall@10 = "
            f"{best_recall_a:.2f}. The model sees the row under **both** arms.",
            "",
            "What this step actually settles:",
            "",
            "- **Retrieval is not where flattening hurts.** 0a measured row adjacency",
            "  collapsing to 17%, and that damage does not propagate here, because",
            "  lexical retrieval is bag-of-words: a row shredded across lines still",
            "  matches every one of its own terms.",
            "- **Arm B has no retrieval benefit to offer.** Not a wash to be re-run",
            "  bigger — there is no headroom on the lexical side and no movement on the",
            "  dense side.",
            "",
            "What remains untested is **comprehension**: once the row is in context, can",
            "the answer model still bind a cell to its column when the text is shredded?",
            "0b cannot answer that, and must not be read as if it had. Any remaining",
            "gain for arm B has to come from comprehension, not recall.",
        ]
    else:
        lines += [
            "**No retriever shows arm B raising the gold row**, and arm A's recall is",
            f"weak ({best_recall_a:.2f} at 10). The gold row is often not retrieved at",
            "all, so this run measured a retrieval stack that is failing for reasons",
            "beyond tables. Fix that before reading anything into the arms.",
        ]

    lines += [
        "",
        "## What this step cannot say",
        "",
        "- Nothing about answer correctness. The queries are the row's own words, not",
        "  user questions.",
        "- Nothing about the 25% of pages with no text layer: `find_tables()` is blind",
        "  there and arm B is byte-identical to arm A.",
        "- Nothing that repairs the clustering bound. 52% of the pool is one document;",
        "  n_eff stays far below the table count no matter how continuous the outcome.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/originals")
    ap.add_argument("--survey", default="data/eval/table_probe_corpus_survey.json")
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--report", dest="report_out")
    ap.add_argument("--no-dense", action="store_true",
                    help="skip TEI embeddings; BM25 only")
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--scope", choices=("corpus", "doc"), default="corpus",
                    help="rank against the whole corpus (default) or one document")
    args = ap.parse_args()

    survey = json.loads(Path(args.survey).read_text(encoding="utf-8"))
    docs = survey["unique_documents_list"]
    if args.limit_docs:
        docs = docs[: args.limit_docs]

    print(f"[0b] questions from {len(docs)} unique documents", file=sys.stderr)
    questions = collect_questions(args.corpus, docs)
    skipped = sum(1 for q in questions if q.get("skipped"))
    questions = [q for q in questions if not q.get("skipped")]
    generic = [q for q in questions if not row_is_identifiable(q["gold_cells"])]
    questions = [q for q in questions if row_is_identifiable(q["gold_cells"])]
    if generic:
        print(f"[0b] {len(generic)} rows too generic to identify — dropped",
              file=sys.stderr)
    print(f"[0b] {len(questions)} questions, {skipped} skipped", file=sys.stderr)

    print("[0b] building arm A (flag off)", file=sys.stderr)
    chunks_a = build_arm(args.corpus, docs, preserve_tables=False)
    print(f"[0b] arm A: {len(chunks_a)} chunks", file=sys.stderr)
    print("[0b] building arm B (flag on)", file=sys.stderr)
    chunks_b = build_arm(args.corpus, docs, preserve_tables=True)
    print(f"[0b] arm B: {len(chunks_b)} chunks", file=sys.stderr)

    use_dense = not args.no_dense
    retrievers = ["bm25"] + (["dense", "rrf"] if use_dense else [])

    print("[0b] scoring arm A", file=sys.stderr)
    ranks_a = score_arm(chunks_a, questions, use_dense, scope=args.scope)
    print("[0b] scoring arm B", file=sys.stderr)
    ranks_b = score_arm(chunks_b, questions, use_dense, scope=args.scope)

    summary = compare(ranks_a, ranks_b, retrievers)
    report = format_report(summary, skipped, len(generic), args.scope)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(
                {"summary": summary, "arm_a": ranks_a, "arm_b": ranks_b,
                 "questions": questions},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
