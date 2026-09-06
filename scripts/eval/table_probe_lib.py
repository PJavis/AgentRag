"""Pure, I/O-free helpers for the table-data probe. Tested in isolation.

PROBE-SCOPED (2026-08-02) — delete with `PDF_PRESERVE_TABLES`.

The decision rule here is deliberately *not* "mean(B) - mean(A) >= 0.10". At the
n this probe can reach (tens of questions at most), a 0.10 mean delta is one or
two questions flipping, and measured A/B noise on this stack is around 0.3. A
mean-delta gate would decide a build on a coin flip.

Instead the same question is scored under both arms and compared per question:
a paired design, reported as wins / losses / ties plus an exact two-sided sign
test on the discordant pairs. Pairing removes per-question difficulty — the
dominant variance term — which is what makes a small n usable at all.

`INCONCLUSIVE` is a first-class outcome. At small n it is usually the honest
one, and the remedy is more questions, not a lower bar.
"""

from __future__ import annotations

from math import comb

_REQUIRED_ROW_FIELDS = ("id", "question", "reference_answer", "gold_contexts")
#: Gold must key on (doc, page): the two arms produce different chunk *text* for
#: the same table, so text matching would compare different objects across arms.
_REQUIRED_PROVENANCE_FIELDS = ("source_doc", "source_page")

#: Scores closer than this count as a tie rather than a win.
TIE_EPSILON = 1e-9
#: GO needs B to win at least this many times per loss.
MIN_WIN_RATIO = 2.0
#: GO needs the sign test below this p-value.
ALPHA = 0.05


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


def arm_index_name(base: str, arm: str) -> str:
    """A per-arm label for reports and output filenames.

    NOT an index the runner can point retrieval at: `get_agent_service()` takes no
    arguments and `ElasticsearchStore` reads `settings.ELASTICSEARCH_INDEX_NAME`
    at construction, so there is no supported way to target a per-arm index. Arms
    are isolated by SEQUENCE instead — wipe, ingest one arm, score it, repeat.
    """
    if arm not in ("a", "b"):
        raise ValueError(f"arm must be 'a' or 'b', got {arm!r}")
    return f"{base}_arm_{arm}"


def rank_targets(
    survey: dict,
    top_n: int | None = None,
    max_tokens: int | None = None,
) -> list[dict]:
    """Target *tables* (not documents) from a corpus-survey report.

    Two things this must not do, both measured on the real survey:

    1. **Concentrate on one document.** A flat `(-cells, doc, page)` sort is
       document-blind: it returned 8 of the top 10 from a single file (80%),
       worse than the 47% doc-level share the table-level unit was introduced to
       avoid. Wins and losses then are not independent draws over the corpus,
       while the sign test's p-value assumes they are. So documents are
       interleaved round-robin, each contributing its densest grid first.

    2. **Rank by a size proxy.** Cell count predicts rendered size badly, so
       `max_tokens` filters on a measured figure from the survey instead.
       The measured figure that matters is `max_block_tokens`, the largest
       rendered BLOCK — not `est_tokens`, the whole-table total. `render_markdown`
       emits a large table as several under-budget blocks and the chunker keeps
       each one whole, so filtering on the total discards precisely the large
       multi-row grids the probe exists to measure. Grids from a survey that
       recorded neither figure are kept.
    """
    by_doc: dict[str, list[dict]] = {}
    for doc in survey.get("per_doc", []):
        for grid in doc.get("data_grids", []):
            target = {
                "doc": doc["doc"],
                "page": grid["page"],
                "rows": grid["rows"],
                "cols": grid["cols"],
                "cells": grid["rows"] * grid["cols"],
            }
            for key in ("est_tokens", "max_block_tokens"):
                if key in grid:
                    target[key] = grid[key]
            # Fall back to est_tokens only when the survey predates
            # max_block_tokens; a table whose largest block fits is eligible.
            measured = target.get("max_block_tokens", target.get("est_tokens"))
            if max_tokens is not None and measured is not None and measured > max_tokens:
                continue
            by_doc.setdefault(doc["doc"], []).append(target)

    for grids in by_doc.values():
        grids.sort(key=lambda t: (-t["cells"], t["page"]))

    # Round-robin: densest grid of each doc, then each doc's second, and so on.
    # Docs are ordered by their densest grid so the overall top target is still
    # the corpus's densest one.
    order = sorted(by_doc, key=lambda d: (-by_doc[d][0]["cells"], d))
    targets: list[dict] = []
    for rank in range(max(len(g) for g in by_doc.values()) if by_doc else 0):
        for doc in order:
            if rank < len(by_doc[doc]):
                targets.append(by_doc[doc][rank])
    return targets[:top_n] if top_n is not None else targets


def validate_probe_row(row: dict) -> list[str]:
    """Return a list of schema errors; empty list means the row is valid."""
    errs: list[str] = []
    for field in _REQUIRED_ROW_FIELDS:
        if field not in row or row[field] in (None, "", []):
            errs.append(f"missing/empty field: {field}")
    for field in _REQUIRED_PROVENANCE_FIELDS:
        if field not in row or row[field] in (None, ""):
            errs.append(f"missing/empty field: {field} (gold must key on doc+page)")
    gold = row.get("gold_contexts")
    if gold is not None and not isinstance(gold, list):
        errs.append("gold_contexts must be a list")
    page = row.get("source_page")
    if page is not None and page != "":
        # `isinstance(True, int)` is True, and page 0 / -3 are not pages either.
        if isinstance(page, bool) or not isinstance(page, int):
            errs.append("source_page must be an int")
        elif page < 1:
            errs.append("source_page must be >= 1 (pages are 1-indexed)")
    return errs


def corpus_matches(row_sha: str | None, survey_sha: str | None) -> tuple[bool, str]:
    """Is this evalset still valid against the corpus in front of us?

    Deliberately NOT `eval/corpus_fingerprint.py`: that one hashes
    (document_title, segment_count), and arm B changes segment counts by design,
    so it would report a mismatch between the two arms of a correct run. This
    compares the survey's `corpus_docs_sha` — a hash of the deduplicated document
    content — which is identical under both arms.

    Unstamped rows are allowed through with a warning rather than blocked: the
    probe's gold keys on `(source_doc, source_page)`, which survives a re-ingest,
    so a missing stamp is a weaker signal here than it is for id-keyed eval sets.
    """
    if not row_sha:
        return True, "evalset carries no corpus_docs_sha — cannot verify the corpus"
    if not survey_sha:
        return True, "no survey fingerprint to compare against"
    if row_sha != survey_sha:
        return False, (
            f"evalset was authored against corpus {row_sha}, survey reports "
            f"{survey_sha} — documents were added or removed; (doc, page) gold "
            "may point at the wrong page"
        )
    return True, f"corpus {row_sha} matches"


def paired_outcomes(
    a_scores: dict[str, float],
    b_scores: dict[str, float],
    eligible: set[str] | None = None,
    eps: float = TIE_EPSILON,
) -> dict:
    """Compare arms per question over questions both arms scored.

    `eligible` restricts the comparison to questions whose gold table passed the
    arm-B gate and survived chunking. Questions where the arms saw identical
    input cannot inform the decision and only add ties.

    Every question that does NOT reach the comparison is counted and named.
    Intersecting the two arms silently is survivorship bias running in the
    direction that favours GO: arm B emits longer table text, so arm B is the
    arm more likely to time out or trip the judge, and dropping exactly those
    questions from both numerator and denominator can turn a mixed result into
    a clean sweep. §4 also requires reporting "n eligible, n excluded", which
    is impossible if the drops are never recorded.
    """
    scored_both = set(a_scores) & set(b_scores)
    a_only = sorted(set(a_scores) - set(b_scores))
    b_only = sorted(set(b_scores) - set(a_scores))

    ids = sorted(scored_both)
    ineligible: list[str] = []
    if eligible is not None:
        ineligible = [qid for qid in ids if qid not in eligible]
        ids = [qid for qid in ids if qid in eligible]

    wins, losses, ties = [], [], []
    for qid in ids:
        delta = b_scores[qid] - a_scores[qid]
        if abs(delta) <= eps:
            ties.append(qid)
        elif delta > 0:
            wins.append(qid)
        else:
            losses.append(qid)
    return {
        "n_compared": len(ids),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "n_ties": len(ties),
        # Accounting — every question that did not reach the comparison.
        "scored_a_only": a_only,
        "scored_b_only": b_only,
        "n_missing": len(a_only) + len(b_only),
        "ineligible": ineligible,
        "n_ineligible": len(ineligible),
    }


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test on discordant pairs (p=0.5 under H0)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def decide_paired(
    outcomes: dict,
    alpha: float = ALPHA,
    min_win_ratio: float = MIN_WIN_RATIO,
) -> dict:
    """GO / NO-GO / INCONCLUSIVE from paired outcomes.

    GO needs both a practical margin (B wins >= min_win_ratio per loss) and a
    statistical one (sign test p < alpha). Meeting only one is INCONCLUSIVE —
    reported honestly, not rounded up to GO.
    """
    wins, losses, ties = outcomes["n_wins"], outcomes["n_losses"], outcomes["n_ties"]
    p = sign_test_p(wins, losses)

    if wins + losses + ties == 0:
        decision, why = "INCONCLUSIVE", "no eligible questions to compare"
    elif wins + losses == 0:
        decision, why = "NO-GO", f"all {ties} eligible questions tied — arms indistinguishable"
    elif wins <= losses:
        decision, why = "NO-GO", f"B did not win more than it lost ({wins}W/{losses}L)"
    elif wins < min_win_ratio * losses:
        decision, why = (
            "INCONCLUSIVE",
            f"B leads {wins}W/{losses}L but below the {min_win_ratio:g}:1 margin",
        )
    elif p >= alpha:
        decision, why = (
            "INCONCLUSIVE",
            f"B leads {wins}W/{losses}L but p={p:.3f} >= {alpha} — need more questions",
        )
    else:
        decision, why = "GO", f"B wins {wins}W/{losses}L, p={p:.3f} < {alpha}"

    return {
        "decision": decision,
        "reason": why,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "p_value": round(p, 4),
        "alpha": alpha,
    }


def mean_delta(
    a_scores: dict[str, float],
    b_scores: dict[str, float],
    eligible: set[str] | None = None,
) -> float | None:
    """Mean(B) - mean(A) over shared questions. Reported as colour only.

    Deliberately not part of `decide_paired`: at this n a mean delta is within
    noise and must not drive the decision. `eligible` must be passed whatever
    was passed to `paired_outcomes` — otherwise the number printed beside the
    decision averages a different population than the decision was made on.
    """
    ids = sorted(set(a_scores) & set(b_scores))
    if eligible is not None:
        ids = [qid for qid in ids if qid in eligible]
    if not ids:
        return None
    return round(
        sum(b_scores[i] for i in ids) / len(ids)
        - sum(a_scores[i] for i in ids) / len(ids),
        4,
    )
