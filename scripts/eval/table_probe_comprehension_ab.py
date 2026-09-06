"""0c — comprehension probe: can the model read a cell out of a shredded table?

0b removed retrieval as the mechanism. Arm A already retrieves the gold row at
recall@10 = 0.95, so the row reaches the model's context under BOTH arms, and
arm B has no retrieval gain to offer. That leaves exactly one live hypothesis
for why tables might matter:

    once the row IS in context, can the answer model still bind a cell to its
    column when the flattened text has shredded the row?

0c tests that, and it is cheap enough to run before committing to Road 1: the
ground truth is `extract()` itself, so there is **no judge, no hand-authored
reference answer, and no gold-context curation** — 2 LLM calls per table.

Honest bounds, unchanged by this step:

- The question is mechanical (`"what is <column> for <row>"`), not a user
  question. This measures cell lookup, not answering.
- The unit is the table and 52% of the pool is one document, so n_eff stays far
  below the table count. Reported as an estimate with a document-clustered
  interval; `decide_paired` is printed but is NOT confirmatory here.
- Arm B's context is a superset of arm A's (flat text plus the markdown). That
  is exactly what arm B does in production, so it is the right comparison —
  but it means a win can come from the second copy as easily as from the
  structure, and the report says so.

Usage:
    PYTHONPATH=. uv run python scripts/eval/table_probe_comprehension_ab.py \
        --corpus data/originals \
        --survey data/eval/table_probe_corpus_survey.json \
        --json data/eval/table_probe_comprehension_ab.json \
        --report docs/eval/table_probe_comprehension_ab_<date>.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval.table_probe_lib import decide_paired  # noqa: E402
from scripts.eval.table_probe_retrieval_ab import (  # noqa: E402
    bootstrap_doc_clustered_ci,
    collect_questions,
    row_is_identifiable,
    tokenize,
)

#: Share of a long cell's tokens an answer must cover before it counts as a hit.
MIN_LONG_CELL_COVERAGE = 0.6
#: A cell this short is judged by containment; longer ones need coverage, so a
#: single shared word cannot claim a paragraph-length cell.
SHORT_CELL_TOKENS = 4
#: The model is told to emit this when the text does not carry the answer.
ABSTAIN = "UNKNOWN"

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_answer(text) -> str:
    if text is None:
        return ""
    return _WS.sub(" ", _PUNCT.sub(" ", str(text).lower())).strip()


def token_f1(answer, truth) -> float:
    """Continuous overlap, so a near-miss is not scored the same as a wrong cell."""
    a, t = tokenize(normalize_answer(answer)), tokenize(normalize_answer(truth))
    if not a or not t:
        return 0.0
    common = 0
    pool = list(t)
    for tok in a:
        if tok in pool:
            pool.remove(tok)
            common += 1
    if not common:
        return 0.0
    precision, recall = common / len(a), common / len(t)
    return 2 * precision * recall / (precision + recall)


def score_answer(answer, truth) -> float:
    """1.0 when the answer carries the cell, else 0.0.

    Short cells are judged by containment — a model that says "the dose is
    500 mg" has read the cell. Long cells need most of their tokens, so a single
    shared word cannot claim a paragraph-length cell. An abstention never scores,
    even when the cell text itself reads like one.
    """
    a, t = normalize_answer(answer), normalize_answer(truth)
    if not a or not t:
        return 0.0
    if a == normalize_answer(ABSTAIN):
        return 0.0
    truth_tokens = tokenize(t)
    if len(truth_tokens) <= SHORT_CELL_TOKENS:
        return 1.0 if (t in a or a == t) else 0.0
    answer_tokens = set(tokenize(a))
    covered = sum(1 for tok in set(truth_tokens) if tok in answer_tokens)
    return 1.0 if covered / len(set(truth_tokens)) >= MIN_LONG_CELL_COVERAGE else 0.0


def cell_bleed(answer, gold_cell, all_cells) -> tuple[int, int]:
    """(tokens borrowed from OTHER cells of the table, tokens beyond the gold cell).

    This is the mechanism test. A model reading a shredded table does not usually
    fail to find the value — it fails to see where the value STOPS, and merges the
    neighbouring cell into its answer. That answer still contains the right cell,
    so a containment-style correctness check scores it as a hit while it is, in an
    answer to a real question, wrong: it attributes another row's content to this
    row. Counting where the surplus tokens came from separates "verbose" from
    "could not find the cell boundary".
    """
    gold = set(tokenize(normalize_answer(gold_cell)))
    others = {
        tok
        for row in all_cells
        for cell in row
        for tok in tokenize(normalize_answer(cell or ""))
    } - gold
    extra = set(tokenize(normalize_answer(answer))) - gold
    return len(extra & others), len(extra)


def build_prompt(context: str, row_label: str, column: str) -> dict:
    return {
        "system": (
            "You read values out of a document. Answer ONLY from the text given. "
            "Reply with the cell value alone — no explanation, no restating the "
            f"question. If the text does not contain the value, reply exactly "
            f"{ABSTAIN}."
        ),
        "user": (
            f"DOCUMENT TEXT:\n{context}\n\n"
            f'QUESTION: In the table, what value sits in column "{column}" for the '
            f'row "{row_label}"?'
        ),
    }


def pair_outcomes(arm_a: list[dict], arm_b: list[dict]) -> dict:
    """Paired win/loss/tie counts. A call that errored in either arm is excluded.

    An API failure measures uptime, not tables. Scoring it as a loss for whichever
    arm happened to fail would put the stack's flakiness into the result.
    """
    wins = losses = ties = excluded = 0
    for a, b in zip(arm_a, arm_b):
        if a.get("error") or b.get("error"):
            excluded += 1
            continue
        delta = b.get("correct", 0.0) - a.get("correct", 0.0)
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            ties += 1
    return {
        "n_wins": wins,
        "n_losses": losses,
        "n_ties": ties,
        "n_excluded": excluded,
    }


# ---------------------------------------------------------------------------
# Corpus wiring
# ---------------------------------------------------------------------------


def page_texts(corpus_dir: str, docs: list[str], preserve_tables: bool) -> dict:
    """{(doc, page): page text} exactly as the pipeline would index it."""
    from src.agentrag.config import settings
    from src.agentrag.ingestion.parsers.pdf_parser import PDFParser

    prev_flag = settings.PDF_PRESERVE_TABLES
    prev_vision = settings.PDF_OCR_VISION_FALLBACK
    settings.PDF_PRESERVE_TABLES = preserve_tables
    settings.PDF_OCR_VISION_FALLBACK = False
    out: dict[tuple[str, int], str] = {}
    try:
        parser = PDFParser()
        for rel in docs:
            path = Path(corpus_dir) / rel
            if not path.exists():
                continue
            try:
                parsed = parser.parse(str(path))
            except Exception as exc:  # noqa: BLE001 — unreadable PDF
                print(f"  ! parse failed {rel}: {exc!r}", file=sys.stderr)
                continue
            for page in parsed["page_data"]:
                out[(rel, page["page_num"])] = page["text"]
    finally:
        settings.PDF_PRESERVE_TABLES = prev_flag
        settings.PDF_OCR_VISION_FALLBACK = prev_vision
    return out


async def _ask(gateway, context: str, question: dict) -> dict:
    prompt = build_prompt(context, question["row_label"], question["column"])
    try:
        answer = await gateway.text_response(
            prompt["system"], prompt["user"], task="answer"
        )
    except Exception as exc:  # noqa: BLE001 — provider error, excluded not scored
        return {"error": repr(exc)[:200]}
    return {
        "answer": answer,
        "correct": score_answer(answer, question["answer_cell"]),
        "f1": token_f1(answer, question["answer_cell"]),
        "abstained": normalize_answer(answer) == normalize_answer(ABSTAIN),
    }


def aggregate_samples(samples: list[dict]) -> dict:
    """Collapse repeated calls for one question into one record.

    `correct` becomes the share of samples that read the cell, so a question the
    model gets right only sometimes contributes a fraction rather than a coin
    flip. `unstable` flags disagreement, which is what makes a single-run number
    untrustworthy at this n.
    """
    errors = [s for s in samples if s.get("error")]
    good = [s for s in samples if not s.get("error")]
    if not good:
        return {"error": errors[0]["error"] if errors else "no samples"}
    corrects = [s["correct"] for s in good]
    return {
        "answer": good[0]["answer"],
        "answers": [s["answer"] for s in good],
        "correct": statistics.fmean(corrects),
        "f1": statistics.fmean([s["f1"] for s in good]),
        "abstained": all(s.get("abstained") for s in good),
        "unstable": len(set(corrects)) > 1,
        "samples": len(good),
    }


def run_arm(questions: list[dict], texts: dict, label: str,
            samples: int = 1) -> list[dict]:
    """Ask every question `samples` times under one arm, at temperature 0.

    The default AGENT_TEMPERATURE of 0.3 makes this probe irreproducible: two
    runs of the same code disagreed 15/19 vs 13/19, which is the size of the
    effect being measured. Temperature is pinned to 0 for both arms and the
    remaining disagreement is counted rather than averaged away silently.
    """
    from src.agentrag.config import settings
    from src.agentrag.services.llm_gateway import LLMGateway

    previous_temp = settings.AGENT_TEMPERATURE
    settings.AGENT_TEMPERATURE = 0.0
    try:
        gateway = LLMGateway()   # constructed AFTER the setting: clients cache it

        async def go():
            out = []
            for idx, question in enumerate(questions, start=1):
                context = texts.get((question["doc"], question["page"]), "")
                if not context:
                    out.append({"error": "no page text"})
                    continue
                got = [await _ask(gateway, context, question) for _ in range(samples)]
                out.append(aggregate_samples(got))
                if idx % 5 == 0:
                    print(f"    {label}: {idx}/{len(questions)}", file=sys.stderr)
            return out

        return asyncio.run(go())
    finally:
        settings.AGENT_TEMPERATURE = previous_temp


def duplicated_context(flat_text: str) -> str:
    """Arm C: the same page text twice — duplication with no added structure.

    Arm B's context is a superset of arm A's: the flat text PLUS the markdown. A
    win could therefore come from the model seeing the content a second time
    rather than from the row/column structure. Arm C holds duplication constant
    and adds no structure, so B − C isolates the structure.
    """
    return f"{flat_text}\n\n{flat_text}"


def table_cells(corpus_dir: str, question: dict):
    """Re-extract one table's cells, for the bleed metric. [] on any failure."""
    import fitz

    path = Path(corpus_dir) / question["doc"]
    if not path.exists():
        return []
    try:
        doc = fitz.open(str(path))
    except Exception:  # noqa: BLE001 — unreadable PDF
        return []
    with doc:
        try:
            page = doc[question["page"] - 1]
            return list(page.find_tables().tables)[question["table"]].extract()
        except Exception:  # noqa: BLE001 — table no longer resolvable
            return []


def _fmt(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def format_report(summary: dict) -> str:
    o, ci = summary["outcomes"], summary["ci95_doc_clustered"]
    ci_txt = "—" if ci[0] is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
    lines = [
        "# Table Probe — 0c Comprehension Probe",
        "",
        "**Question:** 0b showed the gold row reaches the model's context under both",
        "arms (arm A recall@10 = 0.95), so retrieval is not where flattening hurts.",
        "This asks the only hypothesis left: once the row is in context, can the model",
        "still bind a cell to its column when the flat text has shredded the row?",
        "",
        "**No judge.** Ground truth is `extract()` itself, so this costs two LLM calls",
        "per table — no reference answers, no gold-context curation, no LLM grader.",
        "",
        "## Result",
        "",
        "| quantity | arm A (flat) | arm B (+ markdown) |",
        "|---|---|---|",
        f"| cells read correctly | {summary['correct_a']:.1f}/{summary['scored']} "
        f"({_fmt(summary['acc_a'], 2)}) | {summary['correct_b']:.1f}/{summary['scored']} "
        f"({_fmt(summary['acc_b'], 2)}) |",
        f"| questions whose {summary['samples_per_question']} samples disagreed | "
        f"{summary['unstable_a']} | {summary['unstable_b']} |",
        f"| mean token-F1 | {_fmt(summary['f1_a'])} | {_fmt(summary['f1_b'])} |",
        f"| abstained (`{ABSTAIN}`) | {summary['abstain_a']} | {summary['abstain_b']} |",
        f"| surplus tokens beyond the gold cell | {summary['extra_a']} | "
        f"{summary['extra_b']} |",
        f"| — of those, borrowed from OTHER cells of the same table | "
        f"{summary['bleed_a']} ({_fmt(summary['bleed_frac_a'], 2)}) | "
        f"{summary['bleed_b']} ({_fmt(summary['bleed_frac_b'], 2)}) |",
        "",
        "| paired outcome | value |",
        "|---|---|",
        f"| B better / worse / same | {o['n_wins']} / {o['n_losses']} / {o['n_ties']} |",
        f"| excluded (call errored in one arm) | {o['n_excluded']} |",
        f"| mean ΔF1 (B − A) | {_fmt(summary['mean_delta_f1'])} |",
        f"| 95% CI, resampling documents | {ci_txt} |",
        f"| sign test p | {_fmt(summary['decision']['p_value'])} |",
        f"| shipped rule says | **{summary['decision']['decision']}** — "
        f"{summary['decision']['reason']} |",
        "",
        *(
            [
                "## Duplication control (arm C)",
                "",
                "Arm C is the page text **twice**: the same duplication arm B gets, with",
                "no structure added. B − C is therefore the structure, separated from the",
                "second copy.",
                "",
                "| quantity | arm C (duplicated flat) |",
                "|---|---|",
                f"| cells read correctly | {_fmt(summary['control']['acc_c'], 2)} |",
                f"| mean token-F1 | {_fmt(summary['control']['f1_c'])} |",
                f"| abstained | {summary['control']['abstain_c']} |",
                f"| B vs C — better/worse/same | "
                f"{summary['control']['outcomes_b_vs_c']['n_wins']} / "
                f"{summary['control']['outcomes_b_vs_c']['n_losses']} / "
                f"{summary['control']['outcomes_b_vs_c']['n_ties']} |",
                f"| B vs C — mean ΔF1 | "
                f"{_fmt(summary['control']['mean_delta_f1_b_vs_c'])} |",
                "",
            ]
            if summary.get("control")
            else []
        ),
        "## The mechanism",
        "",
        "The binary row barely moves (one table). The interesting number is the last",
        "one. Under arm A the model usually *finds* the value and then cannot see",
        "where it **stops**: its surplus tokens come overwhelmingly from other cells",
        "of the same table. Under arm B that surplus is essentially gone.",
        "",
        "This is why the binary column understates arm A's failures. A containment",
        "check scores \"right cell plus the next row glued on\" as correct, and in an",
        "answer to a real question it is not — it attributes one row's content to",
        "another. The token-F1 column and the bleed row are measuring that; the",
        "correctness column is blind to it.",
        "",
        "## How to read this",
        "",
        "- **The shipped verdict above is NOT confirmatory.** The unit is the table,",
        "  52% of the pool is one document, and the rule cannot return GO below 6",
        "  discordant tables. It is printed because the plan says to print it, and",
        "  labelled because the power analysis says it cannot carry a decision.",
        "- **The interval resamples documents, not tables** — the only interval this",
        "  corpus can honestly support.",
        "- **Arm B's context is a superset of arm A's**: the flat text plus the",
        "  markdown. That is exactly what arm B does in production, so it is the right",
        "  comparison — but a win can come from the second copy as easily as from the",
        "  structure, and nothing here separates those two.",
        f"- **Temperature is pinned to 0 and every question is asked "
        f"{summary['samples_per_question']} times.** At the default 0.3 two runs of",
        "  this same code disagreed 15/19 vs 13/19 — variance the size of the effect.",
        "  Residual sample disagreement is counted in the table above, not averaged",
        "  away silently.",
        "- The question is mechanical cell lookup, not a user question. A model that",
        "  can look a cell up may still answer a real question badly, and vice versa.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/originals")
    ap.add_argument("--survey", default="data/eval/table_probe_corpus_survey.json")
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--report", dest="report_out")
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--control", action="store_true",
                    help="also run arm C (page text duplicated) to separate "
                         "structure from mere duplication")
    ap.add_argument("--samples", type=int, default=3,
                    help="calls per question per arm; disagreement is reported")
    args = ap.parse_args()

    survey = json.loads(Path(args.survey).read_text(encoding="utf-8"))
    docs = survey["unique_documents_list"]
    if args.limit_docs:
        docs = docs[: args.limit_docs]

    questions = [
        q for q in collect_questions(args.corpus, docs)
        if not q.get("skipped") and row_is_identifiable(q["gold_cells"])
    ]
    print(f"[0c] {len(questions)} questions", file=sys.stderr)

    print("[0c] page text, arm A", file=sys.stderr)
    texts_a = page_texts(args.corpus, docs, preserve_tables=False)
    print("[0c] page text, arm B", file=sys.stderr)
    texts_b = page_texts(args.corpus, docs, preserve_tables=True)

    print("[0c] asking arm A", file=sys.stderr)
    arm_a = run_arm(questions, texts_a, "A", samples=args.samples)
    print("[0c] asking arm B", file=sys.stderr)
    arm_b = run_arm(questions, texts_b, "B", samples=args.samples)

    arm_c = None
    if args.control:
        print("[0c] asking arm C (duplication control)", file=sys.stderr)
        texts_c = {k: duplicated_context(v) for k, v in texts_a.items()}
        arm_c = run_arm(questions, texts_c, "C", samples=args.samples)

    outcomes = pair_outcomes(arm_a, arm_b)
    paired = [
        (q, a, b)
        for q, a, b in zip(questions, arm_a, arm_b)
        if not a.get("error") and not b.get("error")
    ]
    per_doc = defaultdict(list)
    bleed = {"a": [0, 0], "b": [0, 0]}
    for q, a, b in paired:
        per_doc[q["doc"]].append(b["f1"] - a["f1"])
        cells = table_cells(args.corpus, q)
        if not cells:
            continue
        for key, res in (("a", a), ("b", b)):
            borrowed, extra = cell_bleed(
                res.get("answer", ""), q["answer_cell"], cells
            )
            bleed[key][0] += borrowed
            bleed[key][1] += extra

    summary = {
        "scored": len(paired),
        "samples_per_question": args.samples,
        "unstable_a": sum(1 for _, a, _ in paired if a.get("unstable")),
        "unstable_b": sum(1 for _, _, b in paired if b.get("unstable")),
        "correct_a": sum(a["correct"] for _, a, _ in paired),
        "correct_b": sum(b["correct"] for _, _, b in paired),
        "acc_a": statistics.fmean([a["correct"] for _, a, _ in paired]) if paired else None,
        "acc_b": statistics.fmean([b["correct"] for _, _, b in paired]) if paired else None,
        "f1_a": statistics.fmean([a["f1"] for _, a, _ in paired]) if paired else None,
        "f1_b": statistics.fmean([b["f1"] for _, _, b in paired]) if paired else None,
        "abstain_a": sum(1 for _, a, _ in paired if a.get("abstained")),
        "abstain_b": sum(1 for _, _, b in paired if b.get("abstained")),
        "mean_delta_f1": statistics.fmean(
            [b["f1"] - a["f1"] for _, a, b in paired]
        ) if paired else None,
        "ci95_doc_clustered": list(bootstrap_doc_clustered_ci(per_doc)),
        "bleed_a": bleed["a"][0],
        "extra_a": bleed["a"][1],
        "bleed_frac_a": (bleed["a"][0] / bleed["a"][1]) if bleed["a"][1] else None,
        "bleed_b": bleed["b"][0],
        "extra_b": bleed["b"][1],
        "bleed_frac_b": (bleed["b"][0] / bleed["b"][1]) if bleed["b"][1] else None,
        "outcomes": outcomes,
        "decision": decide_paired(outcomes),
    }

    if arm_c is not None:
        paired_bc = [
            (q, c, b)
            for q, c, b in zip(questions, arm_c, arm_b)
            if not c.get("error") and not b.get("error")
        ]
        summary["control"] = {
            "acc_c": statistics.fmean([c["correct"] for _, c, _ in paired_bc])
            if paired_bc else None,
            "f1_c": statistics.fmean([c["f1"] for _, c, _ in paired_bc])
            if paired_bc else None,
            "abstain_c": sum(1 for _, c, _ in paired_bc if c.get("abstained")),
            "outcomes_b_vs_c": pair_outcomes(arm_c, arm_b),
            "mean_delta_f1_b_vs_c": statistics.fmean(
                [b["f1"] - c["f1"] for _, c, b in paired_bc]
            ) if paired_bc else None,
        }
    report = format_report(summary)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps({"summary": summary, "questions": questions,
                        "arm_a": arm_a, "arm_b": arm_b, "arm_c": arm_c},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
