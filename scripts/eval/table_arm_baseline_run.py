"""Drive the live stack with table-lookup questions to produce a rollout baseline.

`table_arm_b_monitor.py` reads production answers, but the pre-flip snapshot held
12 answers and **none of them cited a table-bearing page** — so the overrun metric
had never once fired, and the rollback triggers in
`docs/eval/table_arm_b_rollout_2026-09-06.md` §4 compared against nothing.

This asks the real `/chat` endpoint the same row lookups 0c used, phrased as a user
would, so the monitor has table-citing answers to score. Run it once before the flag
flip (arm A) and again after (arm B): same questions, same script, so the two
snapshots are comparable in the one way a trial rollout can be.

It writes real rows into `chat_messages`, so every run gets its own conversation
titled `arm-baseline-<tag>` — filterable by the monitor, and removable with
`DELETE /conversations/<id>` when no longer wanted.

Usage:
    PYTHONPATH=. uv run python scripts/eval/table_arm_baseline_run.py \
        --tag armA-2026-09-06 --json data/eval/arm_baseline_armA.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

from scripts.eval.table_probe_retrieval_ab import (  # noqa: E402
    collect_questions,
    row_is_identifiable,
)


def ask_text(row_label: str, column: str) -> str:
    """A user-shaped phrasing of the same lookup 0c asked mechanically."""
    return (
        f'Trong tài liệu, ở bảng có liên quan, giá trị tại cột "{column}" '
        f'của hàng "{row_label}" là gì?'
    )


class ChatError(RuntimeError):
    """The endpoint answered 200 with an error payload.

    The first version of this script counted transport failures only, so 19
    replies of `{"error": "question is required"}` were reported as "19
    answered, 0 failed". A 200 is not a success here.
    """


def validate_reply(reply: dict) -> None:
    """Raise unless the endpoint actually answered.

    `/chat` returns HTTP 200 with `{"error": ...}` for a rejected request, so
    transport success proves nothing. The first version of this script checked
    only for transport errors and reported 19 replies of
    `{"error": "question is required"}` as "19 answered, 0 failed".
    """
    if not isinstance(reply, dict):
        raise ChatError(f"expected a JSON object, got {type(reply).__name__}")
    if reply.get("error"):
        raise ChatError(str(reply["error"])[:200])
    if not (reply.get("answer") or "").strip():
        raise ChatError("empty answer")


def _post(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--corpus", default="data/originals")
    ap.add_argument("--survey", default="data/eval/table_probe_corpus_survey.json")
    ap.add_argument("--tag", required=True, help="arm label, e.g. armA-2026-09-06")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()

    survey = json.loads(Path(args.survey).read_text(encoding="utf-8"))
    questions = [
        q for q in collect_questions(args.corpus, survey["unique_documents_list"])
        if not q.get("skipped") and row_is_identifiable(q["gold_cells"])
    ]
    if args.limit:
        questions = questions[: args.limit]

    title = f"arm-baseline-{args.tag}"
    conversation = _post(
        f"{args.api}/conversations", {"title": title}, args.timeout
    )
    session_id = conversation["conversation_id"]
    print(f"[baseline] conversation {session_id} ({title})", file=sys.stderr)
    print(f"[baseline] {len(questions)} questions", file=sys.stderr)

    asked = []
    for idx, question in enumerate(questions, start=1):
        message = ask_text(question["row_label"], question["column"])
        started = time.perf_counter()
        try:
            reply = _post(
                f"{args.api}/chat",
                {
                    "question": message,
                    "conversation_id": session_id,
                    "conversation_title": title,
                },
                args.timeout,
            )
            validate_reply(reply)
            error = None
        except (ChatError, urllib.error.URLError, TimeoutError, OSError) as exc:
            error = repr(exc)[:200]
        elapsed = (time.perf_counter() - started) * 1000
        asked.append(
            {
                "doc": question["doc"],
                "page": question["page"],
                "message": message,
                "answer_cell": question["answer_cell"],
                "elapsed_ms": round(elapsed),
                "error": error,
            }
        )
        print(
            f"  {idx}/{len(questions)} {elapsed / 1000:.0f}s"
            + (f" ERROR {error}" if error else ""),
            file=sys.stderr,
        )

    failed = sum(1 for a in asked if a["error"])
    print(f"[baseline] done — {len(asked) - failed} answered, {failed} failed",
          file=sys.stderr)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(
                {"conversation_id": session_id, "title": title, "asked": asked},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
    print(session_id)


if __name__ == "__main__":
    main()
