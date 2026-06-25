# Preference-data Miner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `scripts/mine_preference.py` that turns captured thumbs feedback into KTO (default) or DPO preference-tuning JSONL.

**Architecture:** Pure builder functions (`build_kto_records`, `build_dpo_pairs`, `_normalize_q`) operating on plain dicts — the unit-testable core — plus a thin async DB loader and an argparse CLI that mirror `scripts/mine_sft.py`.

**Tech Stack:** Python async (SQLAlchemy `AsyncSessionLocal`), argparse, JSONL. No new deps. Output schemas target TRL KTOTrainer / DPOTrainer.

## Global Constraints

- `adapter_chat_feedback` row already has `question`/`answer`/`rating` — do NOT join `chat_messages`.
- KTO record: `{"prompt": str, "completion": str, "label": bool}` (label = `rating == 1`). DPO pair: `{"prompt": str, "chosen": str, "rejected": str}`.
- Constants: `MIN_ANSWER_CHARS = 30`, `MAX_PAIRS_PER_GROUP = 8`. Valid rating ∈ `(1, -1)`.
- Builders skip malformed rows (never raise); empty/no-match → empty output.
- Tests import via `from scripts.mine_preference import ...` (works: `pythonpath=["."]`, namespace package — no `__init__.py`).
- DB `load_rows` is thin and NOT unit-tested here (no Postgres on this host).

---

## File Structure

| Path | Responsibility |
|---|---|
| `scripts/mine_preference.py` (create) | pure builders + DB loader + CLI |
| `tests/eval/test_mine_preference.py` (create) | offline unit tests for the pure builders |
| `Makefile` (modify) | add `mine-preference` target |

---

### Task 1: mine_preference.py (builders TDD) + Makefile target

**Files:**
- Create: `scripts/mine_preference.py`
- Create: `tests/eval/test_mine_preference.py`
- Modify: `Makefile` (add `mine-preference` target near `mine-pairs`)

**Interfaces:**
- Consumes: `src.agentrag.adapter.db.AdapterChatFeedback`, `src.agentrag.database.AsyncSessionLocal` (in the DB loader only).
- Produces: `_normalize_q(question: str) -> str`, `build_kto_records(rows: list[dict]) -> list[dict]`, `build_dpo_pairs(rows: list[dict]) -> list[dict]`, `MAX_PAIRS_PER_GROUP: int`. `rows` items are `{"question": str, "answer": str, "rating": int}`.

- [ ] **Step 1: Write the failing builder tests.**

```python
# tests/eval/test_mine_preference.py
from scripts.mine_preference import (
    MAX_PAIRS_PER_GROUP,
    _normalize_q,
    build_dpo_pairs,
    build_kto_records,
)

_LONG = "x" * 40  # ≥ MIN_ANSWER_CHARS (30)


def _row(q, a, r):
    return {"question": q, "answer": a, "rating": r}


def test_kto_labels_and_skips():
    rows = [
        _row("Q1?", "good " + _LONG, 1),
        _row("Q2?", "bad " + _LONG, -1),
        _row("", "no question " + _LONG, 1),   # skip: empty question
        _row("Q3?", "short", 1),                # skip: answer < 30 chars
        _row("Q4?", "zero rating " + _LONG, 0), # skip: rating not in (1,-1)
    ]
    recs = build_kto_records(rows)
    assert len(recs) == 2
    assert recs[0] == {"prompt": "Q1?", "completion": "good " + _LONG, "label": True}
    assert recs[1]["label"] is False


def test_kto_empty():
    assert build_kto_records([]) == []


def test_dpo_pairs_same_question_normalized():
    rows = [
        _row("What is X?", "chosen " + _LONG, 1),
        _row(" what is x? ", "rejected " + _LONG, -1),  # normalizes into same group
    ]
    pairs = build_dpo_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "chosen " + _LONG
    assert pairs[0]["rejected"] == "rejected " + _LONG
    assert pairs[0]["prompt"] == "What is X?"


def test_dpo_no_pair_single_polarity():
    rows = [_row("Q?", "up1 " + _LONG, 1), _row("Q?", "up2 " + _LONG, 1)]
    assert build_dpo_pairs(rows) == []


def test_dpo_identical_answer_excluded():
    same = "same " + _LONG
    rows = [_row("Q?", same, 1), _row("Q?", same, -1)]
    assert build_dpo_pairs(rows) == []


def test_dpo_cap_per_group():
    rows = [_row("Q?", f"up{i} " + _LONG, 1) for i in range(5)] + \
           [_row("Q?", f"down{i} " + _LONG, -1) for i in range(5)]  # 25 possible
    assert len(build_dpo_pairs(rows)) == MAX_PAIRS_PER_GROUP


def test_normalize_q():
    assert _normalize_q("  Hello   World? ") == "hello world?"
    assert _normalize_q(None) == ""
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `uv run pytest tests/eval/test_mine_preference.py -q`
Expected: FAIL — `ImportError: cannot import name '_normalize_q' from 'scripts.mine_preference'` (module absent).

- [ ] **Step 3: Write `scripts/mine_preference.py`.**

```python
#!/usr/bin/env python
"""Mine preference-tuning data from thumbs feedback (adapter_chat_feedback).

KTO (default): one {prompt, completion, label} per rated turn — the correct fit
for binary thumbs. DPO (--format dpo): {prompt, chosen, rejected} pairs where the
SAME question was rated both up and down.

Usage:
    uv run python scripts/mine_preference.py --out data/finetune/preference.jsonl --format kto
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from sqlalchemy import select

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_ANSWER_CHARS = 30
MAX_PAIRS_PER_GROUP = 8
_WS = re.compile(r"\s+")


def _normalize_q(question: str | None) -> str:
    return _WS.sub(" ", (question or "").strip().lower())


def _valid(row: dict[str, Any]) -> tuple[str, str, int] | None:
    q = (row.get("question") or "").strip()
    a = (row.get("answer") or "").strip()
    rating = row.get("rating")
    if not q or len(a) < MIN_ANSWER_CHARS or rating not in (1, -1):
        return None
    return q, a, int(rating)


def build_kto_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        v = _valid(row)
        if v is None:
            continue
        q, a, rating = v
        out.append({"prompt": q, "completion": a, "label": rating == 1})
    return out


def build_dpo_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        v = _valid(row)
        if v is None:
            continue
        q, a, rating = v
        g = groups.setdefault(_normalize_q(q), {"prompt": q, "chosen": [], "rejected": []})
        bucket = g["chosen"] if rating == 1 else g["rejected"]
        if a not in bucket:
            bucket.append(a)
    pairs: list[dict[str, Any]] = []
    for g in groups.values():
        n = 0
        for chosen in g["chosen"]:
            for rejected in g["rejected"]:
                if chosen == rejected:
                    continue
                pairs.append({"prompt": g["prompt"], "chosen": chosen, "rejected": rejected})
                n += 1
                if n >= MAX_PAIRS_PER_GROUP:
                    break
            if n >= MAX_PAIRS_PER_GROUP:
                break
    return pairs


async def load_rows(limit: int) -> list[dict[str, Any]]:
    from src.agentrag.adapter.db import AdapterChatFeedback
    from src.agentrag.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(
                AdapterChatFeedback.question,
                AdapterChatFeedback.answer,
                AdapterChatFeedback.rating,
            )
            .where(AdapterChatFeedback.question.isnot(None))
            .where(AdapterChatFeedback.answer.isnot(None))
            .limit(limit)
        )
    return [{"question": q, "answer": a, "rating": r} for (q, a, r) in result.all()]


async def main(args: argparse.Namespace) -> None:
    rows = await load_rows(args.limit)
    records = build_kto_records(rows) if args.format == "kto" else build_dpo_pairs(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("wrote %d %s records → %s", len(records), args.format.upper(), out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/finetune/preference.jsonl")
    p.add_argument("--format", choices=["kto", "dpo"], default="kto")
    p.add_argument("--limit", type=int, default=5000)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
```

- [ ] **Step 4: Run the tests to verify they pass.**

Run: `uv run pytest tests/eval/test_mine_preference.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Smoke-check the module imports cleanly (incl. the DB-loader path).**

Run: `uv run python -c "import scripts.mine_preference as m; print(m.MAX_PAIRS_PER_GROUP, m.parse_args.__name__)"`
Expected: prints `8 parse_args` with no import error (validates the SQLAlchemy/DB imports inside `load_rows` resolve at module level — they're deferred into the function, so import must not touch the DB).

- [ ] **Step 6: Add the Makefile target.** After the `mine-pairs` target (~line 397), add:

```makefile
.PHONY: mine-preference
mine-preference:
	uv run python scripts/mine_preference.py \
	  --out data/finetune/preference.jsonl --format kto
```

- [ ] **Step 7: Confirm no broader regression.**

Run: `uv run pytest -q --ignore=tests/ontology --ignore=tests/ingestion`
Expected: PASS (prior green count + 7 new).

- [ ] **Step 8: Commit.**

```bash
git add scripts/mine_preference.py tests/eval/test_mine_preference.py Makefile
git commit -m "feat(finetune): preference miner — KTO default + DPO same-question pairs from thumbs

Reads adapter_chat_feedback (question/answer/rating) → KTO {prompt,completion,label}
(default) or DPO {prompt,chosen,rejected} same-question pairs (--format dpo). Closes
the FINETUNE_STRATEGY step-4 DPO gap. Pure builders unit-tested offline; thin async DB
loader mirrors mine_sft.py. make mine-preference target added."
```

---

## Self-Review

**Spec coverage:** `_normalize_q`/`build_kto_records`/`build_dpo_pairs` → Step 3 + tests Step 1; `load_rows` thin DB wrapper → Step 3; CLI (`--out`/`--format`/`--limit`) → Step 3; Makefile target → Step 6; KTO/DPO schemas, constants (`MIN_ANSWER_CHARS=30`, `MAX_PAIRS_PER_GROUP=8`), skip-not-raise, empty→empty → Step 3 + tests. All spec sections mapped.

**Placeholder scan:** none — full module + tests + Makefile snippet inline, exact commands + expected output.

**Type consistency:** `rows` item shape `{"question","answer","rating"}` identical across `_valid`, both builders, `load_rows`, and the tests' `_row` helper. KTO keys `prompt/completion/label`, DPO keys `prompt/chosen/rejected`, constant `MAX_PAIRS_PER_GROUP` consistent between Step 3 impl and the cap test.
