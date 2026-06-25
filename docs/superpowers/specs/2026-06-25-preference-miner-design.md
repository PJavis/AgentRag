# Preference-data miner (KTO + DPO) — design

**Date:** 2026-06-25 · **Objective:** turn captured thumbs feedback into
preference-tuning data (KTO default; DPO same-question pairs), closing the
`FINETUNE_STRATEGY.md` step-4 "DPO" gap.

## Context

Feedback capture is built and now migration-backed (`adapter_chat_feedback`, rev
`2026062501`). `FINETUNE_STRATEGY.md` plans DPO (step 4, target 200–1k preferences,
"`rating=+1` → positives, `-1` → DPO rejected") but no preference miner exists —
only `mine_finetune_pairs.py` (embedding/reranker triplets) and `mine_sft.py` (SFT).

Thumbs are **binary per turn**, so KTO (binary `{prompt, completion, label}`) is the
correct primary fit; real DPO pairs (`{prompt, chosen, rejected}`) only exist where the
*same question* was rated both ways, so they are a secondary, sparse output.

The `adapter_chat_feedback` row already stores `question`, `answer`, `rating` (the
endpoint persists them), so the miner needs **no** join to `chat_messages`.

## Scope

In: `scripts/mine_preference.py` (KTO default + `--format dpo`), unit tests, a
`make mine-preference` target. Out: synthetic-DPO via LLM; the actual KTO/DPO training
script; retrieved-context enrichment of rows.

## Design

**File:** `scripts/mine_preference.py` — structure mirrors `scripts/mine_sft.py`
(argparse, `AsyncSessionLocal`, JSONL out).

### Pure functions (no DB, no LLM — the testable core)
- `_normalize_q(question: str) -> str` — `question.strip().lower()` with internal
  whitespace collapsed to single spaces. Empty/None → `""`.
- `build_kto_records(rows) -> list[dict]` — input `rows` = list of
  `{"question": str, "answer": str, "rating": int}`. For each row with non-empty
  question, an answer of length ≥ `MIN_ANSWER_CHARS` (30), and `rating in (1, -1)`,
  emit `{"prompt": question, "completion": answer, "label": rating == 1}`. Else skip.
- `build_dpo_pairs(rows) -> list[dict]` — group valid rows by `_normalize_q(question)`.
  In each group, `chosen = {distinct answers with rating==1}`,
  `rejected = {distinct answers with rating==-1}`; emit
  `{"prompt": <first raw question in group>, "chosen": c, "rejected": r}` for every
  `c in chosen, r in rejected` where `c != r`. No emit when either set is empty.
  Cap at `MAX_PAIRS_PER_GROUP` (8) per group to avoid combinatorial blowup.

### DB wrapper (thin, the only untested-here surface)
- `async def load_rows(session, limit) -> list[dict]` —
  `select(AdapterChatFeedback.question, AdapterChatFeedback.answer,
  AdapterChatFeedback.rating).where(question.isnot(None), answer.isnot(None))
  .limit(limit)` → list of `{"question","answer","rating"}` dicts.

### CLI / main
- args: `--out` (default `data/finetune/preference.jsonl`),
  `--format` (`kto`|`dpo`, default `kto`), `--limit` (default 5000).
- main: open `AsyncSessionLocal` → `load_rows` → `build_kto_records` or
  `build_dpo_pairs` by format → write one JSON object per line → log count.

### Makefile
- `mine-preference:` → `uv run python scripts/mine_preference.py --out data/finetune/preference.jsonl --format kto`
  (parity with `mine-pairs`).

## Output schemas
- KTO (TRL KTOTrainer): `{"prompt": str, "completion": str, "label": bool}`.
- DPO (TRL DPOTrainer): `{"prompt": str, "chosen": str, "rejected": str}`.

## Data flow
`adapter_chat_feedback` (question, answer, rating) → `load_rows` → builder(format) →
JSONL at `--out` → (later, out of scope) TRL KTO/DPO training.

## Testing (offline, no DB/LLM)
`tests/eval/test_mine_preference.py`:
- `build_kto_records`: 👍→label True, 👎→label False; skips missing question, short
  answer, and `rating` not in (1,−1); empty input → `[]`.
- `build_dpo_pairs`: same-question 👍×👎 → one pair (chosen=up, rejected=down); single
  polarity → no pair; `_normalize_q` groups `"Q?"` and `" q? "`; identical up/down
  answer excluded; cap respected.
- `_normalize_q`: case + whitespace normalization; None → `""`.
The DB `load_rows` wrapper is intentionally thin and not unit-tested here (no Postgres
on this host); it runs against real data in production / a future CI smoke.

## Error handling
Builders skip malformed rows rather than raise. Empty/no-match → empty JSONL (no
crash). `--out` parent dir created if missing (`mkdir parents=True`).
