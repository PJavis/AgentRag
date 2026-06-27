# Eval-Fidelity Phase 2 (minimal): prod-corpus eval set + probe wiring

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Build a small prod-corpus eval set (synthetic Q + gold source-chunk + grounded gold answer over the already-indexed 114 PDFs), and wire the oracle probe to run against it — NO corpus re-ingest. This pivots Phase 1's vn probe onto the real corpus (the live ES index already holds the prod docs; vn gold is not indexed).

**Architecture:** A new generator script samples `Segment.content` rows from Postgres (the indexed corpus), generates a question per chunk and a gold answer grounded only in that chunk (strong model via the `gold_gen` task slot), and writes `EvalExample`-shaped JSONL. A new `local_jsonl` loader in `benchmark_datasets.py` reads it back; the oracle probe gains a `--eval-set` flag to score against it.

**Tech Stack:** Python 3.11+, asyncio, SQLAlchemy (existing `AsyncSessionLocal`), `LLMGateway`, pytest. No new deps.

## Global Constraints

- Source package `src.agentrag.*` — never `src.pam.*`.
- Gateway injected; gold-answer generation uses `task="gold_gen"`, synth-Q uses `task="schema_discovery"` (matches `mine_finetune_pairs.py`).
- Output rows are `EvalExample`-shaped: `{id, question, reference_answer, gold_contexts:[chunk], lang, source:"prod_corpus"}`.
- NO corpus ingest anywhere — read `Segment.content` from PG; the docs are already indexed in ES.
- DB sampling + LLM calls mirror `scripts/mine_finetune_pairs.py::_mine_synthetic_positives` (reuse its shape).
- Tests: `uv run pytest <path> -v`. Repo: `pythonpath=["."]`, `asyncio_mode="auto"`, pytest-asyncio present. `from scripts.eval.X import ...` works.

---

### Task 1: prod-corpus eval-set generator

**Files:**
- Create: `scripts/eval/build_prod_evalset.py`
- Test: `tests/eval/test_build_prod_evalset.py`

**Interfaces:**
- Produces: `detect_lang(text: str) -> str`; `build_eval_row(idx: int, question: str, gold_answer: str, chunk: str, source: str = "prod_corpus") -> dict`; `async synth_question(chunk: str, gateway) -> str | None`; `async gen_gold_answer(question: str, chunk: str, gateway) -> str`.
- Gateway contract: `await gateway.json_response(system_prompt=, user_prompt=, task=) -> (dict, float)`; `await gateway.text_response(system_prompt=, user_prompt=, task=) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_build_prod_evalset.py
import pytest

from scripts.eval.build_prod_evalset import (
    detect_lang, build_eval_row, synth_question, gen_gold_answer,
)


class FakeGateway:
    def __init__(self, json_payloads=None, text="gold answer here"):
        self._json = list(json_payloads or [])
        self._text = text
        self.calls = []

    async def json_response(self, system_prompt, user_prompt, task="general"):
        self.calls.append({"task": task, "user": user_prompt})
        return self._json.pop(0), 1.0

    async def text_response(self, system_prompt, user_prompt, task="general"):
        self.calls.append({"task": task, "user": user_prompt})
        return self._text


def test_detect_lang_vietnamese():
    assert detect_lang("Bệnh nhân được chẩn đoán xơ gan giai đoạn cuối") == "vi"


def test_detect_lang_english():
    assert detect_lang("The patient was diagnosed with end-stage cirrhosis") == "en"


def test_build_eval_row_shape():
    row = build_eval_row(3, "What is X?", "X is a thing.", "X is a thing defined in the source chunk.")
    assert row["id"] == "prod_corpus-3"
    assert row["question"] == "What is X?"
    assert row["reference_answer"] == "X is a thing."
    assert row["gold_contexts"] == ["X is a thing defined in the source chunk."]
    assert row["source"] == "prod_corpus"
    assert row["lang"] in ("vi", "en")


@pytest.mark.asyncio
async def test_synth_question_takes_first():
    gw = FakeGateway(json_payloads=[{"questions": ["Q1?", "Q2?"]}])
    q = await synth_question("a chunk of text long enough to matter", gw)
    assert q == "Q1?"
    assert gw.calls[0]["task"] == "schema_discovery"


@pytest.mark.asyncio
async def test_synth_question_none_when_empty():
    gw = FakeGateway(json_payloads=[{"questions": []}])
    assert await synth_question("chunk", gw) is None


@pytest.mark.asyncio
async def test_gen_gold_answer_uses_chunk_and_task():
    gw = FakeGateway(text="A grounded gold answer.")
    out = await gen_gold_answer("What is X?", "X is defined here in the chunk.", gw)
    assert out == "A grounded gold answer."
    assert gw.calls[0]["task"] == "gold_gen"
    assert "X is defined here in the chunk." in gw.calls[0]["user"]
```

- [ ] **Step 2: Run tests → fail**

Run: `uv run pytest tests/eval/test_build_prod_evalset.py -v`
Expected: FAIL (`ModuleNotFoundError: scripts.eval.build_prod_evalset`).

- [ ] **Step 3: Write the generator**

```python
#!/usr/bin/env python
"""Build a small prod-corpus eval set from the already-indexed corpus.

For N sampled corpus chunks (Segment.content): generate a question the chunk
answers, then a rich gold answer grounded ONLY in that chunk. Emit EvalExample-
shaped JSONL: {id, question, reference_answer, gold_contexts:[chunk], lang, source}.

No ingest — the docs are already in ES; this only reads Segment.content from PG.

Usage:
    uv run python scripts/eval/build_prod_evalset.py --n 30 \\
        --out data/eval/prod_corpus_evalset.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


_SYNTH_SYSTEM = (
    "You are a search-query generator. Given a passage, write ONE short, specific "
    "question that this passage directly answers. Same language as the passage. "
    "No yes/no question, no external knowledge. Return strict JSON: {\"questions\": [\"...\"]}"
)

_GOLD_SYSTEM = (
    "Answer the question using ONLY the provided context. Be complete, precise, and "
    "self-contained — a strong reference answer. Do not add facts beyond the context. "
    "Answer in the context's language."
)


def detect_lang(text: str) -> str:
    """Cheap heuristic: Vietnamese (and other diacritic-heavy text) trips the
    non-ASCII ratio; default to English otherwise."""
    if not text:
        return "en"
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return "vi" if non_ascii / max(len(text), 1) > 0.02 else "en"


def build_eval_row(idx: int, question: str, gold_answer: str, chunk: str,
                   source: str = "prod_corpus") -> dict:
    return {
        "id": f"{source}-{idx}",
        "question": question.strip(),
        "reference_answer": gold_answer.strip(),
        "gold_contexts": [chunk],
        "lang": detect_lang(chunk),
        "source": source,
    }


async def synth_question(chunk: str, gateway) -> str | None:
    payload, _ = await gateway.json_response(
        system_prompt=_SYNTH_SYSTEM,
        user_prompt=json.dumps({"passage": chunk[:2000]}, ensure_ascii=False),
        task="schema_discovery",
    )
    qs = payload.get("questions") or []
    for q in qs:
        if isinstance(q, str) and len(q.strip()) >= 5:
            return q.strip()
    return None


async def gen_gold_answer(question: str, chunk: str, gateway) -> str:
    user = f"CONTEXT:\n{chunk}\n\nQUESTION: {question}"
    return (await gateway.text_response(
        system_prompt=_GOLD_SYSTEM, user_prompt=user, task="gold_gen"
    )).strip()


async def _sample_chunks(n: int) -> list[str]:
    from sqlalchemy import select
    from src.agentrag.database import AsyncSessionLocal
    from src.agentrag.database.models import Segment

    async with AsyncSessionLocal() as s:
        rows = (
            await s.execute(
                select(Segment.content)
                .where(Segment.content.isnot(None))
                .order_by(Segment.id)
                .limit(n * 4)  # over-sample, then filter+shuffle
            )
        ).scalars().all()
    pool = [c for c in rows if c and len(c) >= 200]
    random.shuffle(pool)
    return pool[:n]


async def main(args: argparse.Namespace) -> None:
    from src.agentrag.services.llm_gateway import LLMGateway

    gateway = LLMGateway()
    chunks = await _sample_chunks(args.n)
    logger.info("sampled %d chunks (target %d)", len(chunks), args.n)

    rows: list[dict] = []
    sem = asyncio.Semaphore(4)

    async def _one(idx: int, chunk: str) -> None:
        async with sem:
            try:
                q = await synth_question(chunk, gateway)
                if not q:
                    return
                gold = await gen_gold_answer(q, chunk, gateway)
                if gold:
                    rows.append(build_eval_row(idx, q, gold, chunk))
            except Exception as exc:
                logger.warning("row %d failed: %s", idx, exc)

    await asyncio.gather(*[_one(i, c) for i, c in enumerate(chunks)])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("wrote %d eval rows → %s", len(rows), out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=30, help="number of chunks to sample")
    p.add_argument("--out", default="data/eval/prod_corpus_evalset.jsonl")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
```

- [ ] **Step 4: Run tests → pass**

Run: `uv run pytest tests/eval/test_build_prod_evalset.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/build_prod_evalset.py tests/eval/test_build_prod_evalset.py
git commit -m "feat(eval): prod-corpus eval-set generator (synth-Q + grounded gold answer)"
```

---

### Task 2: local_jsonl loader + probe `--eval-set` flag

**Files:**
- Modify: `src/agentrag/eval/benchmark_datasets.py`
- Modify: `scripts/eval/oracle_probe.py`
- Test: `tests/eval/test_local_evalset.py`

**Interfaces:**
- Consumes: `EvalExample` from `benchmark_datasets`.
- Produces: `load_local_jsonl(path: str, n: int | None = None) -> list[EvalExample]` in `benchmark_datasets.py`; `oracle_probe.main` accepts `--eval-set <jsonl path>` and, when set, loads via `load_local_jsonl` instead of `load_suite`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_local_evalset.py
import json

from src.agentrag.eval.benchmark_datasets import load_local_jsonl, EvalExample


def test_load_local_jsonl_roundtrip(tmp_path):
    p = tmp_path / "eval.jsonl"
    rows = [
        {"id": "prod_corpus-0", "question": "Q0?", "reference_answer": "A0",
         "gold_contexts": ["chunk 0"], "lang": "vi", "source": "prod_corpus"},
        {"id": "prod_corpus-1", "question": "Q1?", "reference_answer": "A1",
         "gold_contexts": ["chunk 1"], "lang": "en", "source": "prod_corpus"},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    out = load_local_jsonl(str(p))
    assert len(out) == 2
    assert isinstance(out[0], EvalExample)
    assert out[0].id == "prod_corpus-0"
    assert out[0].question == "Q0?"
    assert out[0].reference_answer == "A0"
    assert out[0].gold_contexts == ["chunk 0"]
    assert out[0].lang == "vi"


def test_load_local_jsonl_skips_blank_and_invalid(tmp_path):
    p = tmp_path / "eval.jsonl"
    p.write_text(
        '{"id":"a","question":"Q?","reference_answer":"A","gold_contexts":["c"]}\n'
        "\n"  # blank line
        '{"question":"","gold_contexts":[]}\n',  # no question/contexts → skipped
        encoding="utf-8",
    )
    out = load_local_jsonl(str(p))
    assert len(out) == 1
    assert out[0].id == "a"


def test_load_local_jsonl_n_cap(tmp_path):
    p = tmp_path / "eval.jsonl"
    p.write_text("\n".join(
        json.dumps({"id": f"r{i}", "question": f"Q{i}?", "reference_answer": "A",
                    "gold_contexts": ["c"]}) for i in range(5)
    ), encoding="utf-8")
    assert len(load_local_jsonl(str(p), n=2)) == 2
```

- [ ] **Step 2: Run test → fail**

Run: `uv run pytest tests/eval/test_local_evalset.py -v`
Expected: FAIL (`ImportError: cannot import name 'load_local_jsonl'`).

- [ ] **Step 3: Add the loader to `benchmark_datasets.py`**

Append after `load_suite`:

```python
def load_local_jsonl(path: str, n: int | None = None) -> list[EvalExample]:
    """Load EvalExamples from a local JSONL file (prod-corpus eval set).

    Each line is a JSON object with at least `question` and `gold_contexts`.
    Rows missing either are skipped (they can't be scored). `n` caps the count.
    """
    import json as _json

    out: list[EvalExample] = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            question = str(row.get("question", "")).strip()
            contexts = _as_context_list(row.get("gold_contexts"))
            if not question or not contexts:
                continue
            out.append(EvalExample(
                id=str(row.get("id") or f"local-{idx}"),
                question=question,
                reference_answer=str(row.get("reference_answer", "")).strip(),
                gold_contexts=contexts,
                lang=str(row.get("lang", "en")),
                source=str(row.get("source", "local")),
            ))
            if n is not None and len(out) >= n:
                break
    return out
```

- [ ] **Step 4: Add `--eval-set` to the probe**

In `scripts/eval/oracle_probe.py`:

In `parse_args()`, add:
```python
    p.add_argument("--eval-set", default=None,
                   help="path to a local JSONL eval set (EvalExample shape); overrides --suite")
```

In `main()`, replace the `examples = load_suite(...)` line with:
```python
    if args.eval_set:
        from src.agentrag.eval.benchmark_datasets import load_local_jsonl
        examples = load_local_jsonl(args.eval_set, n=args.n)
        print(f"[probe] {len(examples)} examples (eval-set={args.eval_set}, n={args.n})")
    else:
        examples = load_suite(args.suite, n=args.n)
        print(f"[probe] {len(examples)} examples (suite={args.suite}, n={args.n})")
```
(Delete the old single `examples = load_suite(...)` + its print.)

- [ ] **Step 5: Run tests → pass**

Run: `uv run pytest tests/eval/test_local_evalset.py tests/eval/test_oracle_probe.py -v`
Expected: PASS (existing probe tests unaffected; 3 new pass).

- [ ] **Step 6: Commit**

```bash
git add src/agentrag/eval/benchmark_datasets.py scripts/eval/oracle_probe.py tests/eval/test_local_evalset.py
git commit -m "feat(eval): local_jsonl loader + oracle probe --eval-set flag (prod-corpus probe)"
```

---

### Task 3: build the set + run the probe (manual, operational)

- [ ] **Step 1:** `uv run python scripts/eval/build_prod_evalset.py --n 30 --out data/eval/prod_corpus_evalset.jsonl` — confirm row count (~25-30) and spot-check 2-3 rows (question matches chunk; gold answer is grounded + complete).
- [ ] **Step 2:** `uv run python scripts/eval/oracle_probe.py --eval-set data/eval/prod_corpus_evalset.jsonl --n 30 --out docs/eval/eval_fidelity_probe_prod_2026-06-26.md` — needs the stack up + gemini eval slots (set in `.env`).
- [ ] **Step 3:** Record the gate decision in the doc: oracle−system gap (small → eval-capped; large → system headroom) + judge-noise pearson.
- [ ] **Step 4:** `git add data/eval/prod_corpus_evalset.jsonl docs/eval/eval_fidelity_probe_prod_2026-06-26.md && git commit -m "eval: prod-corpus oracle probe results + gate decision"` (check `.gitignore` — `data/eval/` raw reports may be ignored; commit only the doc if so).

## Self-Review
- Generator emits exact `EvalExample` shape (Task 1 `build_eval_row` ↔ Task 2 loader fields) — consistent. ✓
- `synth_question` task=`schema_discovery`, `gen_gold_answer` task=`gold_gen` — both mapped in `.env`. ✓
- No ingest; reads `Segment.content` only (mirrors `mine_finetune_pairs`). ✓
- Probe reuses the ensemble judge from Phase 1 unchanged; only the example source changes. ✓
- Placeholder scan: none. Type consistency: `load_local_jsonl` returns `list[EvalExample]`, consumed by probe `main`. ✓
