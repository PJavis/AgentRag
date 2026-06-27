# Eval-Fidelity Phase 1 (Ensemble Judge + Oracle Probe) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reference-based **ensemble correctness judge** (nugget-recall + reference-guided rubric) and a no-ingest **oracle probe** that proves the 0.74 `answer_correctness` plateau is the ruler, not the system.

**Architecture:** A new `correctness_judge.py` module splits into pure aggregation helpers (unit-tested with no LLM) and async judge orchestration (tested with a scripted `FakeGateway`). A probe script reuses the judge to score an oracle answer (strong model + gold context) against gold, alongside the live system answer, plus a second-judge noise floor — writing a markdown verdict.

**Tech Stack:** Python 3.11+, asyncio, pytest, `LLMGateway` (`json_response`/`text_response`), `dataclasses`. No new dependencies.

## Global Constraints

- Source package is `src.agentrag.*` — **never** `src.pam.*` (pam retired; master = agentrag).
- Judge LLM is always **injected** as a `gateway` parameter — no module-level gateway construction (keeps functions testable with a fake).
- `LLMGateway.json_response(system_prompt, user_prompt, task="...")` returns `tuple[dict, float]` (payload, latency_ms). `text_response(...)` returns `str`. Use `task="eval_judge"` for all judge calls.
- All scores clamped to `[0.0, 1.0]`.
- Phase 1 performs **no corpus ingest**. The vn suite must already be indexed (run with `--skip-ingest`).

---

### Task 1: Pure aggregation helpers + dataclasses

**Files:**
- Create: `src/agentrag/eval/correctness_judge.py`
- Test: `tests/eval/test_correctness_judge.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `clamp01(x) -> float`; `parse_nuggets(raw: dict) -> list[str]`; `NuggetScore` dataclass (`n_total, n_covered, n_contradicted, recall, contradiction_penalty, score`); `aggregate_nugget_labels(labels: list[str]) -> NuggetScore`; `EnsembleScore` dataclass (`nugget, rubric, mean, abs_delta, low_confidence`); `ensemble(nugget: float, rubric: float, *, delta_threshold: float = 0.2) -> EnsembleScore`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_correctness_judge.py
from src.agentrag.eval.correctness_judge import (
    clamp01, parse_nuggets, aggregate_nugget_labels, ensemble,
    NuggetScore, EnsembleScore,
)


def test_clamp01_bounds():
    assert clamp01(-0.3) == 0.0
    assert clamp01(1.7) == 1.0
    assert clamp01(0.42) == 0.42


def test_parse_nuggets_strips_and_drops_empty():
    raw = {"nuggets": ["  fact A ", "", "fact B", 5, None]}
    assert parse_nuggets(raw) == ["fact A", "fact B"]


def test_parse_nuggets_missing_key():
    assert parse_nuggets({}) == []


def test_aggregate_nugget_labels_recall_minus_contradiction():
    # 4 nuggets: 2 covered, 1 contradicted, 1 absent
    s = aggregate_nugget_labels(["covered", "covered", "contradicted", "absent"])
    assert s.n_total == 4
    assert s.n_covered == 2
    assert s.n_contradicted == 1
    assert s.recall == 0.5
    assert s.contradiction_penalty == 0.25
    assert s.score == 0.25  # max(0, 0.5 - 0.25)


def test_aggregate_nugget_labels_empty_is_zero():
    s = aggregate_nugget_labels([])
    assert s == NuggetScore(0, 0, 0, 0.0, 0.0, 0.0)


def test_aggregate_floor_at_zero():
    # contradictions exceed coverage → score floored, not negative
    s = aggregate_nugget_labels(["contradicted", "contradicted", "covered"])
    assert s.score == 0.0


def test_ensemble_mean_and_delta_flag():
    e = ensemble(0.6, 0.7)
    assert e.nugget == 0.6
    assert e.rubric == 0.7
    assert e.mean == 0.65
    assert abs(e.abs_delta - 0.1) < 1e-9
    assert e.low_confidence is False


def test_ensemble_flags_high_delta():
    e = ensemble(0.2, 0.9)
    assert e.low_confidence is True  # abs_delta 0.7 > 0.2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_correctness_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: ... correctness_judge` / `ImportError`.

- [ ] **Step 3: Write the module (pure layer only)**

```python
# src/agentrag/eval/correctness_judge.py
"""Reference-based, phrasing-robust correctness judge.

Two scorers, ensembled:
  - nugget-recall: decompose GOLD into atomic must-have facts, score how many the
    answer covers, penalize only contradictions (extra true info is free).
  - reference-guided rubric: one anchored 0-1 judgement given Q + gold + gold context.

The pure aggregation functions below carry no LLM dependency and are unit-tested
directly. Async orchestration (Task 2) injects an LLM gateway.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def parse_nuggets(raw: dict) -> list[str]:
    """Extract a clean list of non-empty nugget strings from a judge payload."""
    items = raw.get("nuggets") or []
    return [n.strip() for n in items if isinstance(n, str) and n.strip()]


@dataclass
class NuggetScore:
    n_total: int
    n_covered: int
    n_contradicted: int
    recall: float
    contradiction_penalty: float
    score: float


def aggregate_nugget_labels(labels: list[str]) -> NuggetScore:
    """labels are per-nugget verdicts in {"covered","contradicted","absent"}.

    recall = covered/total ; penalty = contradicted/total ;
    score = max(0, recall - penalty).
    """
    total = len(labels)
    if total == 0:
        return NuggetScore(0, 0, 0, 0.0, 0.0, 0.0)
    covered = sum(1 for l in labels if l == "covered")
    contradicted = sum(1 for l in labels if l == "contradicted")
    recall = covered / total
    penalty = contradicted / total
    return NuggetScore(total, covered, contradicted, recall, penalty, clamp01(recall - penalty))


@dataclass
class EnsembleScore:
    nugget: float
    rubric: float
    mean: float
    abs_delta: float
    low_confidence: bool


def ensemble(nugget: float, rubric: float, *, delta_threshold: float = 0.2) -> EnsembleScore:
    nugget = clamp01(nugget)
    rubric = clamp01(rubric)
    abs_delta = abs(nugget - rubric)
    return EnsembleScore(
        nugget=nugget,
        rubric=rubric,
        mean=(nugget + rubric) / 2,
        abs_delta=abs_delta,
        low_confidence=abs_delta > delta_threshold,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_correctness_judge.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/eval/correctness_judge.py tests/eval/test_correctness_judge.py
git commit -m "feat(eval): nugget-recall + ensemble pure helpers for correctness judge"
```

---

### Task 2: Async judge orchestration

**Files:**
- Modify: `src/agentrag/eval/correctness_judge.py`
- Test: `tests/eval/test_correctness_judge_async.py`

**Interfaces:**
- Consumes: `parse_nuggets`, `aggregate_nugget_labels`, `ensemble`, `NuggetScore`, `EnsembleScore`, `clamp01` from Task 1.
- Produces: `async extract_nuggets(gold: str, gateway) -> list[str]`; `async score_nuggets(question: str, answer: str, nuggets: list[str], gateway) -> NuggetScore`; `async score_rubric(question: str, answer: str, gold: str, gold_context: str, gateway) -> float`; `async score_correctness(question: str, answer: str, gold: str, gold_context: str, gateway) -> EnsembleScore`. Gateway contract: `await gateway.json_response(system_prompt, user_prompt, task="eval_judge") -> (dict, float)`.

- [ ] **Step 1: Write the failing tests (with a scripted FakeGateway)**

```python
# tests/eval/test_correctness_judge_async.py
import pytest

from src.agentrag.eval.correctness_judge import (
    extract_nuggets, score_nuggets, score_rubric, score_correctness,
)


class FakeGateway:
    """Returns scripted json payloads in order; records calls."""
    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = []

    async def json_response(self, system_prompt, user_prompt, task="general"):
        self.calls.append({"task": task, "system": system_prompt, "user": user_prompt})
        return self._payloads.pop(0), 1.0


@pytest.mark.asyncio
async def test_extract_nuggets_calls_gateway_and_parses():
    gw = FakeGateway([{"nuggets": ["A", "B"]}])
    out = await extract_nuggets("gold text", gw)
    assert out == ["A", "B"]
    assert gw.calls[0]["task"] == "eval_judge"
    assert "gold text" in gw.calls[0]["user"]


@pytest.mark.asyncio
async def test_score_nuggets_aggregates_labels():
    gw = FakeGateway([{"labels": ["covered", "absent"]}])
    s = await score_nuggets("q", "ans", ["A", "B"], gw)
    assert s.n_total == 2
    assert s.recall == 0.5
    assert s.score == 0.5


@pytest.mark.asyncio
async def test_score_nuggets_empty_skips_gateway():
    gw = FakeGateway([])  # would IndexError if called
    s = await score_nuggets("q", "ans", [], gw)
    assert s.score == 0.0
    assert gw.calls == []


@pytest.mark.asyncio
async def test_score_rubric_clamps():
    gw = FakeGateway([{"score": 1.4}])
    r = await score_rubric("q", "ans", "gold", "ctx", gw)
    assert r == 1.0


@pytest.mark.asyncio
async def test_score_correctness_full_flow():
    # 1) extract nuggets, 2) label them, 3) rubric score
    gw = FakeGateway([
        {"nuggets": ["A", "B", "C", "D"]},
        {"labels": ["covered", "covered", "contradicted", "absent"]},
        {"score": 0.7},
    ])
    e = await score_correctness("q", "ans", "gold", "ctx", gw)
    assert e.nugget == 0.25     # recall 0.5 - penalty 0.25
    assert e.rubric == 0.7
    assert e.mean == 0.475
    assert e.low_confidence is True  # |0.25-0.7| = 0.45 > 0.2
    assert len(gw.calls) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_correctness_judge_async.py -v`
Expected: FAIL with `ImportError: cannot import name 'extract_nuggets'`.

- [ ] **Step 3: Append the async layer to the module**

```python
# append to src/agentrag/eval/correctness_judge.py

_NUGGET_EXTRACT_SYSTEM = (
    "You extract atomic factual claims from a reference answer. Break the "
    "reference into the smallest standalone must-have facts a correct answer "
    "must convey. Do not invent facts not present. Same language as the input. "
    'Return strict JSON: {"nuggets": ["fact 1", "fact 2", ...]}'
)

_NUGGET_SCORE_SYSTEM = (
    "You compare a candidate answer against a list of required facts (nuggets). "
    "For each nugget, output exactly one label:\n"
    "  covered      - the answer states this fact (any phrasing).\n"
    "  contradicted - the answer states something that conflicts with this fact.\n"
    "  absent       - the answer neither states nor contradicts it.\n"
    "Extra correct information in the answer is fine and must NOT be penalized. "
    'Return strict JSON: {"labels": ["covered", "absent", ...]} in nugget order.'
)

_RUBRIC_SYSTEM = (
    "You are a strict grader. Given a question, a reference (gold) answer, the "
    "gold context, and a candidate answer, rate how correctly and completely the "
    "candidate answers the question. Credit valid paraphrase and extra correct "
    "facts; penalize only wrong, missing, or contradictory content.\n"
    "Anchors: 1.0 fully correct & complete; 0.7 correct, minor gap; 0.4 partially "
    "correct; 0.0 wrong or contradicts the gold.\n"
    'Return strict JSON: {"score": <float 0-1>, "reason": "<one sentence>"}'
)


async def extract_nuggets(gold: str, gateway) -> list[str]:
    raw, _ = await gateway.json_response(
        system_prompt=_NUGGET_EXTRACT_SYSTEM,
        user_prompt=gold,
        task="eval_judge",
    )
    return parse_nuggets(raw)


async def score_nuggets(question: str, answer: str, nuggets: list[str], gateway) -> NuggetScore:
    if not nuggets:
        return NuggetScore(0, 0, 0, 0.0, 0.0, 0.0)
    user = json.dumps(
        {"question": question, "answer": answer, "nuggets": nuggets},
        ensure_ascii=False,
    )
    raw, _ = await gateway.json_response(
        system_prompt=_NUGGET_SCORE_SYSTEM, user_prompt=user, task="eval_judge"
    )
    labels = [l for l in (raw.get("labels") or []) if isinstance(l, str)]
    return aggregate_nugget_labels(labels)


async def score_rubric(question: str, answer: str, gold: str, gold_context: str, gateway) -> float:
    user = json.dumps(
        {"question": question, "gold_answer": gold, "gold_context": gold_context[:4000], "answer": answer},
        ensure_ascii=False,
    )
    raw, _ = await gateway.json_response(
        system_prompt=_RUBRIC_SYSTEM, user_prompt=user, task="eval_judge"
    )
    return clamp01(raw.get("score") or 0.0)


async def score_correctness(question: str, answer: str, gold: str, gold_context: str, gateway) -> EnsembleScore:
    nuggets = await extract_nuggets(gold, gateway)
    ns = await score_nuggets(question, answer, nuggets, gateway)
    rubric = await score_rubric(question, answer, gold, gold_context, gateway)
    return ensemble(ns.score, rubric)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_correctness_judge_async.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/eval/correctness_judge.py tests/eval/test_correctness_judge_async.py
git commit -m "feat(eval): async ensemble correctness judge (nugget + rubric)"
```

---

### Task 3: Oracle probe (generator + aggregation + script)

**Files:**
- Create: `scripts/eval/oracle_probe.py`
- Test: `tests/eval/test_oracle_probe.py`

**Interfaces:**
- Consumes: `score_correctness`, `EnsembleScore` from Tasks 1-2; `load_suite` from `src.agentrag.eval.benchmark_datasets`.
- Produces: `async generate_oracle_answer(question: str, gold_context: str, gateway) -> str`; `ProbeRow` dataclass (`qid, system_mean, oracle_mean, judge2_mean`); `summarize_probe(rows: list[ProbeRow]) -> dict` (keys: `n, system_avg, oracle_avg, oracle_minus_system, judge_noise_pearson`); `pearson(xs, ys) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_oracle_probe.py
import pytest

from scripts.eval.oracle_probe import (
    generate_oracle_answer, summarize_probe, pearson, ProbeRow,
)


class FakeTextGateway:
    def __init__(self, text):
        self._text = text
        self.calls = []

    async def text_response(self, system_prompt, user_prompt, task="general"):
        self.calls.append({"system": system_prompt, "user": user_prompt})
        return self._text


@pytest.mark.asyncio
async def test_generate_oracle_answer_uses_gold_context():
    gw = FakeTextGateway("oracle answer")
    out = await generate_oracle_answer("what is X?", "X is defined here.", gw)
    assert out == "oracle answer"
    assert "X is defined here." in gw.calls[0]["user"]


def test_pearson_perfect_correlation():
    assert round(pearson([1, 2, 3], [2, 4, 6]), 6) == 1.0


def test_pearson_degenerate_returns_zero():
    # zero variance → defined as 0.0, not a crash
    assert pearson([1, 1, 1], [2, 3, 4]) == 0.0


def test_summarize_probe_reports_gap_and_noise():
    rows = [
        ProbeRow("q1", system_mean=0.70, oracle_mean=0.74, judge2_mean=0.69),
        ProbeRow("q2", system_mean=0.80, oracle_mean=0.82, judge2_mean=0.81),
    ]
    s = summarize_probe(rows)
    assert s["n"] == 2
    assert s["system_avg"] == 0.75
    assert s["oracle_avg"] == 0.78
    assert abs(s["oracle_minus_system"] - 0.03) < 1e-9
    assert s["judge_noise_pearson"] == 1.0  # system vs judge2 perfectly correlated here
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_oracle_probe.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.eval.oracle_probe`.

- [ ] **Step 3: Write the probe module**

```python
#!/usr/bin/env python
"""Phase-1 oracle probe — prove the correctness cap is the ruler, not the system.

For each vn example: score the live system answer AND an oracle answer
(strong model + GOLD context = perfect retrieval) against gold, through the new
ensemble judge. Also re-score system answers with a SECOND judge model to get the
judge-noise floor. If oracle ≈ system, the cap is the gold/metric.

Run (no ingest — vn suite must already be indexed):
    uv run python scripts/eval/oracle_probe.py --n 20 --out docs/eval/eval_fidelity_probe_2026-06-26.md
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentrag.eval.benchmark_datasets import load_suite
from src.agentrag.eval.correctness_judge import score_correctness


_ORACLE_SYSTEM = (
    "Answer the question using ONLY the provided context. Be complete and precise. "
    "Do not add facts beyond the context. Answer in the context's language."
)


async def generate_oracle_answer(question: str, gold_context: str, gateway) -> str:
    """Strong-model answer given gold context — the achievable correctness ceiling."""
    user = f"CONTEXT:\n{gold_context}\n\nQUESTION: {question}"
    return await gateway.text_response(
        system_prompt=_ORACLE_SYSTEM, user_prompt=user, task="eval_judge"
    )


@dataclass
class ProbeRow:
    qid: str
    system_mean: float
    oracle_mean: float
    judge2_mean: float


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def summarize_probe(rows: list[ProbeRow]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "system_avg": 0.0, "oracle_avg": 0.0,
                "oracle_minus_system": 0.0, "judge_noise_pearson": 0.0}
    sys_avg = sum(r.system_mean for r in rows) / n
    ora_avg = sum(r.oracle_mean for r in rows) / n
    noise = pearson([r.system_mean for r in rows], [r.judge2_mean for r in rows])
    return {
        "n": n,
        "system_avg": sys_avg,
        "oracle_avg": ora_avg,
        "oracle_minus_system": ora_avg - sys_avg,
        "judge_noise_pearson": noise,
    }


def _render(summary: dict, suite: str, n: int) -> str:
    return (
        f"# Eval-fidelity probe — {suite} n={n}\n\n"
        f"- examples: {summary['n']}\n"
        f"- system avg (ensemble): {summary['system_avg']:.3f}\n"
        f"- oracle avg (strong model + gold context): {summary['oracle_avg']:.3f}\n"
        f"- **oracle − system: {summary['oracle_minus_system']:+.3f}**\n"
        f"- judge-noise floor (pearson, judge1 vs judge2): {summary['judge_noise_pearson']:.3f}\n\n"
        "## Read\n\n"
        "If oracle − system is small (< ~0.05), perfect retrieval + a strong generator "
        "barely beats the live system — the cap is the gold/metric, not the system. "
        "A low judge-noise pearson means the judge itself is unreliable and must be fixed "
        "before any correctness number is trusted.\n"
    )


async def main(args: argparse.Namespace) -> None:
    from src.agentrag.services.agent_service import get_agent_service  # live system answers
    from src.agentrag.services.llm_gateway import LLMGateway

    gateway = LLMGateway()
    agent = get_agent_service()
    examples = load_suite(args.suite, n=args.n)
    print(f"[probe] {len(examples)} examples (suite={args.suite}, n={args.n})")

    rows: list[ProbeRow] = []
    for i, ex in enumerate(examples):
        gold_ctx = "\n".join(ex.gold_contexts)
        out = await agent.chat(question=ex.question, document_title=None, conversation_id=f"probe-{ex.id}")
        system_ans = out.get("answer", "") or ""
        oracle_ans = await generate_oracle_answer(ex.question, gold_ctx, gateway)

        sys_e = await score_correctness(ex.question, system_ans, ex.reference_answer, gold_ctx, gateway)
        ora_e = await score_correctness(ex.question, oracle_ans, ex.reference_answer, gold_ctx, gateway)
        # judge2: re-score the SAME system answer; the gateway routes eval_judge to the
        # configured judge — swap via LLM_TASK_MODEL_MAP.eval_judge2 env for the 2nd model.
        j2_e = await score_correctness(ex.question, system_ans, ex.reference_answer, gold_ctx, gateway)

        rows.append(ProbeRow(ex.id, sys_e.mean, ora_e.mean, j2_e.mean))
        print(f"  [{i+1}/{len(examples)}] sys={sys_e.mean:.2f} oracle={ora_e.mean:.2f}")

    summary = summarize_probe(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(summary, args.suite, args.n), encoding="utf-8")
    print(f"[probe] wrote {out_path}")
    print(f"[probe] oracle−system = {summary['oracle_minus_system']:+.3f}, "
          f"judge-noise = {summary['judge_noise_pearson']:.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="vn", choices=["vn", "en", "both"])
    p.add_argument("--n", type=int, default=20, help="examples per dataset")
    p.add_argument("--out", default="docs/eval/eval_fidelity_probe.md")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_oracle_probe.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the full eval test module + commit**

Run: `uv run pytest tests/eval/test_correctness_judge.py tests/eval/test_correctness_judge_async.py tests/eval/test_oracle_probe.py -v`
Expected: PASS (all).

```bash
git add scripts/eval/oracle_probe.py tests/eval/test_oracle_probe.py
git commit -m "feat(eval): oracle ceiling probe — prove correctness cap is the ruler"
```

---

### Task 4: Run the probe + write the verdict (manual, no test)

**Files:**
- Create: `docs/eval/eval_fidelity_probe_2026-06-26.md` (script output)

- [ ] **Step 1: Confirm the vn suite is already indexed** (probe does not ingest). If not, run one `run_benchmark --suite vn --n 20` first, or ingest the vn gold separately.

- [ ] **Step 2: Run the probe**

Run: `uv run python scripts/eval/oracle_probe.py --n 20 --out docs/eval/eval_fidelity_probe_2026-06-26.md`
Expected: prints per-example `sys=/oracle=` lines, then `oracle−system` and `judge-noise`.

- [ ] **Step 3: Record the gate decision in the doc.** Append a "Decision" section: if `oracle − system < ~0.05` → cap confirmed as the ruler → **proceed to Phase 2** (prod-corpus build). If the gap is large → there is real system headroom; revisit before building the prod set.

- [ ] **Step 4: Commit**

```bash
git add docs/eval/eval_fidelity_probe_2026-06-26.md
git commit -m "eval: phase-1 oracle probe results + phase-2 gate decision"
```

---

## Self-Review

**Spec coverage:**
- Spec §2 ensemble judge (nugget + rubric + Δ-flag) → Tasks 1-2. ✓
- Spec §3 oracle ceiling + judge-noise floor → Tasks 3-4. ✓
- Spec §4-5 prod-corpus build + calibration → **deferred to the Phase 2 plan** (gated on Task 4 outcome, per spec §8). Documented, not a gap.
- Spec "fix stale `src.pam.*` in generate_dataset.py" → belongs to Phase 2 (gold-answer generation reuses `generate_golden_dataset`); not needed for Phase 1.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; expected outputs given. ✓

**Type consistency:** `score_correctness(question, answer, gold, gold_context, gateway) -> EnsembleScore` used identically in Task 2 and Task 3. `EnsembleScore.mean` consumed in `ProbeRow`. `NuggetScore` field order matches the `==` assertion in Task 1. ✓

**Note on judge2 (Task 3 main):** the second-judge routing is an env/config swap (`LLM_TASK_MODEL_MAP`), not new code — the probe calls `score_correctness` again; wire the 2nd model via env at run time. If per-call model selection is later wanted, that is a small follow-up, out of scope here.
