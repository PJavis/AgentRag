# Miss-Bucketing + CRAG A/B + Citation Flywheel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the measured +0.088 real-corpus headroom (concentrated in ~5/40 misses) into actionable, fixed failure classes: per-row probe capture → automatic miss bucketing → CRAG on/off A/B → citation-reward reranker training data.

**Architecture:** Three offline-testable code tasks extend the existing eval rig (`scripts/eval/oracle_probe.py` gains a `--rows-out` per-question JSONL dump; two new pure modules under `src/agentrag/eval/` classify misses and mine reranker triplets from those rows), followed by one live-run task that produces the evidence docs and the CRAG enable/keep-off decision. The CRAG loop itself already exists in `graph_service.py` (nodes `critique` → `corrective_retrieve`, default-OFF flag `CRAG_ENABLED`) — this plan measures it, it does not build it.

**Tech Stack:** Python 3.12 + uv, pytest + pytest-asyncio, existing `LLMGateway`/`GraphAgentService`/ensemble `correctness_judge`. No new dependencies.

## Global Constraints

- Run everything with `uv run …` from repo root `/home/nguyenquocdung/AgentRag` (imports are rooted at `src.agentrag.*` / `scripts.eval.*`).
- New analysis logic goes in `src/agentrag/eval/` as pure functions (no I/O, no LLM calls) so tests run without services; CLI wrappers in `scripts/eval/` stay thin.
- Do NOT change retrieval/agent behavior in Tasks 1–3. The only production-code file touched is `scripts/eval/oracle_probe.py`, and only additively (existing CLI flags and output format keep working).
- Eval-set validity rule (docs/HOME-RUN.md): an eval set is only valid against the corpus snapshot it was generated from. `data/eval/prod_corpus_evalset*.jsonl` are 2026-06 residue-corpus sets — INVALID against the real corpus. Use `data/eval/c2_evalset_n40.jsonl` (gitignored, may need rebuild via `scripts/eval/build_prod_evalset.py`).
- Live-run env (Task 4): independent-judge map per docs/HOME-RUN.md — `eval_judge=gemini-2.5-pro` (paid key), `eval_judge2=deepseek-v4-pro`, `oracle_gen`/`gold_gen`=deepseek-v4-pro; `RETRIEVAL_RERANK_BACKEND=local_cross_encoder`, `RETRIEVAL_RELEVANCE_FLOOR=0.55`.
- Commit after each green test cycle. Conventional Commits, subject ≤ 50 chars.

**Row schema (produced by Task 1, consumed by Tasks 2–4).** One JSON object per line in the `--rows-out` file:

```json
{
  "qid": "c2-17",
  "question": "…",
  "system_answer": "… [1] … [3]",
  "oracle_answer": "…",
  "system_mean": 0.25,
  "oracle_mean": 1.0,
  "judge2_mean": 0.5,
  "refusal_class": "hallucinated",
  "cited_sources": [1, 3],
  "packed": [{"content": "…", "rerank_score": 0.71, "document_title": "…", "section_path": "…"}],
  "gold_contexts": ["…"],
  "tool_queries": ["query used by hop 1", "…"],
  "citations_count": 8
}
```

`refusal_class` comes from `src.agentrag.eval.refusal.classify_refusal` (values: `abstained` / `hedged_cited` / `hallucinated` / `empty`). `cited_sources` are the inline `[n]` markers parsed from `system_answer`. `packed` is the agent's returned `context` list reduced to the four fields shown.

---

### Task 1: Per-row capture in oracle_probe (`--rows-out`)

**Files:**
- Create: `src/agentrag/eval/probe_rows.py`
- Modify: `scripts/eval/oracle_probe.py`
- Test: `tests/eval/test_probe_rows.py`

**Interfaces:**
- Consumes: `agent.chat()` output dict (keys `answer`, `context`, `citations`, `tool_trace`), `EnsembleScore.mean`, `classify_refusal(answer, citations)`.
- Produces: `parse_inline_citations(answer: str) -> list[int]` (sorted, deduped); `build_probe_row(qid, question, chat_out, oracle_answer, system_mean, oracle_mean, judge2_mean, gold_contexts) -> dict` (the Row schema above); `ProbeRow.detail: dict | None` field; `oracle_probe.py --rows-out <path>` writing one Row per line.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_probe_rows.py
from src.agentrag.eval.probe_rows import build_probe_row, parse_inline_citations


def test_parse_inline_citations_dedup_sorted():
    ans = "Liều dùng là 5mg [2]. Chống chỉ định suy thận [1][2]."
    assert parse_inline_citations(ans) == [1, 2]


def test_parse_inline_citations_ignores_links_and_empty():
    assert parse_inline_citations("xem [tài liệu](http://x) nhé") == []
    assert parse_inline_citations("") == []
    assert parse_inline_citations(None) == []


def _chat_out():
    return {
        "answer": "Đáp án đúng [1].",
        "context": [
            {"content": "gold text here", "rerank_score": 0.71,
             "document_title": "doc.pdf", "section_path": "1>2", "extra": "dropped"},
            {"content": "distractor", "rerank_score": 0.58,
             "document_title": "doc.pdf", "section_path": "3"},
        ],
        "citations": [{"source": 1}, {"source": 2}],
        "tool_trace": [
            {"tool_name": "search_hybrid_kg", "tool_input": {"query": "q-hop-1"}},
            {"tool_name": "search_hybrid_kg", "tool_input": {"query": "q-hop-2"}},
        ],
    }


def test_build_probe_row_shape():
    row = build_probe_row(
        qid="c2-1", question="q?", chat_out=_chat_out(), oracle_answer="oracle",
        system_mean=0.9, oracle_mean=1.0, judge2_mean=0.85,
        gold_contexts=["gold text here"],
    )
    assert row["qid"] == "c2-1"
    assert row["cited_sources"] == [1]
    assert row["refusal_class"] == "hallucinated"  # confident answer, per classify_refusal
    assert row["tool_queries"] == ["q-hop-1", "q-hop-2"]
    assert row["citations_count"] == 2
    assert row["packed"] == [
        {"content": "gold text here", "rerank_score": 0.71,
         "document_title": "doc.pdf", "section_path": "1>2"},
        {"content": "distractor", "rerank_score": 0.58,
         "document_title": "doc.pdf", "section_path": "3"},
    ]


def test_build_probe_row_abstention():
    out = _chat_out()
    out["answer"] = "Tài liệu hiện có không có thông tin để trả lời câu hỏi này."
    out["citations"] = []
    row = build_probe_row(
        qid="c2-2", question="q?", chat_out=out, oracle_answer="o",
        system_mean=0.0, oracle_mean=1.0, judge2_mean=0.0,
        gold_contexts=["g"],
    )
    assert row["refusal_class"] == "abstained"
    assert row["cited_sources"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_probe_rows.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agentrag.eval.probe_rows'`

- [ ] **Step 3: Implement `src/agentrag/eval/probe_rows.py`**

```python
"""Per-question probe rows — the raw material for miss bucketing and
citation mining. Pure functions only: no I/O, no LLM calls."""
from __future__ import annotations

import re
from typing import Any

from src.agentrag.eval.refusal import classify_refusal

# Inline citation markers the answer prompt mandates: "… [1]." / "… [1][2]."
# \[(\d{1,2})\] deliberately excludes markdown links ([text](url) has no digit-only body).
_CITE_RE = re.compile(r"\[(\d{1,2})\]")

_PACKED_FIELDS = ("content", "rerank_score", "document_title", "section_path")


def parse_inline_citations(answer: str | None) -> list[int]:
    """Source numbers the answer actually cited — the RMM 'used it' signal."""
    if not answer:
        return []
    return sorted({int(m) for m in _CITE_RE.findall(answer)})


def build_probe_row(
    *,
    qid: str,
    question: str,
    chat_out: dict[str, Any],
    oracle_answer: str,
    system_mean: float,
    oracle_mean: float,
    judge2_mean: float,
    gold_contexts: list[str],
) -> dict[str, Any]:
    answer = chat_out.get("answer") or ""
    citations = chat_out.get("citations") or []
    packed = [
        {k: item.get(k) for k in _PACKED_FIELDS}
        for item in (chat_out.get("context") or [])
    ]
    tool_queries = [
        (step.get("tool_input") or {}).get("query")
        for step in (chat_out.get("tool_trace") or [])
        if (step.get("tool_input") or {}).get("query")
    ]
    return {
        "qid": qid,
        "question": question,
        "system_answer": answer,
        "oracle_answer": oracle_answer,
        "system_mean": system_mean,
        "oracle_mean": oracle_mean,
        "judge2_mean": judge2_mean,
        "refusal_class": classify_refusal(answer, citations),
        "cited_sources": parse_inline_citations(answer),
        "packed": packed,
        "gold_contexts": list(gold_contexts),
        "tool_queries": tool_queries,
        "citations_count": len(citations),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_probe_rows.py -v`
Expected: 4 PASS

- [ ] **Step 5: Wire `--rows-out` into `scripts/eval/oracle_probe.py`**

Three edits, all additive:

(a) Extend the dataclass (keep field order — existing tests construct `ProbeRow("q1", system_mean=…)`):

```python
@dataclass
class ProbeRow:
    qid: str
    system_mean: float
    oracle_mean: float
    judge2_mean: float
    detail: dict | None = None
```

(b) In `_score_one` (inside `main`), capture the detail row. Replace the final `return ProbeRow(...)` and add the import at the top of the file with the other `src.agentrag` imports:

```python
from src.agentrag.eval.probe_rows import build_probe_row
```

```python
        detail = build_probe_row(
            qid=ex.id, question=ex.question, chat_out=out,
            oracle_answer=oracle_ans,
            system_mean=sys_e.mean, oracle_mean=ora_e.mean, judge2_mean=j2_e.mean,
            gold_contexts=ex.gold_contexts,
        )
        return ProbeRow(ex.id, sys_e.mean, ora_e.mean, j2_e.mean, detail)
```

(c) After the summary is written in `main` (right after `print(f"[probe] wrote {out_path}")`), dump rows; and register the flag in `parse_args`:

```python
    if args.rows_out:
        import json
        rows_path = Path(args.rows_out)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        with rows_path.open("w", encoding="utf-8") as f:
            for r in rows:
                if r.detail:
                    f.write(json.dumps(r.detail, ensure_ascii=False) + "\n")
        print(f"[probe] wrote {len([r for r in rows if r.detail])} rows → {rows_path}")
```

```python
    p.add_argument("--rows-out", default=None,
                   help="ALSO dump one JSON row per scored question (miss bucketing / citation mining input)")
```

- [ ] **Step 6: Run the full eval test suite (regression check)**

Run: `uv run pytest tests/eval/ -v`
Expected: all PASS, including the untouched `tests/eval/test_oracle_probe.py` (ProbeRow gained only a defaulted field)

- [ ] **Step 7: Commit**

```bash
git add src/agentrag/eval/probe_rows.py scripts/eval/oracle_probe.py tests/eval/test_probe_rows.py
git commit -m "feat(eval): per-row probe capture via --rows-out"
```

---

### Task 2: Miss bucketing + report CLI

**Files:**
- Create: `src/agentrag/eval/miss_buckets.py`
- Create: `scripts/eval/report_miss_buckets.py`
- Test: `tests/eval/test_miss_buckets.py`

**Interfaces:**
- Consumes: Row dicts (Task 1 schema).
- Produces: `gold_overlap(packed: list[dict], gold_contexts: list[str]) -> float`; `bucket_row(row: dict, *, miss_threshold: float = 0.5, overlap_threshold: float = 0.35) -> str | None` returning one of `"false_abstention" | "retrieval_miss" | "generation_miss" | None`; `summarize_buckets(rows: list[dict]) -> dict`; `render_report(rows: list[dict], summary: dict, label: str) -> str` (markdown). CLI: `uv run python scripts/eval/report_miss_buckets.py --rows <in.jsonl> --out <report.md> --label <name>`.

Bucket semantics (the decision each bucket drives):
- `false_abstention` — judged a miss AND `refusal_class` is `abstained`/`hedged_cited`/`empty`: the system refused an answerable question → floor/gate tuning territory.
- `retrieval_miss` — answered, but no packed passage overlaps any gold context (`gold_overlap < overlap_threshold`): the right chunk never reached the answer LLM → retrieval/graph work (HippoRAG-2 gate evidence).
- `generation_miss` — answered, gold context WAS in the packed list, still scored low → answer-prompt/model work, or judge disagreement (flagged via `judge_gap`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_miss_buckets.py
from src.agentrag.eval.miss_buckets import (
    bucket_row, gold_overlap, render_report, summarize_buckets,
)


def _row(**over):
    row = {
        "qid": "c2-1", "question": "q?",
        "system_answer": "Trả lời chắc chắn [1].",
        "system_mean": 0.0, "oracle_mean": 1.0, "judge2_mean": 0.0,
        "refusal_class": "hallucinated",
        "cited_sources": [1],
        "packed": [{"content": "hoàn toàn khác biệt nội dung", "rerank_score": 0.6,
                    "document_title": "d", "section_path": "s"}],
        "gold_contexts": ["thuốc metformin liều 500mg ngày hai lần"],
        "tool_queries": ["q"], "citations_count": 1,
    }
    row.update(over)
    return row


def test_gold_overlap_high_when_gold_packed():
    packed = [{"content": "thuốc metformin liều 500mg ngày hai lần cho bệnh nhân"}]
    assert gold_overlap(packed, ["thuốc metformin liều 500mg ngày hai lần"]) > 0.5


def test_gold_overlap_zero_when_disjoint():
    assert gold_overlap([{"content": "abc def"}], ["xyz uvw"]) == 0.0


def test_gold_overlap_empty_packed():
    assert gold_overlap([], ["gold"]) == 0.0


def test_bucket_not_a_miss():
    assert bucket_row(_row(system_mean=0.9)) is None


def test_bucket_false_abstention():
    row = _row(refusal_class="abstained", cited_sources=[], citations_count=0)
    assert bucket_row(row) == "false_abstention"


def test_bucket_retrieval_miss():
    assert bucket_row(_row()) == "retrieval_miss"  # packed disjoint from gold


def test_bucket_generation_miss():
    row = _row(packed=[{"content": "thuốc metformin liều 500mg ngày hai lần"}])
    assert bucket_row(row) == "generation_miss"


def test_summarize_counts_and_judge_gap():
    rows = [
        _row(),                                                     # retrieval_miss
        _row(qid="c2-2", refusal_class="abstained", cited_sources=[]),  # false_abstention
        _row(qid="c2-3", system_mean=0.9),                          # not a miss
        _row(qid="c2-4", system_mean=0.3, judge2_mean=0.8),         # judge_gap flag
    ]
    s = summarize_buckets(rows)
    assert s["n"] == 4
    assert s["misses"] == 3
    assert s["buckets"]["retrieval_miss"] == 2
    assert s["buckets"]["false_abstention"] == 1
    assert s["judge_gap_rows"] == ["c2-4"]


def test_render_report_contains_buckets_and_rows():
    rows = [_row()]
    md = render_report(rows, summarize_buckets(rows), label="test-set")
    assert "retrieval_miss" in md
    assert "c2-1" in md
    assert md.startswith("# Miss buckets — test-set")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_miss_buckets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agentrag.eval.miss_buckets'`

- [ ] **Step 3: Implement `src/agentrag/eval/miss_buckets.py`**

```python
"""Classify probe-row misses into actionable failure buckets.

Buckets:
    false_abstention — refused an answerable question (floor/gate territory)
    retrieval_miss   — gold passage never reached the answer LLM (retrieval/graph work)
    generation_miss  — gold was packed, answer still wrong (prompt/model work)

Pure functions over Task-1 probe rows; no I/O."""
from __future__ import annotations

from typing import Any

_REFUSED = ("abstained", "hedged_cited", "empty")
JUDGE_GAP = 0.4  # |system_mean - judge2_mean| at/above this → judge-disagreement flag


def _words(text: str) -> set[str]:
    return set((text or "").lower().split())


def gold_overlap(packed: list[dict[str, Any]], gold_contexts: list[str]) -> float:
    """Best Jaccard word overlap between any packed passage and any gold context.
    Proxy for 'did the gold chunk reach the answer LLM' — exact hashes are not
    available in probe rows (the eval set stores raw text, not segment ids)."""
    best = 0.0
    for item in packed or []:
        pw = _words(item.get("content") or "")
        if not pw:
            continue
        for gold in gold_contexts or []:
            gw = _words(gold)
            if not gw:
                continue
            inter = len(pw & gw)
            union = len(pw | gw)
            if union:
                best = max(best, inter / union)
    return best


def bucket_row(
    row: dict[str, Any],
    *,
    miss_threshold: float = 0.5,
    overlap_threshold: float = 0.35,
) -> str | None:
    if float(row.get("system_mean", 0.0)) >= miss_threshold:
        return None
    if row.get("refusal_class") in _REFUSED:
        return "false_abstention"
    if gold_overlap(row.get("packed") or [], row.get("gold_contexts") or []) < overlap_threshold:
        return "retrieval_miss"
    return "generation_miss"


def summarize_buckets(rows: list[dict[str, Any]], **kw) -> dict[str, Any]:
    buckets: dict[str, int] = {}
    misses = 0
    judge_gap_rows: list[str] = []
    for row in rows:
        b = bucket_row(row, **kw)
        if b:
            misses += 1
            buckets[b] = buckets.get(b, 0) + 1
        if abs(float(row.get("system_mean", 0.0)) - float(row.get("judge2_mean", 0.0))) >= JUDGE_GAP:
            judge_gap_rows.append(row.get("qid", "?"))
    return {"n": len(rows), "misses": misses, "buckets": buckets,
            "judge_gap_rows": judge_gap_rows}


def render_report(rows: list[dict[str, Any]], summary: dict[str, Any], label: str) -> str:
    lines = [
        f"# Miss buckets — {label}",
        "",
        f"- rows scored: {summary['n']}",
        f"- misses (system_mean < 0.5): {summary['misses']}",
    ]
    for name, count in sorted(summary["buckets"].items()):
        lines.append(f"- **{name}**: {count}")
    if summary["judge_gap_rows"]:
        lines.append(f"- judge-disagreement rows (|sys−judge2| ≥ {JUDGE_GAP}): "
                     + ", ".join(summary["judge_gap_rows"]))
    lines += ["", "## Miss detail", ""]
    for row in rows:
        b = bucket_row(row)
        if not b:
            continue
        best = gold_overlap(row.get("packed") or [], row.get("gold_contexts") or [])
        scores = [c.get("rerank_score") for c in (row.get("packed") or [])
                  if c.get("rerank_score") is not None]
        lines += [
            f"### {row['qid']} — `{b}`",
            "",
            f"- Q: {row.get('question', '')[:200]}",
            f"- sys={row.get('system_mean')} oracle={row.get('oracle_mean')} "
            f"judge2={row.get('judge2_mean')} refusal={row.get('refusal_class')}",
            f"- gold_overlap={best:.2f} max_rerank={max(scores) if scores else None} "
            f"cited={row.get('cited_sources')}",
            f"- tool_queries: {row.get('tool_queries')}",
            f"- answer: {row.get('system_answer', '')[:300]}",
            "",
        ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_miss_buckets.py -v`
Expected: 9 PASS

- [ ] **Step 5: Write the CLI wrapper `scripts/eval/report_miss_buckets.py`**

```python
#!/usr/bin/env python
"""Render a miss-bucket report from an oracle_probe --rows-out JSONL.

Run:
    uv run python scripts/eval/report_miss_buckets.py \
        --rows docs/eval/rows_c2_n40.jsonl \
        --out docs/eval/miss_buckets_2026-07-14.md --label c2_evalset_n40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentrag.eval.miss_buckets import render_report, summarize_buckets


def main(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.rows).read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = summarize_buckets(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(rows, summary, args.label), encoding="utf-8")
    print(f"[buckets] {summary['misses']}/{summary['n']} misses → {dict(summary['buckets'])}")
    print(f"[buckets] wrote {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="eval-set")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
```

- [ ] **Step 6: Smoke-test the CLI on a synthetic row**

Run:
```bash
printf '%s\n' '{"qid":"x1","question":"q","system_answer":"a [1]","system_mean":0.0,"oracle_mean":1.0,"judge2_mean":0.0,"refusal_class":"hallucinated","cited_sources":[1],"packed":[{"content":"foo bar"}],"gold_contexts":["baz qux"],"tool_queries":[],"citations_count":1}' > /tmp/claude-1000/-home-nguyenquocdung-AgentRag/86c5efd7-ece5-4cb0-aca7-1084de0ac44f/scratchpad/rows_smoke.jsonl
uv run python scripts/eval/report_miss_buckets.py --rows /tmp/claude-1000/-home-nguyenquocdung-AgentRag/86c5efd7-ece5-4cb0-aca7-1084de0ac44f/scratchpad/rows_smoke.jsonl --out /tmp/claude-1000/-home-nguyenquocdung-AgentRag/86c5efd7-ece5-4cb0-aca7-1084de0ac44f/scratchpad/smoke_report.md --label smoke
```
Expected stdout: `[buckets] 1/1 misses → {'retrieval_miss': 1}`

- [ ] **Step 7: Commit**

```bash
git add src/agentrag/eval/miss_buckets.py scripts/eval/report_miss_buckets.py tests/eval/test_miss_buckets.py
git commit -m "feat(eval): miss-bucket classifier + report CLI"
```

---

### Task 3: Citation-reward triplet mining (RMM flywheel)

**Files:**
- Create: `src/agentrag/eval/citation_mining.py`
- Create: `scripts/eval/mine_citation_pairs.py`
- Test: `tests/eval/test_citation_mining.py`

**Interfaces:**
- Consumes: Row dicts (Task 1 schema); `parse_inline_citations` already applied (`cited_sources` field).
- Produces: `mine_triplets(rows: list[dict], *, min_system_mean: float = 0.75) -> list[dict]` where each triplet is `{"query": str, "positive": str, "negative": str, "source": "citation"}` — the exact shape `scripts/finetune_reranker.py` and `scripts/finetune_embedding.py` consume (`query`/`positive`/`negative` keys; extra keys ignored). CLI: `uv run python scripts/eval/mine_citation_pairs.py --rows <in.jsonl> --out data/finetune/citation_pairs.jsonl`.

Mining rules (RMM: retrieved-and-cited = +1, retrieved-but-uncited = −1):
- Only rows with `system_mean ≥ min_system_mean` (positives from wrong answers are noise).
- Positives: packed items whose 1-based position is in `cited_sources`.
- Negatives: packed items NOT cited, hardest first (highest `rerank_score`).
- Each positive pairs with one negative, cycling through negatives; rows with no positive or no negative yield nothing.
- Out-of-range `[n]` markers (n > len(packed)) are ignored.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_citation_mining.py
from src.agentrag.eval.citation_mining import mine_triplets


def _row(**over):
    row = {
        "qid": "c2-1", "question": "liều metformin?",
        "system_answer": "500mg [1].",
        "system_mean": 0.9,
        "refusal_class": "hallucinated",
        "cited_sources": [1],
        "packed": [
            {"content": "metformin 500mg", "rerank_score": 0.72},
            {"content": "insulin liều", "rerank_score": 0.66},
            {"content": "paracetamol", "rerank_score": 0.58},
        ],
    }
    row.update(over)
    return row


def test_mines_cited_vs_hardest_uncited():
    trips = mine_triplets([_row()])
    assert trips == [{
        "query": "liều metformin?",
        "positive": "metformin 500mg",
        "negative": "insulin liều",   # hardest uncited (0.66 > 0.58)
        "source": "citation",
    }]


def test_skips_low_score_rows():
    assert mine_triplets([_row(system_mean=0.4)]) == []


def test_skips_rows_without_negatives():
    row = _row(cited_sources=[1, 2, 3])  # everything cited → no negative
    assert mine_triplets([row]) == []


def test_multiple_positives_cycle_negatives():
    row = _row(cited_sources=[1, 2])
    trips = mine_triplets([row])
    assert len(trips) == 2
    assert {t["positive"] for t in trips} == {"metformin 500mg", "insulin liều"}
    assert all(t["negative"] == "paracetamol" for t in trips)


def test_ignores_out_of_range_citation():
    row = _row(cited_sources=[1, 9])
    trips = mine_triplets([row])
    assert len(trips) == 1
    assert trips[0]["positive"] == "metformin 500mg"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_citation_mining.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agentrag.eval.citation_mining'`

- [ ] **Step 3: Implement `src/agentrag/eval/citation_mining.py`**

```python
"""RMM-style citation-reward mining: the answer LLM's own inline [n] citations
label the rerank pool — cited passage = positive, retrieved-but-uncited =
hard negative. Output triplets feed scripts/finetune_reranker.py /
finetune_embedding.py unchanged. Pure functions; no I/O."""
from __future__ import annotations

from typing import Any


def mine_triplets(
    rows: list[dict[str, Any]],
    *,
    min_system_mean: float = 0.75,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if float(row.get("system_mean", 0.0)) < min_system_mean:
            continue
        packed = row.get("packed") or []
        cited = {n for n in (row.get("cited_sources") or []) if 1 <= n <= len(packed)}
        if not cited:
            continue
        positives = [packed[n - 1].get("content") or "" for n in sorted(cited)]
        negatives = sorted(
            (item for i, item in enumerate(packed, start=1) if i not in cited),
            key=lambda c: float(c.get("rerank_score") or 0.0),
            reverse=True,
        )
        negatives = [c.get("content") or "" for c in negatives if c.get("content")]
        if not negatives:
            continue
        query = row.get("question") or ""
        for i, pos in enumerate(p for p in positives if p):
            out.append({
                "query": query,
                "positive": pos,
                "negative": negatives[i % len(negatives)],
                "source": "citation",
            })
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_citation_mining.py -v`
Expected: 5 PASS

- [ ] **Step 5: Write the CLI wrapper `scripts/eval/mine_citation_pairs.py`**

```python
#!/usr/bin/env python
"""Mine reranker/embedding triplets from oracle_probe --rows-out JSONL.

Cited-in-answer passages become positives; retrieved-but-uncited become hard
negatives (RMM citation reward). Appends cleanly to the same training file
format as scripts/mine_finetune_pairs.py.

Run:
    uv run python scripts/eval/mine_citation_pairs.py \
        --rows docs/eval/rows_c2_n40.jsonl \
        --out data/finetune/citation_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentrag.eval.citation_mining import mine_triplets


def main(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in Path(args.rows).read_text(encoding="utf-8").splitlines() if line.strip()]
    trips = mine_triplets(rows, min_system_mean=args.min_score)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a" if args.append else "w", encoding="utf-8") as f:
        for t in trips:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[mine] {len(trips)} triplets from {len(rows)} rows → {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", required=True)
    p.add_argument("--out", default="data/finetune/citation_pairs.jsonl")
    p.add_argument("--min-score", type=float, default=0.75,
                   help="mine only rows with system_mean >= this (trustworthy positives)")
    p.add_argument("--append", action="store_true",
                   help="append to --out instead of overwriting (accumulate across runs)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
```

- [ ] **Step 6: Run the full eval suite once more**

Run: `uv run pytest tests/eval/ -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/agentrag/eval/citation_mining.py scripts/eval/mine_citation_pairs.py tests/eval/test_citation_mining.py
git commit -m "feat(eval): mine citation-reward reranker triplets"
```

---

### Task 4: Live runs — bucket report, CRAG A/B, flywheel seed

**Files:**
- Create: `docs/eval/miss_buckets_2026-07-14.md` (generated)
- Create: `docs/eval/crag_ab_2026-07-14.md` (hand-written comparison of two generated probe summaries)
- Create: `docs/eval/rows_c2_n40_crag_off.jsonl`, `docs/eval/rows_c2_n40_crag_on.jsonl` (generated; keep out of git if large — they contain corpus text: add to .gitignore in this task)
- Create: `data/finetune/citation_pairs.jsonl` (generated, gitignored dir)

**Interfaces:**
- Consumes: Tasks 1–3 CLIs; live stack (Elasticsearch + TEI embedding on :8080 + reranker GPU + DeepSeek/Gemini keys); `data/eval/c2_evalset_n40.jsonl`.
- Produces: bucket evidence (drives the HippoRAG-2 go/no-go), CRAG enable/keep-off decision, first citation-pair training file.

Prerequisites gate — run first; if any fails, STOP this task and report what is missing rather than running a partial eval:

- [ ] **Step 1: Verify stack + eval set**

```bash
curl -s localhost:9200/_cluster/health | head -c 200        # ES up
curl -s localhost:8080/health && echo                        # TEI embedding up
test -f data/eval/c2_evalset_n40.jsonl && wc -l data/eval/c2_evalset_n40.jsonl \
  || echo "EVAL SET MISSING — rebuild required"
grep -E "DEEPSEEK_API_KEY|GEMINI_API_KEY" .env | sed 's/=.*/=<set>/'
```
Expected: ES status green/yellow, TEI ok, 40-line eval set (or rebuild below), both keys set.

If the eval set is missing, rebuild it from the CURRENT corpus (validity rule) — also add a multi-hop arm for the HippoRAG gate:

```bash
uv run python scripts/eval/build_prod_evalset.py --n 40 --multihop 12 \
  --out data/eval/c2_evalset_n40.jsonl
```
(If rebuilt, note in the report that numbers are not directly comparable to the 2026-07-13 c2 runs — different question sample.)

- [ ] **Step 2: Baseline probe run, CRAG OFF (default), with per-row dump**

```bash
CRAG_ENABLED=false uv run python scripts/eval/oracle_probe.py \
  --eval-set data/eval/c2_evalset_n40.jsonl --n 40 --retries 3 \
  --rows-out docs/eval/rows_c2_n40_crag_off.jsonl \
  --out docs/eval/c2_probe_crag_off_2026-07-14.md
```
Expected: `[probe] oracle−system = +0.0xx`, 40 rows in the JSONL. (~30–60 min at ~30–60 s/question; run under `nohup`/background and poll.)

- [ ] **Step 3: Bucket the misses**

```bash
uv run python scripts/eval/report_miss_buckets.py \
  --rows docs/eval/rows_c2_n40_crag_off.jsonl \
  --out docs/eval/miss_buckets_2026-07-14.md --label c2_evalset_n40-crag-off
```
Expected: `[buckets] ~5/40 misses → {...}` — the bucket split IS the deliverable.

- [ ] **Step 4: CRAG ON arm (same set, same judges)**

```bash
CRAG_ENABLED=true uv run python scripts/eval/oracle_probe.py \
  --eval-set data/eval/c2_evalset_n40.jsonl --n 40 --retries 3 \
  --rows-out docs/eval/rows_c2_n40_crag_on.jsonl \
  --out docs/eval/c2_probe_crag_on_2026-07-14.md
```

- [ ] **Step 5: Abstain-safety check with CRAG ON**

CRAG's `_critique` treats an uncertain answer as ungrounded and retries — it must NOT convert clean out-of-corpus abstentions into hallucinations:

```bash
CRAG_ENABLED=true uv run python scripts/eval/run_refusal_ab.py
```
Expected: refusal classes on the OOC set unchanged vs the committed baseline (`docs/eval/benchmark_answerability_ab_2026-06-24_vi.md`) — zero `hallucinated`.

- [ ] **Step 6: Write the A/B decision doc**

Create `docs/eval/crag_ab_2026-07-14.md` from the two probe summaries + refusal check:
- table: arm | system avg | oracle−system | misses | bucket split | refusal classes
- Decision rule (pre-registered): flip `CRAG_ENABLED: bool = True` in `src/agentrag/config.py` only if CRAG-on shows system_avg ≥ +0.02 over CRAG-off AND zero new `hallucinated` on the refusal set; otherwise keep OFF and record why.
- Also record the HippoRAG-2 gate verdict from Step 3's buckets: `retrieval_miss` majority → green-light the StructMem/graph plan; `false_abstention` majority → floor/gate work instead; `generation_miss` majority → answer-prompt work.

- [ ] **Step 7: Seed the flywheel**

```bash
uv run python scripts/eval/mine_citation_pairs.py \
  --rows docs/eval/rows_c2_n40_crag_off.jsonl \
  --out data/finetune/citation_pairs.jsonl
wc -l data/finetune/citation_pairs.jsonl
```
Expected: dozens of triplets (rows with sys ≥ 0.75 × positives). Too few to train alone — accumulates with future probe runs via `--append`, and blends with `data/finetune/embed_triplets.jsonl` for the next reranker FT.

- [ ] **Step 8: Keep bulky row dumps out of git, commit docs**

```bash
printf 'docs/eval/rows_*.jsonl\n' >> .gitignore
git add .gitignore docs/eval/miss_buckets_2026-07-14.md docs/eval/crag_ab_2026-07-14.md \
  docs/eval/c2_probe_crag_off_2026-07-14.md docs/eval/c2_probe_crag_on_2026-07-14.md
git commit -m "docs(eval): miss buckets + CRAG A/B on real corpus"
```
(If the decision flips `CRAG_ENABLED`, commit that separately: `git add src/agentrag/config.py && git commit -m "feat(agent): enable CRAG critique loop (A/B-validated)"`.)

---

## Deferred (separate plan, gated on Task 4 Step 3 evidence)

**HippoRAG-2-style StructMem evolution** — phrase/passage bi-modal graph over `pam_entries`, synonym edges via embedding threshold (replaces the canonicalization prerequisite), query-to-triple seeding + PPR, LLM recognition-memory filter. Write `docs/superpowers/specs/` design + its own plan ONLY if the bucket report shows `retrieval_miss` (especially on multi-hop rows) as the dominant class. Reference: memory note `notebooklm-rag-course-notebook.md`, `structmem-evolution-hipporag.md`.

## Self-Review

- Spec coverage: (1) bucket misses → Tasks 1–2 + Task 4 Steps 2–3; (2) CRAG retry-before-abstain → already built in `graph_service.py`, measured in Task 4 Steps 4–6; (3) RMM flywheel → Task 3 + Task 4 Step 7; (4) HippoRAG gate → Task 4 Step 6 + Deferred section. ✓
- No placeholders: every code step carries full code; every run step carries the exact command + expected output. ✓
- Type consistency: `ProbeRow.detail` (Task 1) feeds Row dicts consumed by `bucket_row`/`summarize_buckets`/`render_report` (Task 2) and `mine_triplets` (Task 3); field names (`system_mean`, `cited_sources`, `packed`, `refusal_class`) match across all three tasks and both CLIs. ✓
