# VITAL Improvement — P0 + P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make docs match the (post-SQL-removal) reality, lock a green test baseline, then close the #1 safety risk (confident hallucination on out-of-corpus questions) and validate/enable the dormant WS1–5 retrieval features via ablation.

**Architecture:** VITAL is a single-semantic-path Vietnamese-medical RAG agent (LangGraph 13-node). Retrieval = hybrid BM25+kNN+RRF → cross-encoder rerank → relevance-floor abstain. The structured-SQL path was removed in `e4eb895`; docs still describe it. Five RAG features (Contextual Retrieval, RAPTOR, CRAG+multihop, adaptive fast-path, semantic cache) are implemented behind default-OFF flags and never benchmarked.

**Tech Stack:** Python 3 / FastAPI / LangGraph, PostgreSQL+pgvector, Elasticsearch, Ollama+DeepSeek/Gemini, `bge-m3` (TEI) + `bge-reranker-v2-m3`, pytest, Next.js/vitest. Eval: DeepEval/RAGAS via `scripts/eval/run_ablation.py`.

## Global Constraints

- New config flags default **OFF** (`false` / disabled) — production behavior unchanged until a flag wins an ablation row. Verbatim pattern from existing flags in `src/agentrag/config.py`.
- Retrieval abstain logic lives in `src/agentrag/agent/service.py` (`_is_thin_context`, `_answer_system_prompt`) + `src/agentrag/agent/context.py` (`apply_relevance_floor`) + `src/agentrag/agent/graph_service.py:409`. Do not duplicate; extend in place.
- Ablation must run on the **live stack** (PG + ES + DeepSeek key + Ollama) with `STRUCTMEM_INGEST_MODE=sync` and `UPLOAD_DEDUPE_BY_HASH=false` (the runner forces these) — re-ingest per config or CR/RAPTOR silently no-op.
- Citations are page-aware; never weaken the page-citation contract.
- Commit after every task. Run `uv run pytest -q` (or `make test-fast`) before each commit.

---

## File Structure

| Path | Responsibility | Tasks |
|---|---|---|
| `README.md`, `ARCHITECTURE.md` | top-level docs — must describe single semantic path | T1 |
| `.env.example`, module `README.md`s | remove dead structured-path references | T2 |
| `docs/eval/test-baseline-2026-06-24.md` (new) | recorded green/red test baseline | T3 |
| `docs/superpowers/specs/2026-06-24-vital-improvement-roadmap-design.md` | branch decision appendix | T4 |
| `src/agentrag/agent/service.py`, `config.py` | answerability gate + stronger abstain | T5 |
| `tests/agent/test_answerability_gate.py` (new) | unit tests for the gate | T5 |
| `scripts/eval/run_ablation.py` | isolated per-WS ablation rows | T6 |
| `docs/eval/benchmark_ablation_2026-06-24.md` (generated) | ablation results | T6 |
| `docs/eval/rerank_floor_distribution_2026-06-24.md` (new) | per-specialty floor measurement | T7 |

---

## P0 — Truth & baseline

### Task 1: Fix stale top-level docs (single semantic path)

**Files:**
- Modify: `README.md` (lines 3–4, 19–20, 41, 96, 104)
- Modify: `ARCHITECTURE.md` (lines 17–21, 104–105)

**Interfaces:** none (docs only).

- [ ] **Step 1: Rewrite `README.md` intro (lines 3–6).** Replace "Hai luồng suy luận song song (semantic hybrid + structured SQL)" with single-path wording:

```
Nền tảng RAG cho học liệu y khoa Việt Nam. Một luồng suy luận semantic
(hybrid BM25 + kNN + RRF + StructMem KG), bộ nhớ phân cấp (StructMem),
domain-aware retrieval, reasoning trace + cost dashboard, UI Next.js
tương thích open-notebook.
```

- [ ] **Step 2: Fix `README.md` "Tính năng chính" bullet (lines 19–20).** Replace the "Semantic + Structured paths" bullet with:

```
- **Semantic hybrid retrieval** — BM25 + kNN + RRF + StructMem KG, rerank
  cross-encoder, abstain-on-thin-context khi thiếu căn cứ.
```

- [ ] **Step 3: Fix `README.md` Reasoning-Plane summary (line 41).** Replace "classify intent → semantic agent loop **hoặc** structured SQL pipeline → answer" with "semantic agent loop → answer + grounding".

- [ ] **Step 4: Remove dead module-table rows (lines 96, 104).** Delete the `structured` row entirely. In the `mcp` row (line 104) change "MCP tools (hybrid search + SQL reasoning)" to "MCP tools (hybrid search)".

- [ ] **Step 5: Fix `ARCHITECTURE.md` Reasoning-Plane box (lines 17–21).** Delete the three `structured/*` lines (`query_classifier.py`, `pipeline.py`) and the `structured/schema_discovery`/`sql_engine` references. Keep `domain_router.py` and `reasoning_knowledge.py`.

- [ ] **Step 6: Fix `ARCHITECTURE.md` file map (lines 104–105).** Delete the "Intent classifier" and "SQL pipeline" rows pointing at `src/agentrag/structured/*`.

- [ ] **Step 7: Verify no stale refs remain.**

Run: `grep -rnE "structured SQL|SQL reasoning|Hai luồng|two paths|structured/(pipeline|query_classifier|sql_engine|schema_discovery|extractor|synthesizer)" README.md ARCHITECTURE.md`
Expected: no output.

- [ ] **Step 8: Commit.**

```bash
git add README.md ARCHITECTURE.md
git commit -m "docs: align README+ARCHITECTURE with single semantic path (SQL path removed in e4eb895)"
```

### Task 2: Remove dead config flag + stale module-README references

**Files:**
- Modify: `.env.example:219`
- Modify: `src/agentrag/services/README.md` (lines 32, 81, 141), `src/agentrag/common/README.md` (lines 42, 52)

**Interfaces:** none.

- [ ] **Step 1: Confirm the flag is already gone from code.**

Run: `grep -rn "STRUCTURED_REASONING_ENABLED" src/agentrag/config.py`
Expected: no output (removed in `e4eb895`; only `.env.example` + generated `egg-info` still mention it).

- [ ] **Step 2: Delete the dead flag line in `.env.example`.** Remove line 219 `STRUCTURED_REASONING_ENABLED=true` and its comment block if any.

- [ ] **Step 3: Scrub module READMEs.** In `src/agentrag/services/README.md` and `src/agentrag/common/README.md`, replace each `structured/pipeline.py` reference: in `services/README.md` line 32/141 change "Vẫn dùng bởi MCP + structured pipeline" → "Vẫn dùng bởi MCP"; line 81 drop "/ `structured.pipeline`". In `common/README.md` lines 42/52 change the `structured/pipeline.py` caller example to `agent/graph_service.py`.

- [ ] **Step 4: Verify.**

Run: `grep -rn "structured/pipeline\|structured pipeline\|structured.pipeline" src/agentrag/*/README.md`
Expected: no output.

- [ ] **Step 5: Commit.**

```bash
git add .env.example src/agentrag/services/README.md src/agentrag/common/README.md
git commit -m "docs+config: drop dead structured-reasoning flag and module-README refs"
```

### Task 3: Establish green test baseline

**Files:**
- Create: `docs/eval/test-baseline-2026-06-24.md`

**Interfaces:** none.

- [ ] **Step 1: Run the full backend suite, capture output.**

Run: `uv run pytest -q 2>&1 | tee /tmp/pytest-baseline.txt; tail -30 /tmp/pytest-baseline.txt`
Expected: a pass/fail summary line. Note any `ImportError`/`ModuleNotFoundError` referencing removed `structured.*` — those are SQL-removal fallout.

- [ ] **Step 2: Fix SQL-removal fallout only.** For each failing import that references a deleted `structured/*` module, delete or update the orphaned test. Do NOT fix unrelated pre-existing failures here — record them instead.

Run after each fix: `uv run pytest -q 2>&1 | tail -5`

- [ ] **Step 3: Run the frontend suite, capture output.**

Run: `cd frontend && npm test 2>&1 | tee /tmp/vitest-baseline.txt; tail -30 /tmp/vitest-baseline.txt`
Expected: pass/fail summary. The known-pre-existing reds (locale-parity for 9 non-`ragSignals` keys, unused-key detection, 3 e2e Playwright specs vitest mis-collects) are expected — catalogue them, do not fix here.

- [ ] **Step 4: Write the baseline doc.** Record: backend passed/failed counts, which tests were deleted/fixed (with reason), frontend passed/failed counts, and the explicit list of known-pre-existing reds carried forward. State the green floor: "backend green except {list}; frontend green except {known reds}".

- [ ] **Step 5: Commit.**

```bash
git add docs/eval/test-baseline-2026-06-24.md tests/
git commit -m "test: record 2026-06-24 baseline + remove orphaned structured-path tests"
```

### Task 4: Branch integration decision

**Files:**
- Modify: `docs/superpowers/specs/2026-06-24-vital-improvement-roadmap-design.md` (append appendix)

**Interfaces:** none.

- [ ] **Step 1: Inspect branch divergence.**

Run: `git log --oneline master..feat/ragas-langfuse-reranker | wc -l; git log --oneline feat/ragas-langfuse-reranker..master | wc -l; git log --oneline --all --simplify-by-decoration | head`
Expected: commit counts each way + branch tips.

- [ ] **Step 2: Append an "Appendix: branch decision" section** to the roadmap spec recording: how far `feat/ragas-langfuse-reranker` leads `master`, whether `structmem` is merged/dead, and the chosen path (recommended: squash-merge feat → master once P0+P1 land and re-benchmark is green; delete `structmem` if its commits are already in feat).

- [ ] **Step 3: Commit.**

```bash
git add docs/superpowers/specs/2026-06-24-vital-improvement-roadmap-design.md
git commit -m "docs(spec): branch integration decision appendix"
```

---

## P1 — Safety · Quality · Latency

### Task 5: Answerability gate — kill confident hallucination on out-of-corpus

**Problem:** The 8/15 confident-fabricate cases retrieve distractors that the reranker scores **≥ 0.6**, so `_is_thin_context` (max rerank < floor) never fires and the model answers confidently. Fix = a gray-band answerability check + a more forceful abstain prompt, behind a default-OFF flag.

**Files:**
- Modify: `src/agentrag/config.py` (after `RETRIEVAL_RELEVANCE_FLOOR`, ~line 127)
- Modify: `src/agentrag/agent/service.py` (`_is_thin_context` region + `_answer_system_prompt:156-163`)
- Create: `tests/agent/test_answerability_gate.py`

**Interfaces:**
- Produces: `_in_gray_band(packed_context, floor, margin) -> bool` — True when best rerank score ∈ `[floor, floor+margin)`. Pure function.
- Consumes: existing `settings.RETRIEVAL_RELEVANCE_FLOOR`, `settings.ANSWER_ABSTAIN_ON_THIN_CONTEXT`.

- [ ] **Step 1: Write the failing test for the gray-band helper.**

```python
# tests/agent/test_answerability_gate.py
from src.agentrag.agent.service import _in_gray_band

def _ctx(*scores):
    return [{"rerank_score": s} for s in scores]

def test_gray_band_true_when_best_in_band():
    # floor 0.6, margin 0.13 → band [0.60, 0.73)
    assert _in_gray_band(_ctx(0.50, 0.64), floor=0.6, margin=0.13) is True

def test_gray_band_false_when_best_above_band():
    assert _in_gray_band(_ctx(0.74, 0.40), floor=0.6, margin=0.13) is False

def test_gray_band_false_when_best_below_floor():
    # below floor is already handled by _is_thin_context, not the gray band
    assert _in_gray_band(_ctx(0.40, 0.55), floor=0.6, margin=0.13) is False

def test_gray_band_false_when_no_scores():
    assert _in_gray_band([{"text": "x"}], floor=0.6, margin=0.13) is False
```

- [ ] **Step 2: Run it, verify it fails.**

Run: `uv run pytest tests/agent/test_answerability_gate.py -v`
Expected: FAIL with `ImportError: cannot import name '_in_gray_band'`.

- [ ] **Step 3: Implement `_in_gray_band` in `service.py`** (next to `_is_thin_context`):

```python
def _in_gray_band(packed_context: list[dict[str, Any]] | None, floor: float, margin: float) -> bool:
    """True when the BEST rerank score sits in the uncertain band [floor, floor+margin) —
    above the abstain floor but not clearly relevant. These are the out-of-corpus
    distractors that score just over the floor and drive confident hallucination."""
    scores = [c.get("rerank_score") for c in (packed_context or [])
              if c.get("rerank_score") is not None]
    if not scores:
        return False
    best = max(float(s) for s in scores)
    return floor <= best < floor + margin
```

- [ ] **Step 4: Run the test, verify it passes.**

Run: `uv run pytest tests/agent/test_answerability_gate.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Add config knobs (default OFF/safe).** In `config.py` after `RETRIEVAL_RELEVANCE_FLOOR`:

```python
    #: When best rerank score is in [floor, floor+GRAY_MARGIN), treat the
    #: context as uncertain and force the strong-abstain prompt (out-of-corpus
    #: distractors that score just over the floor → confident-hallucination fix).
    ANSWERABILITY_GRAY_MARGIN: float = 0.13
    #: Master switch for the gray-band abstain. Default OFF — enable after the
    #: refusal-set re-eval beats baseline (docs/eval/benchmark_abstain_ab_*).
    ANSWERABILITY_GATE_ENABLED: bool = False
```

- [ ] **Step 6: Wire the gate + strengthen the abstain prompt** in `_answer_system_prompt` (`service.py:156`). Change the guard and the prompt body:

```python
    thin = _is_thin_context(packed_context, settings.RETRIEVAL_RELEVANCE_FLOOR)
    gray = (settings.ANSWERABILITY_GATE_ENABLED
            and _in_gray_band(packed_context, settings.RETRIEVAL_RELEVANCE_FLOOR,
                              settings.ANSWERABILITY_GRAY_MARGIN))
    if settings.ANSWER_ABSTAIN_ON_THIN_CONTEXT and (thin or gray):
        return (
            f"{_lang_instruction(question)} "
            "The retrieved context does NOT contain information that answers the question. "
            "You MUST refuse: reply in ONE sentence that the document/corpus has no information "
            "on this topic. This rule overrides everything else. "
            "Do NOT answer from your own medical/background knowledge even if you are certain of "
            "the answer. Do NOT guess, do NOT hedge, do NOT cite any source. Do NOT return JSON."
        )
```

- [ ] **Step 7: Run the full agent test module to check no regression.**

Run: `uv run pytest tests/agent -q`
Expected: PASS (existing agent tests + the 4 new gray-band tests). Fix any breakage before commit.

- [ ] **Step 8: Commit the code change.**

```bash
git add src/agentrag/config.py src/agentrag/agent/service.py tests/agent/test_answerability_gate.py
git commit -m "feat(agent): gray-band answerability gate + stronger abstain prompt (default OFF)

Out-of-corpus distractors scoring just over RETRIEVAL_RELEVANCE_FLOOR bypassed
_is_thin_context and drove confident hallucination. ANSWERABILITY_GATE_ENABLED
forces strong-abstain when best rerank ∈ [floor, floor+margin)."
```

- [ ] **Step 9: A/B the refusal set on the live stack.** Run the existing abstain A/B harness twice — gate OFF (baseline) then `ANSWERABILITY_GATE_ENABLED=true` — over the 15 out-of-corpus questions. Use the same runner that produced `docs/eval/benchmark_abstain_ab_2026-06-19_vi.md`.

Run (gate ON): `ANSWERABILITY_GATE_ENABLED=true uv run python scripts/eval/<abstain_ab_runner>.py --refusal-only`
Decision rule: enable the flag in `.env` only if **confident-fabricate ≤ 0.15** (from 0.533) AND in-corpus quality (recall/precision/faithfulness) stays flat vs the 19/06 numbers.

- [ ] **Step 10: Record the A/B result + decision** in `docs/eval/benchmark_answerability_ab_2026-06-24_vi.md`; if it wins, flip `ANSWERABILITY_GATE_ENABLED=true` in `.env.example` default-OFF note and `.env`. Commit.

```bash
git add docs/eval/benchmark_answerability_ab_2026-06-24_vi.md .env.example
git commit -m "eval(safety): answerability-gate A/B on out-of-corpus refusal set"
```

### Task 6: Validate WS1–5 via isolated ablation

**Problem:** `run_ablation.py` only has cumulative configs (`baseline`, `cr`, `cr_raptor`, `cr_raptor_crag`, `full`). Cumulative rows can't attribute a regression to one feature. Add isolated single-flag rows so each WS gets a clean baseline-vs-feature comparison.

**Files:**
- Modify: `scripts/eval/run_ablation.py` (`ABLATIONS` dict, ~line 45)
- Generated: `docs/eval/benchmark_ablation_2026-06-24.md`

**Interfaces:**
- Consumes: existing `ABLATIONS: dict[str, dict[str, str]]`, `build_env`, `--only`, `--suite`, `--n`.

- [ ] **Step 1: Add isolated WS rows to `ABLATIONS`.** Insert alongside the cumulative ones:

```python
    "crag_only": {"CRAG_ENABLED": "true"},
    "fastpath_only": {"ADAPTIVE_ROUTING_ENABLED": "true"},
    "semcache_only": {"SEMANTIC_CACHE_ENABLED": "true"},
    "multihop_only": {"AGENT_MULTIHOP_ENABLED": "true"},
```

(`cr` and `cr_raptor` already isolate WS1/WS2.)

- [ ] **Step 2: Sanity-check the config parses.**

Run: `uv run python -c "from scripts.eval.run_ablation import ABLATIONS; print(list(ABLATIONS))"`
Expected: list includes `crag_only`, `fastpath_only`, `semcache_only`, `multihop_only`.

- [ ] **Step 3: Preflight the live stack.**

Run: `make health`
Expected: PG + ES + embedding + reranker reachable; a working judge key (DeepSeek). If Ollama is down, latency/cost rows will be cloud-inflated — note it, do not block quality rows.

- [ ] **Step 4: Run the ablation matrix** (smaller n first to catch errors, then full):

Run: `uv run python scripts/eval/run_ablation.py --suite both --n 10 --only baseline,cr,cr_raptor,crag_only,fastpath_only,semcache_only,multihop_only,full`
Then (if clean): rerun at `--n 20`.
Expected: writes `docs/eval/benchmark_ablation_2026-06-24.md` with one row per config.

- [ ] **Step 5: Decide per flag.** For each WS, compare its isolated row to `baseline`:
  - latency: `fastpath_only` / `semcache_only` should cut p50 toward < 10s with quality flat.
  - quality: `cr`, `cr_raptor`, `crag_only` should lift precision (0.699 → 0.80+) / correctness without dropping faithfulness.
  Enable in `.env.example`/`.env` only the flags whose row beats baseline on its target metric and does not regress faithfulness.

- [ ] **Step 6: Commit results + any flag flips.**

```bash
git add scripts/eval/run_ablation.py docs/eval/benchmark_ablation_2026-06-24.md .env.example
git commit -m "eval: isolated WS1-5 ablation rows + 2026-06-24 results; enable winning flags"
```

### Task 7: Per-specialty rerank-floor measurement

**Problem:** The global floor 0.6 was set from a corpus-wide rerank distribution (in-corpus ≈0.73, out-of-corpus ≈0.50). Different specialties may separate at different thresholds; a per-domain floor can raise precision without losing recall.

**Files:**
- Create: `docs/eval/rerank_floor_distribution_2026-06-24.md`
- (Optional, only if data supports it) Modify: `src/agentrag/config.py` to add a per-domain floor override map.

**Interfaces:**
- Consumes: `DomainRouter.classify`, `RetrievalService.search`, rerank scores on packed context.

- [ ] **Step 1: Collect rerank-score distributions per specialty.** Over the in-corpus benchmark questions grouped by their `system`/`specialty` domain, record the max rerank score per question. Bucket by domain.

Run: `uv run python scripts/eval/<floor_distribution_script>.py --group-by specialty` (write this small read-only script if absent; it reuses the agent retrieval path, no writes).

- [ ] **Step 2: Tabulate** in `docs/eval/rerank_floor_distribution_2026-06-24.md`: per specialty, the median/min in-corpus max-score and (if available) out-of-corpus max-score, and the separating threshold.

- [ ] **Step 3: Decide.** If specialties separate cleanly at distinct thresholds (gap > 0.05 between domains), add a `RETRIEVAL_RELEVANCE_FLOOR_BY_DOMAIN: dict` override (default empty → falls back to global 0.6). If they don't separate, record "global 0.6 is adequate" and stop — do not add complexity for no gain (YAGNI).

- [ ] **Step 4: Commit.**

```bash
git add docs/eval/rerank_floor_distribution_2026-06-24.md src/agentrag/config.py
git commit -m "eval+config: per-specialty rerank-floor measurement (+ optional per-domain override)"
```

---

## Self-Review

**Spec coverage:** P0.1→T1, P0.2→T2, P0.3→T3, P0.4→T4, P1.5→T5, P1.6→T6, P1.7→T7. All P0+P1 spec items mapped. (P2/P3 deferred to later plans, per spec.)

**Placeholder scan:** Empirical tasks (T3, T5.9, T6, T7) carry decision rules and exact commands instead of fabricated PASS output, because their results are measured on the live stack — the procedure and the accept/reject threshold are fully specified. `<abstain_ab_runner>` / `<floor_distribution_script>` are the only name placeholders: the executor confirms the exact filename via `ls scripts/eval/` (the runner that wrote `benchmark_abstain_ab_2026-06-19_vi.md`) before running.

**Type consistency:** `_in_gray_band(packed_context, floor, margin)` defined in T5.3 and called in T5.6 with the same signature; config names `ANSWERABILITY_GRAY_MARGIN` / `ANSWERABILITY_GATE_ENABLED` consistent across T5.5/T5.6/T5.9. `ABLATIONS` keys added in T6.1 reused verbatim in T6.4.

## Note on remaining placeholders

Two eval scripts are referenced by role, not exact filename (`scripts/eval/` abstain-A/B runner; a small per-specialty floor-distribution script that may need writing). First executor step for T5.9 and T7.1 is `ls scripts/eval/` to bind the name; if the floor script is absent, write it as a read-only reuse of the agent retrieval path before Step 2.
