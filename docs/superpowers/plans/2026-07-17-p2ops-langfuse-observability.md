# P2-ops slice 1 — Langfuse Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn on self-hosted Langfuse tracing for prod chat turns + user-feedback scores, PHI-safe (metadata only), and prove it end-to-end.

**Architecture:** No app-logic change — `langfuse_client.py` is already wired (`main.py` lifespan calls `init_langfuse()` at startup and `langfuse_flush()` at shutdown; `chat.py` `/feedback` mirrors the rating via `score_trace`). This slice adds one regression test for the enabled-but-unreachable degradation contract, enables the flag + observability docker profile in this environment, verifies live, and writes a runbook.

**Tech Stack:** FastAPI (`main:app`), self-hosted Langfuse v2 + its Postgres (docker-compose `observability` profile, headless `LANGFUSE_INIT_*` bootstrap), pytest, `uv`.

## Global Constraints

- `config.py` default `LANGFUSE_ENABLED=False` stays False — enable per-deployment via `.env` (opt-in; observability depends on the profile being up). Copy verbatim: `LANGFUSE_ENABLED`, `OBSERVABILITY_CAPTURE_CONTENT`.
- PHI-safe: `OBSERVABILITY_CAPTURE_CONTENT` stays `false` — traces carry NO question/answer text.
- Langfuse being down MUST NOT break chat — every `langfuse_client` call is best-effort (try/except / disabled-gate). This is the safety contract Task 1 locks.
- `.env` is gitignored — the "enable" is environment state, documented in the runbook, not committed. Only tests + the runbook are committed.

---

### Task 1: Regression test — enabled-but-unreachable degrades gracefully

The existing tests (`tests/observability/test_langfuse_helpers.py`, `test_langfuse_client.py`) cover the DISABLED path. The untested gap is the prod-safety contract: when `LANGFUSE_ENABLED=True` but the Langfuse SDK/host raises, the helpers must swallow the error (never propagate into a chat turn). The behavior already exists (try/except in `score_trace`, `update_turn_trace`, `langfuse_flush`); this task locks it against regression.

**Files:**
- Modify/Test: `tests/observability/test_langfuse_helpers.py` (append 3 tests)

**Interfaces:**
- Consumes: `score_trace(trace_id, *, name, value, comment=None) -> None`, `update_turn_trace(*, name=None, session_id=None, metadata=None) -> None`, `langfuse_flush() -> None` from `src.agentrag.common.langfuse_client`; `settings` from `src.agentrag.config`.
- Produces: nothing downstream (test-only).

- [ ] **Step 1: Write the failing/locking tests**

Append to `tests/observability/test_langfuse_helpers.py`:

```python
def test_score_trace_swallows_when_langfuse_raises(monkeypatch):
    """Enabled but Langfuse unreachable → score_trace must NOT raise into the caller
    (the /feedback endpoint mirrors ratings via this; Langfuse downtime must not 500)."""
    import langfuse
    from src.agentrag.common.langfuse_client import score_trace
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", True)

    class _Boom:
        def __init__(self, *a, **k):
            pass
        def score(self, *a, **k):
            raise RuntimeError("langfuse down")

    monkeypatch.setattr(langfuse, "Langfuse", _Boom, raising=False)
    assert score_trace("trace-xyz", name="user_feedback", value=1.0) is None  # swallowed


def test_update_turn_trace_swallows_when_langfuse_raises(monkeypatch):
    """Enabled but the decorator context raises → update_turn_trace must no-op, not raise."""
    from src.agentrag.common import langfuse_client
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", True)

    class _BoomCtx:
        @staticmethod
        def update_current_trace(*a, **k):
            raise RuntimeError("no active trace")

    import langfuse.decorators as lfd
    monkeypatch.setattr(lfd, "langfuse_context", _BoomCtx, raising=False)
    assert langfuse_client.update_turn_trace(name="q", session_id="c1", metadata={"k": 1}) is None


def test_langfuse_flush_swallows_when_langfuse_raises(monkeypatch):
    """Enabled but flush paths raise → langfuse_flush must swallow so shutdown never breaks."""
    from src.agentrag.common.langfuse_client import langfuse_flush
    import langfuse
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", True)

    class _Boom:
        def __init__(self, *a, **k):
            pass
        def flush(self, *a, **k):
            raise RuntimeError("flush failed")

    # break both the openai-integration flush and the singleton flush
    import sys
    monkeypatch.setitem(sys.modules, "langfuse.openai", type(sys)("langfuse.openai"))
    monkeypatch.setattr(langfuse, "Langfuse", _Boom, raising=False)
    assert langfuse_flush() is None  # swallowed, no raise
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/observability/test_langfuse_helpers.py -v`
Expected: all PASS (the try/except contract already holds; these lock it). If any RAISES instead of returning None, that is a real safety bug in `langfuse_client` — fix the missing try/except there before proceeding.

- [ ] **Step 3: Run the full observability suite (no regressions)**

Run: `uv run pytest tests/observability/ -q`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/observability/test_langfuse_helpers.py
git commit -m "test(observability): lock Langfuse enabled-but-unreachable degradation contract"
```

---

### Task 2: Enable + live end-to-end verification

Bring up the self-hosted Langfuse, enable the flag in this environment, and prove a real turn traces (metadata-only) with a feedback score — and that Langfuse downtime does not break chat. This task's deliverable is the verification evidence captured in the runbook (Task 3); nothing here is committed (`.env` is gitignored).

**Files:**
- Modify (uncommitted, environment): `.env` (`LANGFUSE_ENABLED=true`)
- No source changes.

**Interfaces:**
- Consumes: docker-compose `observability` profile; `get_agent_service().chat(...)`; `score_trace(...)`; Langfuse public API `GET /api/public/traces` (basic-auth = public:secret key).

- [ ] **Step 1: Bring up the observability profile**

```bash
docker compose --profile observability up -d langfuse-db langfuse
# wait for healthy
until curl -sf localhost:3000/api/public/health >/dev/null; do sleep 3; done; echo "langfuse up"
```
Expected: `langfuse up`. (Headless project bootstrapped via `LANGFUSE_INIT_*` in docker-compose.yml.)

- [ ] **Step 2: Enable the flag in `.env`**

Set in `.env` (leave `config.py` default False):
```
LANGFUSE_ENABLED=true
OBSERVABILITY_CAPTURE_CONTENT=false
LANGFUSE_HOST=http://localhost:3000
```
(If driving from inside the compose network use `http://langfuse:3000`.)

- [ ] **Step 3: Drive one real traced turn (agent path = same chokepoints as prod HTTP)**

Write `/tmp/trace_turn.py`:
```python
import asyncio, sys
sys.path.insert(0, ".")
from src.agentrag.common.langfuse_client import init_langfuse, current_trace_id, score_trace, langfuse_flush
from src.agentrag.agent.factory import get_agent_service

async def main():
    init_langfuse()  # same call main.py lifespan makes
    agent = get_agent_service()
    out = await agent.chat(question="Metformin được chỉ định trong điều trị bệnh gì?",
                           document_title=None, conversation_id="obs-verify-1")
    print("answer_len:", len((out.get("answer") or "")))
    langfuse_flush()

asyncio.run(main())
```
Run: `LANGFUSE_ENABLED=true OBSERVABILITY_CAPTURE_CONTENT=false LANGFUSE_HOST=http://localhost:3000 PYTHONPATH=. uv run python /tmp/trace_turn.py`
Expected: prints `answer_len: <n>` with no traceback.

- [ ] **Step 4: Verify the trace exists and is metadata-only (PHI gate)**

```bash
source .env
curl -s -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
  "http://localhost:3000/api/public/traces?limit=5" | python3 -m json.tool | head -60
```
Expected: at least one recent trace. Inspect its `input`/`output` fields — they MUST be null/empty (content off). Confirm the medical question text does NOT appear anywhere in the JSON. If the question text appears, STOP — the PHI gate is not holding; do not proceed.

- [ ] **Step 5: Verify a feedback score attaches to a trace**

```bash
LANGFUSE_ENABLED=true LANGFUSE_HOST=http://localhost:3000 PYTHONPATH=. uv run python -c "
from src.agentrag.common.langfuse_client import init_langfuse, score_trace, langfuse_flush
import sys; sys.path.insert(0,'.')
init_langfuse()
# use a trace id from step 4's listing:
score_trace('$(source .env; curl -s -u \"$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY\" localhost:3000/api/public/traces?limit=1 | python3 -c \"import sys,json;print(json.load(sys.stdin)['data'][0]['id'])\")', name='user_feedback', value=1.0)
langfuse_flush()
print('scored')
"
```
Then re-query the trace and confirm a `user_feedback` score (value 1.0) is attached:
```bash
source .env; curl -s -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" "http://localhost:3000/api/public/scores?limit=5" | python3 -m json.tool | head -30
```
Expected: a `user_feedback` NUMERIC score present.

- [ ] **Step 6: Verify graceful degradation (Langfuse down ≠ chat down)**

```bash
docker compose stop langfuse
LANGFUSE_ENABLED=true LANGFUSE_HOST=http://localhost:3000 PYTHONPATH=. uv run python /tmp/trace_turn.py
docker compose --profile observability up -d langfuse
```
Expected: `/tmp/trace_turn.py` still prints `answer_len: <n>` with no traceback — the turn succeeds even though Langfuse is unreachable.

- [ ] **Step 7: Record results (feeds Task 3)**

Note the observed trace id, the confirmed absence of content, the score, and the degradation result — these go verbatim into the runbook's "verified" section in Task 3. No commit here.

---

### Task 3: Runbook

Document bring-up + the verification so the next operator can reproduce it and knows the PHI/degradation guarantees.

**Files:**
- Create: `docs/ops/langfuse-observability.md`

**Interfaces:**
- Consumes: the commands + observed results from Task 2.

- [ ] **Step 1: Write the runbook**

Create `docs/ops/langfuse-observability.md`:
```markdown
# Langfuse observability — bring-up & verification runbook (P2-ops slice 1)

Self-hosted Langfuse v2 tracing for prod chat turns + user-feedback scores. PHI-safe:
traces carry metadata only (latency, tokens, tool trace, rerank scores, feedback score) —
NO question/answer text (`OBSERVABILITY_CAPTURE_CONTENT=false`).

## Enable
1. `docker compose --profile observability up -d langfuse-db langfuse`
   (Langfuse v2 + its Postgres; project keys bootstrapped headlessly via `LANGFUSE_INIT_*`.)
2. In `.env` (NOT config.py — this is per-deployment opt-in):
   `LANGFUSE_ENABLED=true`, `OBSERVABILITY_CAPTURE_CONTENT=false`,
   `LANGFUSE_HOST=http://localhost:3000` (host) or `http://langfuse:3000` (in-network).
3. Restart the API (`make api`) — `main.py` lifespan calls `init_langfuse()`; traces flow.
   UI at http://localhost:3000.

## Verify (4-point check)
1. Boot: API starts, logs `Langfuse tracing enabled → <host>`, no error.
2. Trace: a chat turn produces a trace via `GET /api/public/traces`; `input`/`output` are
   empty (content off) — confirm no question/answer text in the JSON.
3. Score: `POST /on/api/feedback {turn_id, rating:+1}` (or `score_trace(...)`) attaches a
   `user_feedback` NUMERIC score to that turn's trace (`GET /api/public/scores`).
4. Degradation: with the `langfuse` container stopped, a chat turn still succeeds — every
   `langfuse_client` call is best-effort (locked by `tests/observability/test_langfuse_helpers.py`).

## Guarantees
- PHI: no free-text (question/answer/comment) in traces while `OBSERVABILITY_CAPTURE_CONTENT=false`.
- Safety: Langfuse downtime never breaks chat (best-effort client; regression-tested).
- Default off: `config.py LANGFUSE_ENABLED=False` — nothing traces unless a deployment opts in via `.env`.

## Cost / notes
- Adds two containers (Langfuse + its Postgres) on the box (~RAM); fine for a home/dev rig.
- Flush on shutdown via `langfuse_flush()` in the lifespan — restart won't drop buffered traces.

## Not in this slice
Automated online-eval judges on sampled turns, and flywheel scheduling
(`mine_citation_pairs_prod` + FT) — separate later slices.

## Verified <DATE>
<paste the Task 2 observations: trace id, content-absent confirmation, score present, degradation OK>
```
(Fill the `## Verified` block from Task 2 Step 7 before committing.)

- [ ] **Step 2: Commit**

```bash
git add docs/ops/langfuse-observability.md
git commit -m "docs(ops): Langfuse observability bring-up + verification runbook"
```
