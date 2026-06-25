# Feedback → Langfuse Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push each thumbs 👍/👎 as a Langfuse score on the turn's trace.

**Architecture:** Capture `get_current_trace_id()` inside the observed `chat`, return it, persist on the assistant turn's `extra_metadata`; `/chat/feedback` loads the turn by `turn_id`, reads the trace id, and scores. Two guarded helpers keep langfuse specifics in `langfuse_client.py`. Non-stream path only (streaming deferred).

**Tech Stack:** Langfuse v2 (`get_current_trace_id`, `Langfuse().score`), SQLAlchemy async, FastAPI.

## Global Constraints

- All Langfuse touchpoints guarded by `settings.LANGFUSE_ENABLED` + try/except; the app is unchanged when off; feedback upsert must never fail because of scoring.
- Helpers live in `src/agentrag/common/langfuse_client.py` (the chokepoint).
- Score: `name="user_feedback"`, `value=float(rating ±1)`, `data_type="NUMERIC"`.
- Scope: `chat()` (used by `execute_chat`, `regenerate_chat`). `chat_stream` / streaming handler = out of scope (documented follow-up).

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/agentrag/common/langfuse_client.py` (modify) | `current_trace_id`, `score_trace` |
| `src/agentrag/agent/graph_service.py` (modify ~544) | return `langfuse_trace_id` |
| `src/agentrag/adapter/routers/chat.py` (modify ~448, ~695, ~1485) | persist id (×2) + score in `/feedback` |
| `tests/observability/test_langfuse_helpers.py` (extend) | offline guard tests |

---

### Task 1: helpers + capture + persist + score

**Files:**
- Modify: `src/agentrag/common/langfuse_client.py`
- Modify: `src/agentrag/agent/graph_service.py`
- Modify: `src/agentrag/adapter/routers/chat.py`
- Modify: `tests/observability/test_langfuse_helpers.py`

**Interfaces:**
- Produces: `current_trace_id() -> str | None`, `score_trace(trace_id, *, name, value, comment=None) -> None`.
- Consumes: `settings.LANGFUSE_ENABLED`; chat `result["langfuse_trace_id"]`; `ChatMessage.extra_metadata`.

- [ ] **Step 1: Write failing guard tests** (append to `tests/observability/test_langfuse_helpers.py`):

```python
def test_current_trace_id_none_when_disabled(monkeypatch):
    from src.agentrag.common.langfuse_client import current_trace_id
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", False)
    assert current_trace_id() is None


def test_score_trace_noop_when_disabled(monkeypatch):
    from src.agentrag.common.langfuse_client import score_trace
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", False)
    assert score_trace("trace-1", name="user_feedback", value=1.0) is None


def test_score_trace_noop_when_no_trace_id(monkeypatch):
    from src.agentrag.common.langfuse_client import score_trace
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", True)
    # No trace_id → must no-op (and not construct a Langfuse client).
    assert score_trace(None, name="user_feedback", value=-1.0) is None
```

- [ ] **Step 2: Run them — expect FAIL** (`ImportError: cannot import name 'current_trace_id'`).

Run: `uv run pytest tests/observability/test_langfuse_helpers.py -q`

- [ ] **Step 3: Add the helpers** to `langfuse_client.py` (after `update_turn_trace`):

```python
def current_trace_id() -> str | None:
    """The active Langfuse trace id inside an observed fn, else None (or when off)."""
    if not settings.LANGFUSE_ENABLED:
        return None
    try:
        from langfuse.decorators import langfuse_context

        return langfuse_context.get_current_trace_id()
    except Exception as exc:
        log.debug("langfuse get_current_trace_id skipped: %s", exc)
        return None


def score_trace(trace_id, *, name, value, comment=None) -> None:
    """Attach a score to a trace by id. No-op when off or trace_id is falsy."""
    if not settings.LANGFUSE_ENABLED or not trace_id:
        return
    try:
        from langfuse import Langfuse

        Langfuse().score(trace_id=str(trace_id), name=name, value=value,
                         data_type="NUMERIC", comment=comment)
    except Exception as exc:
        log.debug("langfuse score skipped: %s", exc)
```

- [ ] **Step 4: Run tests — expect PASS** (5 passed total in the file).

Run: `uv run pytest tests/observability/test_langfuse_helpers.py -q`

- [ ] **Step 5: Return the trace id from `chat()`.** In `graph_service.py`, extend the import and the return dict (~544):

```python
from src.agentrag.common.langfuse_client import current_trace_id, observe_chat_turn, update_turn_trace
```

Add this entry to the dict returned by `chat` (next to `"timings_ms"`):

```python
            "langfuse_trace_id": current_trace_id(),
```

- [ ] **Step 6: Persist the id on both non-stream assistant turns.** In `chat.py`,
in `execute_chat` (~448) and `regenerate_chat` (~695), add to each
`extra_metadata={...}` dict (after the `"sql_query"` line):

```python
                "langfuse_trace_id": result.get("langfuse_trace_id"),
```

- [ ] **Step 7: Score in `/chat/feedback`.** In `submit_chat_feedback` (~1485), right
after `await session.commit()` (the upsert) and before the activity-event block, add:

```python
    # Mirror the rating to Langfuse as a score on the turn's trace (best-effort).
    if settings.LANGFUSE_ENABLED:
        try:
            import uuid as _uuid
            from src.agentrag.common.langfuse_client import score_trace

            async with AsyncSessionLocal() as s2:
                msg = (
                    await s2.execute(
                        select(ChatMessage).where(ChatMessage.id == _uuid.UUID(str(turn_id)))
                    )
                ).scalar_one_or_none()
            trace_id = (msg.extra_metadata or {}).get("langfuse_trace_id") if msg else None
            score_trace(trace_id, name="user_feedback", value=float(rating_int),
                        comment=body.get("comment"))
        except Exception:
            _log.debug("feedback langfuse score skipped", exc_info=True)
```

(`settings`, `select`, `ChatMessage`, `AsyncSessionLocal`, `_log` are already imported in `chat.py` — confirm at the top; add any that are missing.)

- [ ] **Step 8: Import smoke + no regression.**

Run: `uv run python -c "import src.agentrag.adapter.routers.chat; from src.agentrag.agent.graph_service import GraphAgentService; print('ok')"`
Expected: `ok`.
Run: `uv run pytest -q --ignore=tests/ontology --ignore=tests/ingestion`
Expected: PASS (prior green + 3 new).

- [ ] **Step 9: Commit.**

```bash
git add src/agentrag/common/langfuse_client.py src/agentrag/agent/graph_service.py \
        src/agentrag/adapter/routers/chat.py tests/observability/test_langfuse_helpers.py
git commit -m "feat(observability): mirror thumbs feedback to Langfuse as a trace score

chat() returns its langfuse_trace_id; non-stream handlers persist it on the assistant
turn's extra_metadata; /chat/feedback loads the turn and scores user_feedback=rating
on that trace. Guarded helpers current_trace_id/score_trace, default-off. Streaming
path deferred."
```

---

### Task 2: live verification

**Files:** none (verification + a doc note). Requires the stack + Langfuse up.

- [ ] **Step 1: Produce a traced turn and persist it with a known turn id.** Easiest:
send a real `/chat` via `execute_chat` (UI at `:3000` or a curl) so an assistant
`ChatMessage` is persisted with `extra_metadata.langfuse_trace_id`. Note the assistant
turn id from the response (`turn_id`).

- [ ] **Step 2: Submit feedback for that turn.**

Run (substitute the real turn_id + a valid auth token/header):
`curl -s -X POST http://localhost:8000/chat/feedback -H 'Content-Type: application/json' -d '{"turn_id":"<id>","rating":1,"session_id":"<conv>"}'`
Expected: `{"ok": true, "rating": 1}`.

- [ ] **Step 3: Confirm the score on the trace.**

Run: `curl -s -u "pk-lf-agentrag-dev:sk-lf-agentrag-dev" "http://localhost:3002/api/public/scores?name=user_feedback&limit=3"`
Expected: a `user_feedback` score (value 1) whose `traceId` matches the turn's
`langfuse_trace_id`.

- [ ] **Step 4: Record + commit a short note** to `docs/eval/langfuse-online-2026-06-25.md` (append a "feedback→score verified" line), commit.

---

## Self-Review

**Spec coverage:** helpers → T1 S3 + tests S1; capture in chat → T1 S5; persist on the 2 non-stream turns → T1 S6; score in /feedback → T1 S7; offline tests → T1 S1; live verify → T2. Streaming explicitly out of scope (spec + Global Constraints). All mapped.

**Placeholder scan:** none — full helper + wiring code inline, exact anchors, exact commands. The `<id>`/`<conv>`/token in T2 are runtime values the operator substitutes from the live run, not code placeholders.

**Type consistency:** `current_trace_id() -> str|None` and `score_trace(trace_id, *, name, value, comment=None)` identical across helper def (S3), chat return (S5), and feedback call (S7); key `langfuse_trace_id` consistent across chat return, both metadata writes, and the feedback lookup; score name `user_feedback` consistent in S7 and the T2 verify query.
