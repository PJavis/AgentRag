# Langfuse Online + Per-turn Traces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable the already-wired Langfuse tracing and group each `/chat` turn into one named trace.

**Architecture:** Two guarded helpers in the existing `langfuse_client.py` chokepoint (`observe_chat_turn` decorator + `update_turn_trace`), applied once to `GraphAgentService.chat`, plus ops enablement (containers + `.env` + restart). Everything is a no-op when `LANGFUSE_ENABLED` is False.

**Tech Stack:** Langfuse v2 (2.60.10) `decorators.observe` / `langfuse_context.update_current_trace`, FastAPI, Docker Compose.

## Global Constraints

- All langfuse touchpoints guarded by `settings.LANGFUSE_ENABLED` + try/except; app behavior unchanged when off; a chat must never break if Langfuse is unreachable.
- Decorator/helpers live in `src/agentrag/common/langfuse_client.py` (the single chokepoint). Do not scatter langfuse imports.
- Trace grouping key: `session_id = conversation_id`. Trace `name = question[:80]`.
- Compose service is on host **3002** (`LANGFUSE_HOST=http://localhost:3002`); auto-provisioned keys `pk-lf-agentrag-dev` / `sk-lf-agentrag-dev`.

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/agentrag/common/langfuse_client.py` (modify) | add `observe_chat_turn`, `update_turn_trace` |
| `src/agentrag/agent/graph_service.py` (modify ~499) | decorate `chat`, set trace attrs |
| `tests/observability/test_langfuse_helpers.py` (create) | offline guard tests |
| `.env` (modify, gitignored) | enable + keys + host |

---

### Task 1: guarded helpers + wiring + offline tests

**Files:**
- Modify: `src/agentrag/common/langfuse_client.py`
- Modify: `src/agentrag/agent/graph_service.py` (decorate `chat`, line ~499)
- Create: `tests/observability/test_langfuse_helpers.py`

**Interfaces:**
- Produces: `observe_chat_turn(fn: Callable) -> Callable`, `update_turn_trace(*, name: str | None = None, session_id: str | None = None, metadata: dict | None = None) -> None`.
- Consumes: `settings.LANGFUSE_ENABLED`.

- [ ] **Step 1: Write the failing offline guard tests.**

```python
# tests/observability/test_langfuse_helpers.py
from src.agentrag.common.langfuse_client import observe_chat_turn, update_turn_trace
from src.agentrag.config import settings


def test_observe_is_passthrough_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", False)

    async def f(x):
        return x + 1

    wrapped = observe_chat_turn(f)
    assert wrapped is f  # no wrapping, zero overhead when off


def test_update_turn_trace_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", False)
    # Must not raise and must return None even with no Langfuse server.
    assert update_turn_trace(name="q", session_id="conv-1") is None
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `uv run pytest tests/observability/test_langfuse_helpers.py -q`
Expected: FAIL — `ImportError: cannot import name 'observe_chat_turn'`.

- [ ] **Step 3: Add the helpers to `langfuse_client.py`** (after `make_async_openai`):

```python
def observe_chat_turn(fn):
    """Decorator: when Langfuse is on, wrap fn in @observe() so nested LLM
    generations group under one trace per call. No-op passthrough when off
    (decided at decoration time — the flag is set from .env at import)."""
    if not settings.LANGFUSE_ENABLED:
        return fn
    try:
        from langfuse.decorators import observe
    except Exception as exc:  # langfuse not installed
        log.warning("LANGFUSE_ENABLED but observe import failed (%s); untraced", type(exc).__name__)
        return fn
    return observe(name="chat_turn")(fn)


def update_turn_trace(*, name=None, session_id=None, metadata=None) -> None:
    """Set attrs on the current Langfuse trace (inside an observed fn). No-op when off."""
    if not settings.LANGFUSE_ENABLED:
        return
    try:
        from langfuse.decorators import langfuse_context

        langfuse_context.update_current_trace(name=name, session_id=session_id, metadata=metadata)
    except Exception as exc:
        log.debug("langfuse update_current_trace skipped: %s", exc)
```

- [ ] **Step 4: Run the tests to verify they pass.**

Run: `uv run pytest tests/observability/test_langfuse_helpers.py -q`
Expected: PASS (2 passed). No langfuse server needed (both paths are the disabled branch).

- [ ] **Step 5: Wire into `GraphAgentService.chat`.** In `src/agentrag/agent/graph_service.py`, add the import near the top of the file (with the other `from src.agentrag...` imports):

```python
from src.agentrag.common.langfuse_client import observe_chat_turn, update_turn_trace
```

Decorate the method and set trace attrs as its first statement (line ~499):

```python
    @observe_chat_turn
    async def chat(
        self,
        question: str,
        document_title: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        conversation_id: str | None = None,
        domain_filter: dict[str, Any] | None = None,
        verbosity: str | None = None,
    ) -> dict[str, Any]:
        update_turn_trace(name=(question or "")[:80], session_id=conversation_id)
```

(Leave the rest of the method body unchanged — `update_turn_trace` is a no-op when Langfuse is off, so this is safe with the flag disabled.)

- [ ] **Step 6: Confirm import + no regression (Langfuse still OFF here).**

Run: `uv run python -c "from src.agentrag.agent.graph_service import GraphAgentService; print('import ok')"`
Expected: `import ok`.
Run: `uv run pytest -q --ignore=tests/ontology --ignore=tests/ingestion`
Expected: PASS (prior green + 2 new).

- [ ] **Step 7: Commit.**

```bash
git add src/agentrag/common/langfuse_client.py src/agentrag/agent/graph_service.py \
        tests/observability/test_langfuse_helpers.py
git commit -m "feat(observability): Langfuse per-turn trace grouping (guarded, default off)

observe_chat_turn decorator + update_turn_trace helper in the langfuse_client
chokepoint; GraphAgentService.chat wrapped so each /chat turn is one trace
(name=question, session=conversation_id) with nested LLM generations. No-op
passthrough when LANGFUSE_ENABLED is false. Offline guard tests."
```

---

### Task 2: ops enablement + live verification

**Files:** `.env` (gitignored — not committed).
No code. Requires the local stack (Docker) running.

- [ ] **Step 1: Bring up the Langfuse containers.**

Run: `docker compose up -d langfuse langfuse-db`
Expected: both become healthy; UI reachable. Verify: `curl -sf http://localhost:3002/api/public/health` → `200`/OK (retry for ~30s on first boot while it migrates).

- [ ] **Step 2: Enable Langfuse in `.env`.** Append (or set) these four lines:

```
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://localhost:3002
LANGFUSE_PUBLIC_KEY=pk-lf-agentrag-dev
LANGFUSE_SECRET_KEY=sk-lf-agentrag-dev
```

- [ ] **Step 3: (Re)start the API so the lifespan picks up the creds.**

Run: `make up-bg` (starts api+worker+frontend in background) — or restart an already-running API.
Verify the API logged it: `grep -i "Langfuse tracing enabled" .run/api.log` → matches `→ http://localhost:3002`.

- [ ] **Step 4: Send a chat and confirm a trace.**

Send one `/chat` (via the UI at `http://localhost:3000`, or `curl` the chat endpoint with a question + `conversation_id`). Then open `http://localhost:3002` → Traces. Expect: **one trace named by the question**, with `session_id = <conversation_id>` (Sessions view groups the conversation), containing nested LLM generations (classify/decide/answer).

- [ ] **Step 5: Record the verification.** Append a short "Langfuse online — verified" note (date, trace screenshot/URL or the trace id observed) to `docs/CHANGELOG-2026-06-25.md` or a new `docs/eval/langfuse-online-2026-06-25.md`. Commit that doc only (`.env` stays gitignored).

```bash
git add docs/eval/langfuse-online-2026-06-25.md
git commit -m "docs: Langfuse online verified — per-turn traces flowing at :3002"
```

---

## Self-Review

**Spec coverage:** helpers (`observe_chat_turn`/`update_turn_trace`) → T1 Step 3 + tests Step 1; wiring `GraphAgentService.chat` → T1 Step 5; ops (.env + containers + restart) → T2 Steps 1–3; live trace-per-/chat verification → T2 Step 4. Error-handling guard → Step 3 (`LANGFUSE_ENABLED` + try/except). All spec sections mapped.

**Placeholder scan:** none — full helper code, exact `.env` lines, exact commands + expected output. (`docs/eval/langfuse-online-...md` content is a verification note authored at T2 Step 5 from the observed run, not a code placeholder.)

**Type consistency:** `observe_chat_turn(fn) -> fn|wrapped` and `update_turn_trace(*, name, session_id, metadata)` signatures identical across the helper def (T1 Step 3), the wiring call (T1 Step 5), and the tests (T1 Step 1). Trace key `session_id=conversation_id` consistent in spec, helper, and wiring.
