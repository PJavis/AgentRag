# Langfuse online + per-turn traces — design

**Date:** 2026-06-25 · **Objective:** turn on the already-wired Langfuse tracing and
group each `/chat` turn into one named trace (roadmap P2.8 "live trace per /chat").

## Context

Langfuse is fully wired in code: `common/langfuse_client.py::make_async_openai`
returns the langfuse-traced OpenAI client when `LANGFUSE_ENABLED`; every LLM client
(`agent/llm.py`, `retrieval/reranker.py`, `services/llm_gateway.py`) goes through it;
`main.py` lifespan calls `init_langfuse()` / `langfuse_flush()`; `langfuse>=2,<3` is a
dep (installed 2.60.10); `docker-compose` has a `langfuse` + `langfuse-db` service on
host port **3002** (auto-provisions org/project + keys `pk-lf-agentrag-dev` /
`sk-lf-agentrag-dev`).

Two gaps: (1) the live `.env` has no `LANGFUSE_*` settings (so it's off); (2) no
explicit trace grouping — each LLM call is its own scattered generation, not one
trace per `/chat` turn.

## Scope

In: two guarded helpers + one decorator application + ops enablement + offline guard
tests + live verification. Out: feedback→Langfuse score; `user_id` propagation to the
agent layer.

## Design

### Helpers (`src/agentrag/common/langfuse_client.py`)
- `observe_chat_turn(fn)` — decorator. When `settings.LANGFUSE_ENABLED` is False,
  return `fn` unchanged (no import, no overhead). When True, return
  `langfuse.decorators.observe(name="chat_turn")(fn)` so nested langfuse-openai
  generations group under one trace. On langfuse import failure, return `fn`.
- `update_turn_trace(*, name=None, session_id=None, metadata=None) -> None` — when
  enabled, `from langfuse.decorators import langfuse_context;
  langfuse_context.update_current_trace(name=name, session_id=session_id,
  metadata=metadata)` inside try/except (debug-log on failure). No-op when off.

### Wiring (`src/agentrag/agent/graph_service.py`)
- Decorate `GraphAgentService.chat` (line ~499) with `@observe_chat_turn`.
- First statement inside: `update_turn_trace(name=(question or "")[:80],
  session_id=conversation_id)`.
- This is the single entry used by `execute_chat`, `regenerate_chat`,
  `execute_chat_stream`, and the eval scripts — all inherit per-turn grouping.

### Ops (`.env` + containers)
- `.env` add: `LANGFUSE_ENABLED=true`, `LANGFUSE_HOST=http://localhost:3002`,
  `LANGFUSE_PUBLIC_KEY=pk-lf-agentrag-dev`, `LANGFUSE_SECRET_KEY=sk-lf-agentrag-dev`.
- `docker compose up -d langfuse langfuse-db`.
- Restart API so lifespan `init_langfuse()` exports creds.

## Data flow
`/chat` → `GraphAgentService.chat` (now `@observe_chat_turn`) → `update_turn_trace`
sets trace name + session → nested LLM generations (via `make_async_openai`) attach →
flushed to Langfuse at `:3002`. Grouped by `session_id=conversation_id` (Langfuse
sessions = one conversation's turns).

## Error handling
All langfuse touchpoints guarded by `LANGFUSE_ENABLED` + try/except so the app is
unchanged when off and never breaks a chat if Langfuse is unreachable.

## Testing
- **Offline unit** (`tests/observability/test_langfuse_helpers.py`): with
  `LANGFUSE_ENABLED=False`, `observe_chat_turn(fn) is fn` and calling the returned fn
  works unchanged; `update_turn_trace(name="x", session_id="s")` returns None without
  raising. (No langfuse server needed.)
- **Live** (stack is up): `docker compose up -d langfuse langfuse-db`; set `.env`;
  restart API; send a `/chat`; confirm at `http://localhost:3002` one trace named by
  the question with `session_id=<conversation_id>` and nested generations.
