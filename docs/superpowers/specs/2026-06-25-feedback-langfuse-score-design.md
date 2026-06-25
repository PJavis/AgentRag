# Feedback → Langfuse score — design

**Date:** 2026-06-25 · **Objective:** push each thumbs 👍/👎 as a Langfuse score on
the turn's trace, so low-rated turns are filterable in the trace UI (online quality
monitoring). Connects the feedback pipeline to the now-live per-turn traces.

## Context

- Feedback: `FeedbackButtons.tsx` → `POST /chat/feedback` → upsert `AdapterChatFeedback`
  (`adapter/routers/chat.py::submit_chat_feedback`, ~1444). Body has `turn_id`
  (assistant `ChatMessage.id`), `rating` ±1, `session_id`.
- Langfuse online (2026-06-25): `GraphAgentService.chat` is `@observe_chat_turn`, one
  trace per turn (`name=question`, `session_id=conversation_id`). Verified live.
- Verified SDK facts: `update_current_trace` has no `id` param; `langfuse_context.
  get_current_trace_id()` returns the active trace id inside an observed fn;
  `Langfuse().score(trace_id=, name=, value=, data_type=)` attaches a score to any trace.

The feedback arrives in a *later* request, so we must carry the trace id from chat-time
to feedback-time. **Approach B (chosen):** capture `get_current_trace_id()` inside the
observed `chat`, return it, persist it on the assistant turn's `extra_metadata`; the
feedback endpoint reads it and scores. (No id pre-generation / no `append_message`
signature change — lower risk than threading `langfuse_observation_id`.)

## Scope

In: 2 guarded helpers; `chat()` returns `langfuse_trace_id`; the **non-stream**
assistant-persist sites (`execute_chat`, `regenerate_chat`) store it in
`extra_metadata`; `/chat/feedback` scores the trace. Offline helper tests + live verify.

Out (documented follow-up): the **streaming** path (`chat_stream` / `execute_chat_stream`)
— capturing a trace id out of an async SSE generator is a separate integration;
`user_id` on the score.

## Design

### Helpers (`src/agentrag/common/langfuse_client.py`)
- `current_trace_id() -> str | None` — when `LANGFUSE_ENABLED`, return
  `langfuse_context.get_current_trace_id()` (try/except → None); else None.
- `score_trace(trace_id, *, name, value, comment=None) -> None` — when enabled and
  `trace_id` truthy, `Langfuse().score(trace_id=trace_id, name=name, value=value,
  data_type="NUMERIC", comment=comment)` in try/except (debug-log on failure); else no-op.

### Capture (`src/agentrag/agent/graph_service.py`)
- In `GraphAgentService.chat` (single return, ~544) add `"langfuse_trace_id":
  current_trace_id()` to the returned dict. (Called inside the `@observe_chat_turn`
  context, so the id is live.)

### Persist (`src/agentrag/adapter/routers/chat.py`)
- `execute_chat` assistant `append_message(... extra_metadata={...})` (~450) and
  `regenerate_chat` (~688): add `"langfuse_trace_id": result.get("langfuse_trace_id")`
  to the `extra_metadata` dict.

### Score (`/chat/feedback`, ~1444)
- After the upsert, when `settings.LANGFUSE_ENABLED`: load
  `ChatMessage` where `id == turn_id` (UUID), read
  `extra_metadata.get("langfuse_trace_id")`, and if present call
  `score_trace(trace_id, name="user_feedback", value=float(rating_int), comment=comment)`.
  Wrapped in try/except so feedback never fails on a Langfuse hiccup.

## Data flow
chat (observed) → `current_trace_id()` → returned + stored on assistant
`extra_metadata.langfuse_trace_id` → later `/chat/feedback` loads the message by
`turn_id` → `score_trace(trace_id, "user_feedback", rating)` → score on the trace in
Langfuse UI.

## Error handling
All Langfuse touchpoints guarded by `LANGFUSE_ENABLED` + try/except. Feedback upsert
(the training-data write) happens first and is never blocked by the scoring step.

## Testing
- **Offline unit** (`tests/observability/test_langfuse_helpers.py`, extend): with
  `LANGFUSE_ENABLED=False`, `current_trace_id() is None`; `score_trace("t", name="x",
  value=1.0) is None` and does not raise.
- **Live** (stack up): set a 👍 via `/chat/feedback` on a turn whose trace exists, then
  query `GET /api/public/scores` (or the trace) at `:3002` and confirm a `user_feedback`
  score on that trace id.
