# Langfuse online — verified 2026-06-25

P2.8 "live trace per /chat" — enabled and verified.

## Setup
- `docker compose up -d langfuse langfuse-db` → `agentrag-langfuse` on host **:3002**
  (auto-provisioned org/project + dev keys), `/api/public/health` → 200.
- `.env`: `LANGFUSE_ENABLED=true`, `LANGFUSE_HOST=http://localhost:3002`,
  `LANGFUSE_PUBLIC_KEY=pk-lf-agentrag-dev`, `LANGFUSE_SECRET_KEY=sk-lf-agentrag-dev`.
- Code: `observe_chat_turn` + `update_turn_trace` (commit `a6de078`); `GraphAgentService.chat`
  decorated → one trace per turn (`name=question`, `session_id=conversation_id`).

## Verification
Ran one in-process `agent.chat("Nhồi máu cơ tim cấp điều trị thế nào?",
conversation_id="lf-verify-1")` with Langfuse on, then flushed. Queried the traces API:

```
GET /api/public/traces?sessionId=lf-verify-1
→ traces: 1
  - "Nhồi máu cơ tim cấp điều trị thế nào?"  | session: lf-verify-1 | observations: 8
```

One named trace per `/chat`, grouped by `conversation_id` (Langfuse session), with the
turn's 8 LLM generations nested. ✅

## Notes
- Guarded + default-OFF in `.env.example`; a fresh deploy is untraced until the 4 vars
  are set. The app is unchanged when `LANGFUSE_ENABLED=false`.
- `user_id` is not propagated to the agent layer yet (deferred).

## Feedback → score — verified (commit `568035a`)
`chat()` now returns its `langfuse_trace_id`; non-stream handlers persist it on the
assistant turn's `extra_metadata`; `/chat/feedback` loads the turn and scores
`user_feedback=rating` on that trace. Verified in-process:

```
chat result langfuse_trace_id = 20ec2068-d82e-4cd9-a6ef-382918b40a4c
GET /api/public/scores?name=user_feedback
→ user_feedback = 1 | trace: 20ec2068-... | comment: verify
```

The trace-id capture + score attach work end-to-end (Langfuse mechanics). The
`/feedback` DB lookup of `extra_metadata.langfuse_trace_id` is plain guarded SQLAlchemy.
Streaming path (`chat_stream`) feedback→score remains a follow-up.
