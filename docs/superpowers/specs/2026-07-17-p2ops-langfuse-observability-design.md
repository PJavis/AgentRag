# P2-ops slice 1 — Enable + verify self-hosted Langfuse observability (2026-07-17)

## Why

The eval-trust campaign shipped a system at the eval ceiling, but there is **no live
signal** for prod quality: the tracing code exists yet `LANGFUSE_ENABLED=False`, so no
prod turn is observed. P2-ops turns that on. This is **slice 1** — observability only.
Automated online-eval and flywheel scheduling are deliberately deferred to later slices.

## Goal / success criteria

Live tracing of prod chat turns + user-feedback scores in the self-hosted Langfuse,
**PHI-safe** (no question/answer text), proven end-to-end. Done when:

1. The `observability` docker profile is up and the app connects to Langfuse at boot.
2. A real chat turn produces a Langfuse trace carrying metadata only (latency, token
   counts, tool trace, rerank scores) — **no question/answer text**.
3. A `POST /feedback` thumbs rating attaches a `user_feedback` score to that turn's trace.
4. Langfuse being down does **not** break chat (best-effort/no-op), and traces flush on
   shutdown.

## Non-goals (later slices)

- Automated online-eval judges on sampled prod turns.
- Scheduling the flywheel (`mine_citation_pairs_prod` + FT) via ARQ/cron.
- Capturing question/answer content (`OBSERVABILITY_CAPTURE_CONTENT` stays `False`).
- Phoenix (separate client, out of scope here).

## What already exists (no code change needed)

- `src/agentrag/common/langfuse_client.py` — `init_langfuse()`, `langfuse_flush()`,
  `observe_chat_turn`, `update_turn_trace`, `score_trace()`, and the PHI gate
  `_content_or_none` driven by `OBSERVABILITY_CAPTURE_CONTENT`.
- Wiring in `agent/llm.py`, `agent/graph_service.py`, `services/llm_gateway.py`,
  `adapter/routers/chat.py` (feedback endpoint mirrors the rating via `score_trace`,
  `chat.py:1491–1507`).
- `docker-compose.yml` `observability` profile: Langfuse v2 + `langfuse-db`, headless key
  bootstrap via `LANGFUSE_INIT_*`.
- `config.py`: `LANGFUSE_ENABLED=False`, `LANGFUSE_PUBLIC_KEY/SECRET_KEY` (present in
  `.env`), `LANGFUSE_HOST`, `OBSERVABILITY_CAPTURE_CONTENT=False`.

The slice is therefore **configuration + verification + a runbook**, not a build.

## Design

### Configuration
- Enable via **`.env`** in this environment: `LANGFUSE_ENABLED=true`,
  `OBSERVABILITY_CAPTURE_CONTENT=false`, `LANGFUSE_HOST` matching where the app runs
  (`http://langfuse:3000` in-compose-network, `http://localhost:3000` host-side).
- **`config.py` default stays `LANGFUSE_ENABLED=False`.** Observability depends on the
  profile being up; shipping it on by default would trace-to-nothing (or log noise) wherever
  the profile isn't running. Opt-in per deployment via `.env` matches the existing
  flags-off-by-default pattern.

### Data flow (unchanged, just switched on)
```
chat turn ─▶ graph_service (observe_chat_turn + update_turn_trace)
          ─▶ llm_gateway / llm.py (langfuse.openai auto-trace, metadata only)
          ─▶ Langfuse trace  (latency, tokens, tool_trace, rerank; NO content)
POST /feedback ─▶ AdapterChatFeedback (Postgres)  +  score_trace(user_feedback) ─▶ same trace
```

### Ops hardening (light, in scope)
- **Graceful degradation:** verify `langfuse_client` no-ops when `LANGFUSE_ENABLED=false`
  and when the host is unreachable — chat must succeed regardless. This is the one behaviour
  to actively confirm (a hang or exception here would take down prod chat).
- **Flush on shutdown:** confirm `langfuse_flush()` is in the app lifespan so buffered
  traces aren't lost on restart.

### Runbook (deliverable)
`docs/ops/langfuse-observability.md`: bring-up command, the four-point verification
checklist above, the PHI note (content off), and the degradation expectation.

## Verification procedure

1. `docker compose --profile observability up -d`; wait for `langfuse` healthy; open
   `http://localhost:3000` (headless-bootstrapped project).
2. Restart the API with `LANGFUSE_ENABLED=true`; confirm `init_langfuse` logs success, no
   boot error.
3. Drive one real chat turn against the API; in Langfuse confirm: a trace exists with
   metadata, and **no question/answer text** anywhere on it (PHI gate).
4. `POST /feedback` with `{turn_id, rating:+1}`; confirm a `user_feedback` score on that
   trace.
5. Stop the Langfuse container, drive another chat turn; confirm the turn still succeeds
   (degradation). Restart Langfuse.

## Risks

- **Content leak:** a wiring path that ignores the PHI gate would put question/answer text
  in traces. Mitigation: step 3 explicitly inspects a trace for absence of content.
- **Chat coupling:** if tracing raised instead of no-op'ing, Langfuse downtime breaks chat.
  Mitigation: step 5 tests it directly.
- **Self-hosted resource cost:** Langfuse v2 + its Postgres add containers/RAM on the WSL
  box. Acceptable for observability; documented in the runbook.
