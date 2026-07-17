# Langfuse observability — runbook (P2-ops slice 1, 2026-07-18)

Self-hosted Langfuse v2 tracing for prod chat turns + user-feedback scores.

## STATUS: NOT enabled in prod — blocked on a PHI leak

`config.py` ships `LANGFUSE_ENABLED=False` and it must stay that way in any
environment holding medical data until the content-leak below is fixed.

**Why:** with `OBSERVABILITY_CAPTURE_CONTENT=false`, the trace-level span is clean,
but the per-LLM-call **GENERATION** observations auto-captured by the
`langfuse.openai` drop-in still record the full question text, tool-call JSON, and
retrieved passages. Verified live (2026-07-18): a traced turn wrote the question
token into 9/9 generations. The intended fix — langfuse's `langfuse_mask` /
generation redaction — does **not** reliably apply in langfuse v2.60: the
integration caches its client across three layers (a global `openai` monkeypatch,
`LangfuseSingleton`, and per-client `initialize()`), so setting the mask attribute
after import (as `init_langfuse` must) does not reach the client that ingests
events (`task_manager._mask` stays `None`). Two fix attempts (set-attr, then
singleton `reset()`) both still leaked. Tracked as a follow-up.

**Do NOT enable Langfuse on a corpus with PHI until a live re-verify shows
generation `input`/`output` redacted.**

## What IS verified working (2026-07-18)
- Tracing: a chat turn produces a Langfuse trace (metadata: latency, tokens, tool
  trace, rerank scores) — the trace-level span honours `OBSERVABILITY_CAPTURE_CONTENT`.
- Feedback: `POST /on/api/feedback {turn_id, rating:+1}` (and `score_trace(...)`)
  attach a `user_feedback` NUMERIC score to the turn's trace.
- Graceful degradation: with the `langfuse` container stopped, a chat turn still
  succeeds — every `langfuse_client` call is best-effort. Locked by
  `tests/observability/test_langfuse_helpers.py` (enabled-but-unreachable swallow
  tests).

## Bring-up (for the eventual enable, after the leak is fixed)
1. Install the extra (NOTE: include every extra you use — `uv sync --extra X`
   reconciles the venv to *exactly* the named extras and will uninstall others):
   `uv sync --extra observability --extra deepeval`
2. `docker compose --profile observability up -d langfuse-db langfuse`
   (Langfuse v2 + its Postgres; project keys bootstrapped headlessly via
   `LANGFUSE_INIT_*`). **Host port is 3002** (`http://localhost:3002`) — 3000 is the
   Next.js frontend.
3. In `.env` (per-deployment opt-in; NOT config.py): `LANGFUSE_ENABLED=true`,
   `OBSERVABILITY_CAPTURE_CONTENT=false`, `LANGFUSE_HOST=http://localhost:3002`.
4. Restart the API (`make api`) — `main.py` lifespan calls `init_langfuse()`.

## Verify (the check that currently FAILS on PHI — must pass before prod-enable)
1. Boot: API logs `Langfuse tracing enabled → <host>`, no error.
2. Trace: `GET /api/public/traces` shows a recent trace; trace `input`/`output` empty.
3. **Generations (the blocker):** `GET /api/public/observations?limit=20` → filter
   `type=GENERATION` → their `input`/`output` MUST NOT contain question/answer/passage
   text. Currently they DO — this is the open leak.
4. Score: `GET /api/public/scores` shows a `user_feedback` NUMERIC score.
5. Degradation: stop the `langfuse` container, drive a turn → still succeeds.

## Guarantees today
- Default off: `config.py LANGFUSE_ENABLED=False` — nothing traces unless a
  deployment opts in via `.env`. Safe by default.
- Safety: Langfuse downtime never breaks chat (best-effort client; regression-tested).

## Follow-up (before this can ship enabled)
Solve generation redaction in langfuse v2 — options to investigate: build the
`LangfuseSingleton` explicitly with `mask=` inside `init_langfuse` before any client
initialises; pin/patch the integration; or move to a langfuse version whose masking
applies reliably. Then re-run the step-3 generation check and confirm redaction.

## Not in this slice
Automated online-eval judges on sampled turns, and flywheel scheduling
(`mine_citation_pairs_prod` + FT) — separate later slices.
