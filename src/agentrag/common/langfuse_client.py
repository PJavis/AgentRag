"""Langfuse tracing — single chokepoint for OpenAI client construction.

Every AsyncOpenAI client in the codebase is built via `make_async_openai()`.
When `LANGFUSE_ENABLED` is on, this returns the `langfuse.openai` drop-in
subclass, which auto-traces every `chat.completions.create` to the configured
Langfuse project. When off (or when langfuse is not installed), it returns the
plain `openai.AsyncOpenAI` so the app runs unchanged.

The cost ledger (`observability/cost.py`) is intentionally kept independent —
Langfuse gives the trace/latency UI; the ledger gives the USD aggregates.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from openai import AsyncOpenAI

from src.agentrag.config import settings

log = logging.getLogger(__name__)


def init_langfuse() -> None:
    """Export Langfuse credentials to the environment so the SDK picks them up.

    Pydantic loads keys from `.env` into `settings` but does NOT populate
    `os.environ`; the Langfuse SDK reads only `os.environ`. Call once at app
    startup (FastAPI lifespan) before any traced LLM call fires.
    """
    if not settings.LANGFUSE_ENABLED:
        return
    if settings.LANGFUSE_PUBLIC_KEY:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.LANGFUSE_PUBLIC_KEY)
    if settings.LANGFUSE_SECRET_KEY:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.LANGFUSE_SECRET_KEY)
    os.environ.setdefault("LANGFUSE_HOST", settings.LANGFUSE_HOST)
    log.info("Langfuse tracing enabled → %s", settings.LANGFUSE_HOST)


def _with_timeout_default(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Inject a per-request timeout if the caller didn't set one. Without it a
    stalled provider connection (gemini 503/high-demand) hangs the call forever —
    a 42-min agent.chat hang was observed (2026-06-26). The openai SDK then retries
    on timeout instead of blocking."""
    kwargs.setdefault("timeout", settings.LLM_REQUEST_TIMEOUT_S)
    return kwargs


def make_async_openai(**kwargs: Any) -> AsyncOpenAI:
    """Construct an AsyncOpenAI client, Langfuse-traced when enabled."""
    kwargs = _with_timeout_default(kwargs)
    if not settings.LANGFUSE_ENABLED:
        return AsyncOpenAI(**kwargs)
    try:
        from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI
    except Exception as exc:  # langfuse not installed
        log.warning(
            "LANGFUSE_ENABLED but langfuse import failed (%s); using untraced client",
            type(exc).__name__,
        )
        return AsyncOpenAI(**kwargs)
    return LangfuseAsyncOpenAI(**kwargs)


def _content_or_none(value):
    """Privacy gate for free-text (question/answer/comment). Returns the value only
    when OBSERVABILITY_CAPTURE_CONTENT is on; else None so PHI never reaches the trace
    store. Default off."""
    return value if settings.OBSERVABILITY_CAPTURE_CONTENT else None


def observe_chat_turn(fn):
    """Decorator: when Langfuse is on, wrap fn in @observe() so nested LLM
    generations group under one trace per call. No-op passthrough when off
    (decided at decoration time — the flag is set from .env at import). Input/output
    text is captured only when OBSERVABILITY_CAPTURE_CONTENT is on (PHI-safe default)."""
    if not settings.LANGFUSE_ENABLED:
        return fn
    try:
        from langfuse.decorators import observe
    except Exception as exc:  # langfuse not installed
        log.warning("LANGFUSE_ENABLED but observe import failed (%s); untraced", type(exc).__name__)
        return fn
    cap = settings.OBSERVABILITY_CAPTURE_CONTENT
    return observe(name="chat_turn", capture_input=cap, capture_output=cap)(fn)


def update_turn_trace(*, name=None, session_id=None, metadata=None) -> None:
    """Set attrs on the current Langfuse trace (inside an observed fn). No-op when off."""
    if not settings.LANGFUSE_ENABLED:
        return
    try:
        from langfuse.decorators import langfuse_context

        langfuse_context.update_current_trace(name=_content_or_none(name), session_id=session_id, metadata=metadata)
    except Exception as exc:
        log.debug("langfuse update_current_trace skipped: %s", exc)


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
                         data_type="NUMERIC", comment=_content_or_none(comment))
    except Exception as exc:
        log.debug("langfuse score skipped: %s", exc)


def langfuse_flush() -> None:
    """Flush buffered traces. Call on app shutdown so nothing is lost.

    v2 SDK: the OpenAI integration exposes `langfuse.openai.openai.flush_langfuse()`.
    Fall back to a Langfuse() singleton flush if that's unavailable.
    """
    if not settings.LANGFUSE_ENABLED:
        return
    try:
        from langfuse.openai import openai as lf_openai

        lf_openai.flush_langfuse()
        return
    except Exception as exc:
        log.debug("langfuse openai flush unavailable: %s", exc)
    try:
        from langfuse import Langfuse

        Langfuse().flush()
    except Exception as exc:
        log.debug("langfuse flush skipped: %s", exc)
