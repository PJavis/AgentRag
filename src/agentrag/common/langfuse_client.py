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


def make_async_openai(**kwargs: Any) -> AsyncOpenAI:
    """Construct an AsyncOpenAI client, Langfuse-traced when enabled."""
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
