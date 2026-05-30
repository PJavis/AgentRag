"""Arize Phoenix tracing — local OTEL trace + eval UI, alongside Langfuse.

When `PHOENIX_ENABLED`, `init_phoenix()` registers an OTEL tracer pointed at the
local Phoenix collector and auto-instruments the OpenAI client via
OpenInference. Every LLM call (built through `make_async_openai`) is then traced
to Phoenix in addition to whatever Langfuse captures. Disabled by default →
zero overhead. Lazy imports so `arize-phoenix` is only needed when enabled.
"""
from __future__ import annotations

import logging

from src.agentrag.config import settings

log = logging.getLogger(__name__)

_REGISTERED = False


def init_phoenix() -> None:
    """Register the Phoenix OTEL tracer + OpenAI auto-instrumentation (once)."""
    global _REGISTERED
    if not settings.PHOENIX_ENABLED or _REGISTERED:
        return
    try:
        from phoenix.otel import register

        register(
            project_name=settings.PHOENIX_PROJECT,
            endpoint=settings.PHOENIX_COLLECTOR_ENDPOINT,
            auto_instrument=True,  # instruments installed OpenInference libs (openai)
        )
        _REGISTERED = True
        log.info("Phoenix tracing enabled → %s", settings.PHOENIX_COLLECTOR_ENDPOINT)
    except Exception as exc:  # arize-phoenix not installed / collector unreachable
        log.warning("Phoenix init skipped (%s): %s", type(exc).__name__, exc)
