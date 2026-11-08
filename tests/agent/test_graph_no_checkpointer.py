"""The chat graph keeps no per-conversation state between turns.

An InMemorySaver keyed by conversation_id grew without bound in every gunicorn
worker (one thread per conversation and per anonymous turn) and merged each
new turn into the previous turn's state, so counters such as critique_retries
carried over between turns.
"""
from __future__ import annotations

import inspect


def test_graph_compiled_without_checkpointer():
    from src.agentrag.agent import graph_service

    assert graph_service._GRAPH.checkpointer is None


def test_no_thread_id_config_passed():
    from src.agentrag.agent import graph_service

    src = inspect.getsource(graph_service)
    assert "thread_id" not in src
    assert "langgraph.checkpoint" not in src
