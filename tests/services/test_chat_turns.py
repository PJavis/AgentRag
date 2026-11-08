"""Every chat entry point shares one turn pipeline.

Only the root routes used to enqueue `chat_memory`, so with
CHAT_STRUCTMEM_ENABLED the UI (adapter routes only) never wrote chat memory.
History windows had drifted to 20 / 10 / CHAT_HISTORY_WINDOW.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import uuid
from types import SimpleNamespace

import pytest

from src.agentrag.chat import turns
from src.agentrag.config import settings


class FakePool:
    def __init__(self, fail: bool = False):
        self.jobs: list[tuple[str, dict]] = []
        self.fail = fail

    async def enqueue_job(self, name, **kwargs):
        if self.fail:
            raise ConnectionError("redis down")
        self.jobs.append((name, kwargs))
        return SimpleNamespace(job_id="j1")


@pytest.fixture
def pool(monkeypatch):
    p = FakePool()
    import src.agentrag.worker.pool as pool_mod

    monkeypatch.setattr(pool_mod, "get_pool", lambda: p)
    monkeypatch.setattr(settings, "CHAT_STRUCTMEM_ENABLED", True)
    return p


async def test_record_turn_memory_enqueues_chat_memory(pool):
    assert await turns.record_turn_memory("c1", "q?", "answer") is True
    [(name, kw)] = pool.jobs
    assert name == "chat_memory"
    assert kw["conversation_id"] == "c1"
    assert kw["user_message"] == "q?" and kw["assistant_message"] == "answer"
    uuid.UUID(kw["turn_id"])
    assert kw["turn_timestamp"].endswith("+00:00")


@pytest.mark.parametrize(
    "question,answer,failed",
    [("q", "", False), ("q", "   ", False), ("", "a", False), ("q", "⚠️ Agent failed: X", True)],
)
async def test_no_memory_for_empty_or_failed_turns(pool, question, answer, failed):
    assert await turns.record_turn_memory("c1", question, answer, failed=failed) is False
    assert pool.jobs == []


async def test_disabled_flag_skips(pool, monkeypatch):
    monkeypatch.setattr(settings, "CHAT_STRUCTMEM_ENABLED", False)
    assert await turns.record_turn_memory("c1", "q", "a") is False
    assert pool.jobs == []


async def test_enqueue_failure_does_not_raise(monkeypatch):
    # The answer is already persisted; losing memory must not 500 the turn.
    import src.agentrag.worker.pool as pool_mod

    monkeypatch.setattr(pool_mod, "get_pool", lambda: FakePool(fail=True))
    monkeypatch.setattr(settings, "CHAT_STRUCTMEM_ENABLED", True)
    assert await turns.record_turn_memory("c1", "q", "a") is False


async def test_load_history_uses_configured_window(monkeypatch):
    seen = {}

    class Store:
        async def list_messages(self, cid, limit):
            seen["args"] = (cid, limit)
            return []

    monkeypatch.setattr(settings, "CHAT_HISTORY_WINDOW", 7)
    await turns.load_history(Store(), "c9")
    assert seen["args"] == ("c9", 7)


async def test_attach_follow_ups_skips_without_message():
    assert await turns.attach_follow_ups(object(), "c", None, ["x"]) is False
    assert await turns.attach_follow_ups(object(), "c", {"message_id": "m"}, []) is False


# ── wiring: each chat route goes through the shared helpers ────────────────────

def _calls(module: str, func: str) -> set[str]:
    fn = getattr(importlib.import_module(module), func)
    tree = ast.parse(inspect.cleandoc("\n" + inspect.getsource(fn)))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


CHAT = "src.agentrag.adapter.routers.chat"


@pytest.mark.parametrize(
    "module,func",
    [("main", "chat"), ("main", "chat_stream"), (CHAT, "execute_chat"),
     (CHAT, "execute_chat_stream"), (CHAT, "send_message")],
)
def test_turn_routes_record_memory_and_window_history(module, func):
    calls = _calls(module, func)
    assert "record_turn_memory" in calls
    assert "load_history" in calls
    assert "enqueue_job" not in calls  # no private copy of the enqueue


@pytest.mark.parametrize("func", ["execute_chat", "regenerate_chat", "send_message"])
def test_follow_ups_persist_through_shared_helper(func):
    assert "attach_follow_ups" in _calls(CHAT, func)


def test_regenerate_history_excludes_question_and_is_windowed():
    src = inspect.getsource(importlib.import_module(CHAT).regenerate_chat)
    assert "msgs[:user_idx][-settings.CHAT_HISTORY_WINDOW:]" in src
    assert "msgs[:user_idx + 1]" not in src
