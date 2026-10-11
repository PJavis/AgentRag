"""Calls that resolve a routed client directly must still label the cost-ledger
entry with their task, not the client method's default ("json"/"text"/"stream")."""
from __future__ import annotations

import asyncio
import time

from src.agentrag.agent import graph_service as gs
from src.agentrag.agent.service import AgentService


class _Client:
    def __init__(self):
        self.tasks: list[str] = []

    async def json_response(self, system_prompt, user_prompt, task="json", **kw):
        self.tasks.append(task)
        return {"multi_step": False, "subqueries": []}

    async def text_response(self, system_prompt, user_prompt, task="text", **kw):
        self.tasks.append(task)
        return "Xin chào!"


class _Gateway:
    def __init__(self, client):
        self.client = client

    def _resolve_client(self, task, content=None):
        return self.client


def test_plan_call_is_labelled_plan():
    client = _Client()
    svc = AgentService.__new__(AgentService)
    svc.llm_gateway = _Gateway(client)
    asyncio.run(svc._plan_subqueries("Câu hỏi nhiều bước về điều trị?", None))
    assert client.tasks == ["plan"]


def test_graph_chitchat_call_is_labelled_classify(monkeypatch):
    client = _Client()
    monkeypatch.setattr(gs._INNER, "llm_gateway", _Gateway(client))
    asyncio.run(gs.chitchat_answer({"question": "chào bạn", "total_started": time.perf_counter()}))
    assert client.tasks == ["classify"]
