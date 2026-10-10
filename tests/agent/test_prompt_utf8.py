"""LLM prompts carry Vietnamese as raw UTF-8, not \\uXXXX escapes.

json.dumps(..., ensure_ascii=True) turned every accented character into a
six-character escape: 2.22x the DeepSeek tokens for the same answer context
(7,048 vs 3,168 on 8 real chunks, 2026-10-10), and the model read escape codes
instead of Vietnamese.
"""
from __future__ import annotations

import asyncio

from src.agentrag.agent.service import AgentService

_CHUNK = "Sốt xuất huyết điều trị bằng bù dịch và theo dõi tiểu cầu."
_QUESTION = "Điều trị sốt xuất huyết thế nào?"


class _CapturingGateway:
    def __init__(self, reply):
        self.reply = reply
        self.prompts: list[str] = []

    async def json_response(self, *args, **kwargs):
        self.prompts.append(kwargs.get("user_prompt") or (args[1] if len(args) > 1 else ""))
        return self.reply, 0.0

    async def json_response_multimodal(self, *args, **kwargs):
        self.prompts.append(kwargs.get("user_text") or kwargs.get("user_prompt") or "")
        return self.reply, 0.0


def _svc(reply):
    svc = AgentService.__new__(AgentService)
    svc.llm_gateway = _CapturingGateway(reply)
    svc.knowledge = type("K", (), {"describe_tools": staticmethod(lambda: [{"name": "search_hybrid"}])})()
    return svc


def _assert_raw_utf8(prompt: str):
    assert "xuất huyết" in prompt
    assert "\\u" not in prompt


def test_decide_prompt_is_raw_utf8():
    svc = _svc({"done": True, "tool_name": "", "tool_input": {}, "reflection": "", "reason": ""})
    trace = [{"tool_name": "search_hybrid", "tool_input": {"query": _QUESTION},
              "tool_output": {"results": [{"content": _CHUNK, "document_title": "Nhi khoa"}]}}]
    asyncio.run(svc._decide(_QUESTION, None, trace, None))
    _assert_raw_utf8(svc.llm_gateway.prompts[0])


def test_answer_prompt_is_raw_utf8():
    svc = _svc({"answer": "Bù dịch [1].", "citations": [], "highlights": []})
    packed = [{"source": 1, "document_title": "Nhi khoa", "content": _CHUNK, "rerank_score": 0.99}]
    asyncio.run(svc._answer(_QUESTION, packed, [], None, None))
    _assert_raw_utf8(svc.llm_gateway.prompts[0])
