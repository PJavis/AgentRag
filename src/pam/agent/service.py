from __future__ import annotations

import json
from typing import Any

from src.pam.agent.context import ContextAssembler
from src.pam.agent.llm import AgentLLM
from src.pam.agent.tools import AgentTools
from src.pam.config import settings


class AgentService:
    def __init__(self):
        self.llm = AgentLLM()
        self.tools = AgentTools()
        self.context = ContextAssembler()

    async def chat(self, question: str, document_title: str | None = None) -> dict[str, Any]:
        tool_trace: list[dict[str, Any]] = []
        final_answer: dict[str, Any] | None = None
        seen_calls: set[str] = set()

        for _ in range(settings.AGENT_MAX_STEPS):
            decision = await self._decide(question, document_title, tool_trace)
            if decision.get("done"):
                final_answer = decision
                break

            tool_name = decision.get("tool_name")
            tool_input = decision.get("tool_input") or {}
            if document_title and "document_title" not in tool_input:
                tool_input["document_title"] = document_title
            if not self.tools.has_tool(tool_name):
                tool_name = "search_hybrid_kg"
                tool_input = {
                    "query": question,
                    "top_k": settings.AGENT_TOOL_TOP_K,
                    "document_title": document_title,
                }
            call_fingerprint = json.dumps(
                {"tool_name": tool_name, "tool_input": tool_input},
                ensure_ascii=True,
                sort_keys=True,
            )
            if call_fingerprint in seen_calls:
                break
            seen_calls.add(call_fingerprint)
            tool_output = await self.tools.call(tool_name, tool_input)
            tool_trace.append(
                {
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_output": tool_output,
                }
            )

        if not tool_trace:
            fallback_input = {
                "query": question,
                "top_k": settings.AGENT_TOOL_TOP_K,
                "document_title": document_title,
            }
            tool_trace.append(
                {
                    "tool_name": "search_hybrid_kg",
                    "tool_input": fallback_input,
                    "tool_output": await self.tools.call("search_hybrid_kg", fallback_input),
                }
            )

        assembly = self.context.assemble(question, [entry["tool_output"] for entry in tool_trace])
        answer = await self._answer(question, assembly["packed_context"], tool_trace, final_answer)
        grounded_citations = self._ground_citations(answer.get("citations", []), assembly["packed_context"])
        return {
            "question": question,
            "document_title": document_title,
            "tool_trace": tool_trace,
            "context": assembly["packed_context"],
            "answer": answer.get("answer", ""),
            "citations": grounded_citations,
        }

    async def _decide(
        self,
        question: str,
        document_title: str | None,
        tool_trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system_prompt = (
            "You are a retrieval agent. Decide one next tool call at a time. "
            "Always return JSON with keys: done, tool_name, tool_input, reason. "
            "If enough evidence exists, set done=true and do not request another tool."
        )
        user_prompt = json.dumps(
            {
                "question": question,
                "document_title": document_title,
                "available_tools": self.tools.describe(),
                "tool_trace": tool_trace,
            },
            ensure_ascii=True,
        )
        decision = await self.llm.json_response(system_prompt, user_prompt)
        if decision.get("done"):
            return decision
        tool_name = decision.get("tool_name") or "search_hybrid_kg"
        tool_input = decision.get("tool_input") or {"query": question, "top_k": settings.AGENT_TOOL_TOP_K}
        return {
            "done": False,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "reason": decision.get("reason"),
        }

    def _ground_citations(
        self,
        citations: list[dict[str, Any]],
        packed_context: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        allowed = {
            (
                item.get("document_title"),
                item.get("section_path"),
                item.get("position"),
                item.get("content_hash"),
            )
            for item in packed_context
        }
        grounded: list[dict[str, Any]] = []
        for citation in citations:
            key = (
                citation.get("document_title"),
                citation.get("section_path"),
                citation.get("position"),
                citation.get("content_hash"),
            )
            if key in allowed:
                grounded.append(
                    {
                        "document_title": citation.get("document_title"),
                        "section_path": citation.get("section_path"),
                        "position": citation.get("position"),
                        "content_hash": citation.get("content_hash"),
                    }
                )
        return grounded

    async def _answer(
        self,
        question: str,
        packed_context: list[dict[str, Any]],
        tool_trace: list[dict[str, Any]],
        final_answer: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if final_answer and final_answer.get("answer"):
            return {
                "answer": final_answer["answer"],
                "citations": final_answer.get("citations", []),
            }

        system_prompt = (
            "Answer only from the provided context. "
            "Return JSON with keys: answer, citations. "
            "Each citation must include document_title, section_path, position, content_hash. "
            "If context is insufficient, say so explicitly. "
            "Use the same language as the question. "
            "Answer in clear, natural sentences and avoid broken wording. "
            "Only cite claims that are directly supported by the provided context."
        )
        user_prompt = json.dumps(
            {
                "question": question,
                "context": packed_context,
                "tool_trace_summary": [
                    {"tool_name": step["tool_name"], "tool_input": step["tool_input"]}
                    for step in tool_trace
                ],
            },
            ensure_ascii=True,
        )
        return await self.llm.json_response(system_prompt, user_prompt)
