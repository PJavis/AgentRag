from __future__ import annotations

import json
import time
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

    async def chat(
        self,
        question: str,
        document_title: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        tool_trace: list[dict[str, Any]] = []
        final_answer: dict[str, Any] | None = None
        seen_calls: set[str] = set()
        decide_latency_ms = 0.0
        tool_latency_ms = 0.0
        answer_latency_ms = 0.0
        assemble_latency_ms = 0.0

        bootstrap_input = {
            "query": question,
            "top_k": settings.AGENT_TOOL_TOP_K,
            "document_title": document_title,
        }
        bootstrap_fp = json.dumps(
            {"tool_name": "search_hybrid_kg", "tool_input": bootstrap_input},
            ensure_ascii=True,
            sort_keys=True,
        )
        started = time.perf_counter()
        bootstrap_output = await self.tools.call("search_hybrid_kg", bootstrap_input)
        tool_latency_ms += (time.perf_counter() - started) * 1000
        tool_trace.append(
            {
                "tool_name": "search_hybrid_kg",
                "tool_input": bootstrap_input,
                "tool_output": bootstrap_output,
                "tool_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
        seen_calls.add(bootstrap_fp)

        additional_steps = max(min(settings.AGENT_MAX_STEPS - 1, 1), 0)
        for _ in range(additional_steps):
            started = time.perf_counter()
            decision = await self._decide(question, document_title, tool_trace, chat_history)
            decide_elapsed = (time.perf_counter() - started) * 1000
            decide_latency_ms += decide_elapsed
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
            started = time.perf_counter()
            tool_output = await self.tools.call(tool_name, tool_input)
            tool_elapsed = (time.perf_counter() - started) * 1000
            tool_latency_ms += tool_elapsed
            tool_trace.append(
                {
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_output": tool_output,
                    "decision_latency_ms": round(decide_elapsed, 2),
                    "tool_latency_ms": round(tool_elapsed, 2),
                }
            )
        started = time.perf_counter()
        assembly = self.context.assemble(question, [entry["tool_output"] for entry in tool_trace])
        assemble_latency_ms += (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        answer = await self._answer(
            question,
            assembly["packed_context"],
            tool_trace,
            final_answer,
            chat_history,
        )
        answer_latency_ms += (time.perf_counter() - started) * 1000
        grounded_citations = self._ground_citations(answer.get("citations", []), assembly["packed_context"])
        total_latency_ms = (time.perf_counter() - total_started) * 1000
        return {
            "question": question,
            "document_title": document_title,
            "tool_trace": tool_trace,
            "context": assembly["packed_context"],
            "answer": answer.get("answer", ""),
            "citations": grounded_citations,
            "timings_ms": {
                "total": round(total_latency_ms, 2),
                "decide": round(decide_latency_ms, 2),
                "tool": round(tool_latency_ms, 2),
                "assemble": round(assemble_latency_ms, 2),
                "answer": round(answer_latency_ms, 2),
            },
        }

    @staticmethod
    def summarize_history(
        messages: list[dict[str, Any]] | None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        scoped = messages[-limit:]
        summary: list[dict[str, Any]] = []
        for item in scoped:
            summary.append(
                {
                    "role": item.get("role"),
                    "content": (item.get("content") or "")[:800],
                }
            )
        return summary

    async def _decide(
        self,
        question: str,
        document_title: str | None,
        tool_trace: list[dict[str, Any]],
        chat_history: list[dict[str, Any]] | None,
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
                "chat_history": self.summarize_history(chat_history, limit=6),
                "available_tools": self.tools.describe(),
                "tool_trace_summary": [
                    {
                        "tool_name": step.get("tool_name"),
                        "tool_input": step.get("tool_input"),
                        "result_count": len((step.get("tool_output") or {}).get("results") or []),
                        "top_sections": [
                            item.get("section_path")
                            for item in ((step.get("tool_output") or {}).get("results") or [])[:3]
                        ],
                    }
                    for step in tool_trace
                ],
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
        chat_history: list[dict[str, Any]] | None,
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
                "chat_history": self.summarize_history(chat_history, limit=10),
                "context": packed_context,
                "tool_trace_summary": [
                    {"tool_name": step["tool_name"], "tool_input": step["tool_input"]}
                    for step in tool_trace
                ],
            },
            ensure_ascii=True,
        )
        return await self.llm.json_response(system_prompt, user_prompt)
