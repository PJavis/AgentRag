from __future__ import annotations

import json
import re
import time
from typing import Any, AsyncIterator

_VI_RE = re.compile(
    r"[àáảãạăắặằẳẵâấầẩẫậèéẻẽẹêếềệểễìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)

# Chit-chat / small-talk patterns (Vietnamese + English). Matched as substrings
# on lowercased message. Short messages with these tokens skip retrieval and
# get a quick warm reply from the cheap model.
_CHITCHAT_TOKENS = (
    "hi", "hello", "hey", "yo", "sup",
    "chào", "xin chào", "alo",
    "thanks", "thank you", "thx", "ty",
    "cảm ơn", "cám ơn", "thanks bro",
    "bye", "goodbye", "tạm biệt",
    "how are you", "how r u", "khỏe không", "bạn khỏe",
    "you there", "still there", "wassup",
    "ok thanks", "ok cool", "got it", "đã hiểu",
)


def _is_chitchat(question: str) -> bool:
    """Heuristic — short message containing a greeting/thanks token and no
    obvious request-for-info signal. Conservative; LLM still answers when
    in doubt by running full pipeline.
    """
    if not question:
        return False
    q = question.strip().lower()
    if len(q) > 60:
        return False
    # Strong information-request signals → not chitchat.
    if any(c in q for c in ("?", "tại sao", "vì sao", "what", "why", "how do",
                            "explain", "tóm tắt", "summarize", "list", "compare",
                            "khi nào", "ở đâu", "what is", "là gì", "định nghĩa",
                            "ví dụ", "example")):
        return False
    return any(tok in q for tok in _CHITCHAT_TOKENS)


def _lang_instruction(question: str) -> str:
    """Return an explicit language instruction based on the question."""
    if _VI_RE.search(question):
        return "Ngôn ngữ phản hồi: Tiếng Việt. Toàn bộ câu trả lời PHẢI bằng tiếng Việt."
    return "Response language: English."


_CHITCHAT_SYSTEM_PROMPT = (
    "You are a warm, friendly research companion. The user is making small talk "
    "(greeting / thanks / casual remark) — reply briefly and naturally, like a friend. "
    "1–3 sentences max. Do NOT pretend to retrieve documents. Do NOT cite anything. "
    "Match the user's language (Vietnamese ↔ English). Keep it light."
)

from src.agentrag.config import settings
from src.agentrag.services import (
    ContextAssemblyService,
    KnowledgeService,
    LLMGateway,
    SecurityService,
)
from src.agentrag.structured.pipeline import StructuredReasoningPipeline
from src.agentrag.structured.query_classifier import QueryIntentClassifier


class AgentService:
    def __init__(self):
        self.llm_gateway = LLMGateway()
        self.knowledge = KnowledgeService(llm_gateway=self.llm_gateway)
        self.context = ContextAssemblyService()
        self.security = SecurityService()
        self.classifier = QueryIntentClassifier(llm_gateway=self.llm_gateway)
        self.structured_pipeline = StructuredReasoningPipeline(
            knowledge_service=self.knowledge,
            llm_gateway=self.llm_gateway,
            security_service=self.security,
        )

    async def _retrieve_memory(self, conversation_id: str, question: str) -> list[dict[str, Any]]:
        if not settings.CHAT_STRUCTMEM_ENABLED or not conversation_id:
            return []
        try:
            from src.agentrag.chat.structmem import ChatMemoryService
            svc = ChatMemoryService()
            return await svc.retrieve(conversation_id, question)
        except Exception:
            return []

    async def chat(
        self,
        question: str,
        document_title: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        self.security.validate_chat_request(question=question, document_title=document_title)
        memory_context = await self._retrieve_memory(conversation_id, question)

        # ── Chit-chat fast-path ──────────────────────────────────────────────
        if _is_chitchat(question):
            client = self.llm_gateway._resolve_client("classify")  # cheap model
            answer = await client.text_response(
                system_prompt=f"{_lang_instruction(question)} {_CHITCHAT_SYSTEM_PROMPT}",
                user_prompt=question,
            )
            return {
                "answer": (answer or "").strip(),
                "citations": [],
                "tool_trace": [],
                "reasoning_path": "chitchat",
                "timings_ms": {"total": round((time.perf_counter() - total_started) * 1000, 2)},
            }

        # ── Classify intent (shared by structured gate + semantic path) ─────────
        classifier_output = None
        if settings.STRUCTURED_REASONING_ENABLED:
            classifier_output = await self.classifier.classify(
                question=question,
                document_title=document_title,
                chat_history=chat_history,
            )
            if classifier_output.intent == "structured":
                result = await self.structured_pipeline.run(
                    question=question,
                    document_title=document_title,
                    chat_history=chat_history,
                    query_type=classifier_output.query_type or "comparison",
                    classifier_confidence=classifier_output.confidence,
                )
                if not result.get("_structured_fallback"):
                    # Structured path success — trả về ngay
                    return result
                # Fallback: tiếp tục nhánh semantic bên dưới
        # ─────────────────────────────────────────────────────────────────────

        tool_trace: list[dict[str, Any]] = []
        final_answer: dict[str, Any] | None = None
        seen_calls: set[str] = set()
        decide_latency_ms = 0.0
        tool_latency_ms = 0.0
        answer_latency_ms = 0.0
        assemble_latency_ms = 0.0
        plan_latency_ms = 0.0
        plan_subqueries: list[str] = []

        # ── Plan-then-execute (complex / multi-hop queries) ──────────────────
        if (
            settings.AGENT_PLAN_THEN_EXECUTE_ENABLED
            and len(question.strip()) >= settings.AGENT_PLAN_TRIGGER_MIN_CHARS
        ):
            started = time.perf_counter()
            plan_subqueries = await self._plan_subqueries(question, document_title)
            plan_latency_ms = (time.perf_counter() - started) * 1000

        if plan_subqueries:
            # Parallel retrieve per sub-query. Merge into tool_trace so the
            # reactive decide loop sees rich evidence and usually short-circuits.
            import asyncio as _asyncio
            started = time.perf_counter()
            results = await _asyncio.gather(
                *[
                    self.knowledge.bootstrap_search(
                        query=sq, document_title=document_title, intent=classifier_output,
                    )
                    for sq in plan_subqueries
                ],
                return_exceptions=True,
            )
            tool_latency_ms += (time.perf_counter() - started) * 1000
            for sq, res in zip(plan_subqueries, results):
                if isinstance(res, BaseException):
                    continue
                sub_input, sub_output = res
                sub_output = self.security.filter_tool_results(
                    tool_output=sub_output, document_title=document_title,
                )
                fp = self.knowledge.fingerprint_call("search_hybrid_kg", sub_input)
                if fp in seen_calls:
                    continue
                seen_calls.add(fp)
                tool_trace.append({
                    "tool_name": "search_hybrid_kg",
                    "tool_input": sub_input,
                    "tool_output": sub_output,
                    "plan_subquery": sq,
                })

        # Always run a final bootstrap on the original question — covers cases
        # the planner missed.
        bootstrap_input, bootstrap_output = await self.knowledge.bootstrap_search(
            query=question,
            document_title=document_title,
            intent=classifier_output,
        )
        bootstrap_fp = self.knowledge.fingerprint_call(
            tool_name="search_hybrid_kg",
            tool_input=bootstrap_input,
        )
        started = time.perf_counter()
        bootstrap_output = self.security.filter_tool_results(
            tool_output=bootstrap_output,
            document_title=document_title,
        )
        tool_latency_ms += (time.perf_counter() - started) * 1000
        if bootstrap_fp not in seen_calls:
            tool_trace.append(
                {
                    "tool_name": "search_hybrid_kg",
                    "tool_input": bootstrap_input,
                    "tool_output": bootstrap_output,
                    "tool_latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
            seen_calls.add(bootstrap_fp)

        additional_steps = max(settings.AGENT_MAX_STEPS - 1, 0)
        for _ in range(additional_steps):
            started = time.perf_counter()
            decision = await self._decide(question, document_title, tool_trace, chat_history, memory_context)
            decide_elapsed = (time.perf_counter() - started) * 1000
            decide_latency_ms += decide_elapsed
            if decision.get("done"):
                final_answer = decision
                break

            tool_name = decision.get("tool_name")
            tool_input = decision.get("tool_input") or {}

            normalized_tool_name, normalized_tool_input = self.knowledge.normalize_tool_call(
                tool_name=tool_name or "search_hybrid_kg",
                tool_input=tool_input,
                question=question,
                document_title=document_title,
            )
            call_fingerprint = self.knowledge.fingerprint_call(
                tool_name=normalized_tool_name,
                tool_input=normalized_tool_input,
            )
            if call_fingerprint in seen_calls:
                break
            seen_calls.add(call_fingerprint)
            started = time.perf_counter()
            _, _, tool_output = await self.knowledge.execute_tool(
                tool_name=normalized_tool_name,
                tool_input=normalized_tool_input,
                question=question,
                document_title=document_title,
            )
            tool_output = self.security.filter_tool_results(
                tool_output=tool_output,
                document_title=document_title,
            )
            tool_elapsed = (time.perf_counter() - started) * 1000
            tool_latency_ms += tool_elapsed
            tool_trace.append(
                {
                    "tool_name": normalized_tool_name,
                    "tool_input": normalized_tool_input,
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
            memory_context,
        )
        answer_latency_ms += (time.perf_counter() - started) * 1000

        # ── Optional self-critique pass ─────────────────────────────────────
        critique_latency_ms = 0.0
        critique_meta: dict[str, Any] | None = None
        if settings.AGENT_SELF_CRITIQUE_ENABLED and self._should_critique(assembly["packed_context"]):
            started = time.perf_counter()
            revised, critique_meta = await self._self_critique(
                question=question,
                draft=answer.get("answer", ""),
                packed_context=assembly["packed_context"],
            )
            critique_latency_ms = (time.perf_counter() - started) * 1000
            if revised:
                answer["answer"] = revised

        grounded_citations = self._ground_citations(answer.get("citations", []), assembly["packed_context"])
        total_latency_ms = (time.perf_counter() - total_started) * 1000
        return {
            "question": question,
            "document_title": document_title,
            "tool_trace": tool_trace,
            "context": assembly["packed_context"],
            "answer": answer.get("answer", ""),
            "citations": grounded_citations,
            "reasoning_path": "semantic",
            "sql_query": None,
            "critique": critique_meta,
            "plan_subqueries": plan_subqueries,
            "timings_ms": {
                "total": round(total_latency_ms, 2),
                "plan": round(plan_latency_ms, 2),
                "decide": round(decide_latency_ms, 2),
                "tool": round(tool_latency_ms, 2),
                "assemble": round(assemble_latency_ms, 2),
                "answer": round(answer_latency_ms, 2),
                "critique": round(critique_latency_ms, 2),
            },
        }

    def _should_critique(self, packed_context: list[dict[str, Any]]) -> bool:
        """Critique when retrieval looks thin (low top RRF score) — that is
        the regime where hallucination / sycophancy is most dangerous."""
        if not packed_context:
            return True
        top = packed_context[0].get("score") or packed_context[0].get("rrf_score") or 0.0
        try:
            return float(top) < settings.AGENT_SELF_CRITIQUE_RRF_THRESHOLD
        except (TypeError, ValueError):
            return True

    async def _self_critique(
        self,
        question: str,
        draft: str,
        packed_context: list[dict[str, Any]],
    ) -> tuple[str | None, dict[str, Any]]:
        """Second-pass check: does draft cite anything not in context? Does it
        agree with implicit user assumptions that the context contradicts?
        Returns (revised_answer_or_None, metadata).
        """
        if not draft.strip():
            return None, {"verdict": "empty_draft", "revised": False}

        ctx_snippets = [
            {
                "i": i,
                "title": c.get("document_title"),
                "section": c.get("section_path"),
                "excerpt": (c.get("content") or c.get("excerpt") or "")[:600],
            }
            for i, c in enumerate(packed_context[:8])
        ]
        critique_system = (
            f"{_lang_instruction(question)} "
            "You are an evidence auditor. Given a question, a draft answer, and the retrieved context, "
            "decide if the draft (a) contains claims NOT supported by the context, "
            "(b) agrees with a false premise in the question, or (c) is fine. "
            "Return strict JSON: {verdict: 'ok'|'unsupported'|'sycophantic', "
            "issues: [string], revised: string|null}. "
            "If verdict=ok, revised=null. Otherwise revised = corrected answer using ONLY context. "
            "If context is too thin to answer, revised should state that plainly."
        )
        critique_user = json.dumps({
            "question": question,
            "draft": draft,
            "context": ctx_snippets,
        }, ensure_ascii=False)
        try:
            client = self.llm_gateway._resolve_client("decide")
            result = await client.json_response(critique_system, critique_user)
        except Exception as exc:
            return None, {"verdict": "error", "error": str(exc)[:200], "revised": False}

        verdict = result.get("verdict", "ok")
        revised = result.get("revised") if verdict != "ok" else None
        return (revised if isinstance(revised, str) and revised.strip() else None), {
            "verdict": verdict,
            "issues": result.get("issues") or [],
            "revised": bool(revised),
        }

    async def chat_stream(
        self,
        question: str,
        document_title: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        conversation_id: str | None = None,
    ) -> AsyncIterator[str]:
        """
        SSE generator. Yields chuỗi "data: <json>\\n\\n" theo Server-Sent Events format.

        Event types:
          status   — bước đang chạy (retrieval, thinking, ...)
          token    — một token của câu trả lời
          done     — payload cuối gồm citations, tool_trace, timings_ms
          error    — lỗi
        """
        def _sse(event: str, data: Any) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            self.security.validate_chat_request(question=question, document_title=document_title)
            memory_context = await self._retrieve_memory(conversation_id, question)

            # ── Chit-chat fast-path ───────────────────────────────────────────
            if _is_chitchat(question):
                yield _sse("status", {"step": "chitchat"})
                client = self.llm_gateway._resolve_client("classify")
                async for token in client.stream_text(
                    f"{_lang_instruction(question)} {_CHITCHAT_SYSTEM_PROMPT}",
                    question,
                ):
                    yield _sse("token", {"text": token})
                yield _sse("done", {
                    "citations": [],
                    "reasoning_path": "chitchat",
                    "tool_trace": [],
                })
                return

            # ── Classify + structured path ────────────────────────────────────
            classifier_output = None
            if settings.STRUCTURED_REASONING_ENABLED:
                yield _sse("status", {"step": "classify"})
                classifier_output = await self.classifier.classify(
                    question=question,
                    document_title=document_title,
                    chat_history=chat_history,
                )
                if classifier_output.intent == "structured":
                    yield _sse("status", {"step": "structured_reasoning"})
                    result = await self.structured_pipeline.run(
                        question=question,
                        document_title=document_title,
                        chat_history=chat_history,
                        query_type=classifier_output.query_type or "comparison",
                        classifier_confidence=classifier_output.confidence,
                    )
                    if not result.get("_structured_fallback"):
                        answer = result.get("answer", "")
                        for token in answer:
                            yield _sse("token", {"text": token})
                        yield _sse("done", {
                            "citations": result.get("citations", []),
                            "reasoning_path": "structured",
                            "sql_query": result.get("sql_query"),
                            "timings_ms": result.get("timings_ms", {}),
                        })
                        return

            # ── Semantic retrieval ────────────────────────────────────────────
            tool_trace: list[dict[str, Any]] = []
            seen_calls: set[str] = set()

            yield _sse("status", {"step": "retrieve"})
            bootstrap_input, bootstrap_output = await self.knowledge.bootstrap_search(
                query=question,
                document_title=document_title,
                intent=classifier_output,
            )
            bootstrap_output = self.security.filter_tool_results(
                tool_output=bootstrap_output,
                document_title=document_title,
            )
            tool_trace.append({
                "tool_name": "search_hybrid_kg",
                "tool_input": bootstrap_input,
                "tool_output": bootstrap_output,
            })
            seen_calls.add(self.knowledge.fingerprint_call("search_hybrid_kg", bootstrap_input))

            for _ in range(max(settings.AGENT_MAX_STEPS - 1, 0)):
                yield _sse("status", {"step": "decide"})
                decision = await self._decide(question, document_title, tool_trace, chat_history, memory_context)
                if decision.get("done"):
                    break
                tool_name = decision.get("tool_name") or "search_hybrid_kg"
                tool_input = decision.get("tool_input") or {}
                norm_name, norm_input = self.knowledge.normalize_tool_call(
                    tool_name=tool_name, tool_input=tool_input,
                    question=question, document_title=document_title,
                )
                fp = self.knowledge.fingerprint_call(norm_name, norm_input)
                if fp in seen_calls:
                    break
                seen_calls.add(fp)
                yield _sse("status", {"step": "tool", "tool": norm_name})
                _, _, tool_output = await self.knowledge.execute_tool(
                    tool_name=norm_name, tool_input=norm_input,
                    question=question, document_title=document_title,
                )
                tool_output = self.security.filter_tool_results(
                    tool_output=tool_output, document_title=document_title,
                )
                tool_trace.append({
                    "tool_name": norm_name,
                    "tool_input": norm_input,
                    "tool_output": tool_output,
                })

            assembly = self.context.assemble(question, [e["tool_output"] for e in tool_trace])

            # ── Stream answer tokens ──────────────────────────────────────────
            yield _sse("status", {"step": "answer"})
            system_prompt = (
                f"{_lang_instruction(question)} "
                "You are a knowledgeable research companion — warm, direct, and intellectually honest. Sound like a smart friend, not a corporate FAQ. "
                "PRIMARY RULE: factual claims about the documents must come ONLY from the provided context. Do NOT invent facts, page numbers, or quotations. "
                "ANTI-SYCOPHANCY: if the user states something contradicted by the context, push back politely and cite the contradicting passage. "
                "Do NOT agree to be agreeable. It is better to disagree correctly than to flatter wrongly. "
                "UNCERTAINTY: if context is thin or absent for the question, say so plainly (\"the document doesn't cover X\") instead of guessing. "
                "Never fabricate citations. "
                "LENGTH: match question intent — 'tóm tắt'/'explain'/'overview' → structured multi-paragraph with bullets; factual lookup → 1–3 sentences. "
                "Surface concrete details (names, numbers, definitions, examples) from context — do NOT default to vague descriptions when specifics exist. "
                "MULTI-DOC: focus only on the document(s) directly relevant; ignore unrelated documents. "
                "CONVERSATIONAL: greetings / small talk / off-topic chat → reply briefly and warmly without retrieval; do NOT force-fit context. "
                "Do NOT return JSON. If the question is too vague to answer usefully, ask ONE focused clarifying question (don't both answer and ask)."
            )
            user_payload: dict[str, Any] = {
                "question": question,
                "chat_history": self.summarize_history(chat_history, limit=6),
                "context": assembly["packed_context"],
            }
            if memory_context:
                user_payload["conversation_memory"] = memory_context
            user_prompt = json.dumps(user_payload, ensure_ascii=True)

            client = self.llm_gateway._resolve_client("answer")
            async for token in client.stream_text(system_prompt, user_prompt):
                yield _sse("token", {"text": token})

            packed = assembly["packed_context"]
            # Dedupe by content_hash, keep first occurrence
            seen: set[str] = set()
            deduped_citations = []
            for c in packed:
                h = c.get("content_hash", "")
                if h and h not in seen:
                    seen.add(h)
                    page_start = c.get("page_start")
                    page_end = c.get("page_end")
                    entry = {
                        "document_title": c.get("document_title"),
                        "section_path": c.get("section_path"),
                        "content_hash": h,
                        "excerpt": (c.get("excerpt") or c.get("content") or "")[:300],
                        "segment_type": c.get("segment_type", "text"),
                    }
                    if page_start is not None:
                        entry["page"] = page_start if page_start == page_end else f"{page_start}-{page_end}"
                        entry["page_start"] = page_start
                        entry["page_end"] = page_end
                    deduped_citations.append(entry)
            yield _sse("done", {
                "citations": deduped_citations,
                "highlights": [],
                "reasoning_path": "semantic",
                "sql_query": None,
                "tool_trace": [
                    {"tool_name": s["tool_name"], "tool_input": s["tool_input"]}
                    for s in tool_trace
                ],
            })

        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

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

    async def _plan_subqueries(
        self,
        question: str,
        document_title: str | None,
    ) -> list[str]:
        """Decompose a complex question into 1–N self-contained sub-queries.

        Returns an empty list when the question is single-step (planner says
        so) or on any planner failure — caller falls back to plain reactive
        retrieve.
        """
        system = (
            f"{_lang_instruction(question)} "
            "You are a research planner. Given a user question, decide if it is "
            "multi-step (needs intermediate facts to answer) or single-step. "
            "Output strict JSON: {multi_step: bool, subqueries: [string]}. "
            "If single_step → subqueries=[]. "
            "If multi_step → subqueries are 2–N self-contained retrieval queries "
            "in the original language; each query must be answerable independently "
            "with document search. Avoid restating the full question. Max N follows "
            "the configured cap. Do NOT include the original question verbatim."
        )
        user = json.dumps(
            {"question": question, "document_title": document_title},
            ensure_ascii=False,
        )
        try:
            client = self.llm_gateway._resolve_client("decide")
            result = await client.json_response(system, user)
        except Exception:
            return []
        if not isinstance(result, dict):
            return []
        if not result.get("multi_step"):
            return []
        subs = result.get("subqueries") or []
        if not isinstance(subs, list):
            return []
        out: list[str] = []
        for sq in subs:
            if isinstance(sq, str):
                sq = sq.strip()
                if sq and sq != question.strip():
                    out.append(sq)
            if len(out) >= settings.AGENT_PLAN_MAX_SUBQUERIES:
                break
        return out

    async def _decide(
        self,
        question: str,
        document_title: str | None,
        tool_trace: list[dict[str, Any]],
        chat_history: list[dict[str, Any]] | None,
        memory_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        system_prompt = (
            "You are a retrieval agent with self-reflection. "
            "Before deciding, reflect on the evidence collected so far:\n"
            "  (1) Does it DIRECTLY answer the specific question? (not just related info)\n"
            "  (2) What specific fact is still missing?\n"
            "  (3) Would a more targeted query (different keywords, entity name, or sub-question) find it?\n"
            "Return JSON: {done, reflection, tool_name, tool_input, reason}.\n"
            "Set done=true only when the evidence directly answers the question.\n"
            "If missing info: set tool_name to a search tool, tool_input.query to a specific refined query "
            "that targets EXACTLY what is missing — avoid repeating the same query already used.\n"
            "For multi-hop questions, retrieve intermediate facts first, then use those to form the next query."
        )
        decide_payload: dict[str, Any] = {
            "question": question,
            "document_title": document_title,
            "chat_history": self.summarize_history(chat_history, limit=6),
            "available_tools": self.knowledge.describe_tools(),
            "tool_trace_summary": [
                    {
                        "tool_name": step.get("tool_name"),
                        "tool_input": step.get("tool_input"),
                        "result_count": len((step.get("tool_output") or {}).get("results") or []),
                        "top_results": [
                            {
                                "section_path": item.get("section_path"),
                                "excerpt": (item.get("content") or "")[:200],
                            }
                            for item in ((step.get("tool_output") or {}).get("results") or [])[:3]
                        ],
                    }
                    for step in tool_trace
                ],
        }
        if memory_context:
            decide_payload["conversation_memory"] = memory_context
        user_prompt = json.dumps(decide_payload, ensure_ascii=True)
        decision, _latency_ms = await self.llm_gateway.json_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            task="decide",
        )
        if decision.get("done"):
            return decision
        tool_name = decision.get("tool_name") or "search_hybrid_kg"
        tool_input = decision.get("tool_input") or {"query": question, "top_k": settings.AGENT_TOOL_TOP_K}
        return {
            "done": False,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "reason": decision.get("reason"),
            "reflection": decision.get("reflection"),  # self-reflection reasoning
        }

    def _ground_citations(
        self,
        citations: list[dict[str, Any]],
        packed_context: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Build lookup: content_hash → full context item (for page + excerpt enrichment)
        hash_to_ctx: dict[str, dict[str, Any]] = {
            item["content_hash"]: item
            for item in packed_context
            if item.get("content_hash")
        }
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
            if key not in allowed:
                continue
            ctx = hash_to_ctx.get(citation.get("content_hash", ""), {})
            entry: dict[str, Any] = {
                "document_title": citation.get("document_title"),
                "section_path": citation.get("section_path"),
                "position": citation.get("position"),
                "content_hash": citation.get("content_hash"),
                "excerpt": ctx.get("excerpt", ""),
            }
            # Include page reference when available (PDF sources)
            if ctx.get("page") is not None:
                entry["page"] = ctx["page"]
            if ctx.get("page_start") is not None:
                entry["page_start"] = ctx["page_start"]
            if ctx.get("page_end") is not None:
                entry["page_end"] = ctx["page_end"]
            grounded.append(entry)
        return grounded

    async def _answer(
        self,
        question: str,
        packed_context: list[dict[str, Any]],
        tool_trace: list[dict[str, Any]],
        final_answer: dict[str, Any] | None,
        chat_history: list[dict[str, Any]] | None,
        memory_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if final_answer and final_answer.get("answer"):
            return {
                "answer": final_answer["answer"],
                "citations": final_answer.get("citations", []),
                "highlights": final_answer.get("highlights", []),
            }

        system_prompt = (
            f"{_lang_instruction(question)} "
            "Answer ONLY from the provided context. "
            "Return JSON with keys: answer, citations, highlights. "
            "Each citation must include document_title, section_path, position, content_hash. "
            "highlights: array of 3-5 strings — the most important facts or takeaways from the answer, "
            "each a complete self-contained sentence. "
            "In the answer text, wrap important medical/technical terms in **bold**. "
            "Be concise and direct — answer the specific question, do not add background or general overviews. "
            "When context contains multiple documents, focus on the document(s) directly relevant to the question; ignore unrelated documents. "
            "If the question is too vague to give a useful answer (e.g., which item, which document, or which aspect is missing), "
            "set answer to ONE focused clarifying question, citations to [], and highlights to []. "
            "Do not answer and ask at the same time. "
            "If context is insufficient for a specific question, say so explicitly. "
            "Answer in clear, natural sentences and avoid broken wording. "
            "Only cite claims directly supported by the provided context. "
            "Do NOT add examples, field names, or details not explicitly present in the context. "
            "You MAY perform simple arithmetic (×, ÷, +, −) on numeric values explicitly stated in the context; "
            "show the calculation briefly (e.g. '10 × 3000 = 30,000 gold')."
        )
        answer_payload: dict[str, Any] = {
            "question": question,
            "chat_history": self.summarize_history(chat_history, limit=10),
            "context": packed_context,
            "tool_trace_summary": [
                {"tool_name": step["tool_name"], "tool_input": step["tool_input"]}
                for step in tool_trace
            ],
        }
        if memory_context:
            answer_payload["conversation_memory"] = memory_context
        user_prompt = json.dumps(answer_payload, ensure_ascii=True)
        answer, _latency_ms = await self.llm_gateway.json_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            task="answer",
        )
        return answer
