"""LangGraph backend for AgentService — proper node split (v2).

Each node = one phase of the existing chat() loop. Helper methods on the
shared AgentService instance handle the actual work; the graph supplies
orchestration, conditional branching, and checkpointing.

Nodes:
    validate            → security gate
    memory              → ChatMemoryService.retrieve
    chitchat_check      → conditional: chitchat fast-path vs semantic
    chitchat_answer     → end
    semantic_plan       → optional sub-query plan
    bootstrap           → initial hybrid_kg search
    decide              → LLM: next tool or done
    tool_exec           → run tool, append trace (loop → decide)
    assemble            → ContextAssembler.assemble
    answer              → LLM: final synthesis
    critique            → CRAG groundedness check (gated)
    corrective_retrieve → CRAG step-back re-retrieve (gated)
    ground              → citation grounding

Conditional edges:
    chitchat_check  → chitchat_answer | semantic_plan
    decide          → tool_exec (more) | assemble (done or max steps)
    tool_exec       → decide (always loop back)
    critique        → ground | corrective_retrieve (gated CRAG)
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.agentrag.agent.service import AgentService, _is_chitchat, _lang_instruction, _CHITCHAT_SYSTEM_PROMPT, _attach_source_ids
from src.agentrag.common.langfuse_client import current_trace_id, observe_chat_turn, update_turn_trace
from src.agentrag.config import settings


# ── State ────────────────────────────────────────────────────────────────────

class ChatState(TypedDict, total=False):
    # Input
    question: str
    document_title: Optional[str]
    chat_history: list[dict[str, Any]]
    conversation_id: Optional[str]
    verbosity: Optional[str]            # "concise" | "detailed" | None (auto)
    total_started: float

    # Intermediate
    memory_context: list[dict[str, Any]]
    is_chitchat: bool
    plan_subqueries: list[str]
    tool_trace: list[dict[str, Any]]
    seen_calls: list[str]             # set serialized as list for state pickle
    step_count: int
    decide_decision: Optional[dict[str, Any]]
    packed_context: list[dict[str, Any]]
    critique_decision: Optional[dict[str, Any]]
    critique_latency_ms: float
    critique_retries: int

    # Timings
    decide_latency_ms: float
    tool_latency_ms: float
    answer_latency_ms: float
    assemble_latency_ms: float
    plan_latency_ms: float

    # Output
    answer: str
    citations: list[dict[str, Any]]
    highlights: list[str]
    reasoning_path: str
    timings_ms: dict[str, float]


_INNER = AgentService()
_CHECKPOINTER = InMemorySaver()


def _chain_query(subquery: str, prior_output: dict[str, Any] | None) -> str:
    """Multi-hop chaining (WS3): seed a dependent sub-query with the top snippet
    from the previous hop so later hops build on earlier answers instead of
    being retrieved blind."""
    if not prior_output:
        return subquery
    hits = prior_output.get("hits") or []
    if not hits:
        return subquery
    snippet = (hits[0].get("content") or "")[:240]
    if not snippet:
        return subquery
    return f"Bối cảnh: {snippet}\n\nCâu hỏi: {subquery}"


def _message_signals(tool_trace: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Derive UI message-level signals from the tool trace: whether the first
    retrieval was a semantic-cache hit, the retrieval mode, and the domain
    route. All default-absent so the UI chips stay hidden when unavailable."""
    cache_hit = False
    mode: str | None = None
    domain: str | None = None
    for entry in tool_trace or []:
        out = entry.get("tool_output") or {}
        if not isinstance(out, dict):
            continue
        if out.get("semantic_cache_hit"):
            cache_hit = True
        if mode is None and out.get("mode"):
            mode = out.get("mode")
        dr = out.get("domain_route")
        if domain is None and dr:
            if isinstance(dr, dict):
                systems = dr.get("systems") or []
                domain = ", ".join(str(s) for s in systems) or None
            else:
                domain = str(dr)
    return {"semantic_cache_hit": cache_hit, "retrieval_mode": mode, "domain_route": domain}


# ── Nodes ────────────────────────────────────────────────────────────────────

async def validate(state: ChatState) -> dict[str, Any]:
    _INNER.security.validate_chat_request(
        question=state["question"], document_title=state.get("document_title")
    )
    return {"total_started": time.perf_counter(), "tool_trace": [], "seen_calls": []}


async def memory(state: ChatState) -> dict[str, Any]:
    mem = await _INNER._retrieve_memory(
        state.get("conversation_id") or "", state["question"]
    )
    return {"memory_context": mem}


async def chitchat_check(state: ChatState) -> dict[str, Any]:
    return {"is_chitchat": _is_chitchat(state["question"])}


async def chitchat_answer(state: ChatState) -> dict[str, Any]:
    client = _INNER.llm_gateway._resolve_client("classify")
    answer = await client.text_response(
        system_prompt=f"{_lang_instruction(state['question'])} {_CHITCHAT_SYSTEM_PROMPT}",
        user_prompt=state["question"],
    )
    elapsed = (time.perf_counter() - state["total_started"]) * 1000
    return {
        "answer": (answer or "").strip(),
        "citations": [],
        "highlights": [],
        "reasoning_path": "chitchat",
        "timings_ms": {"total": round(elapsed, 2)},
    }


async def semantic_plan(state: ChatState) -> dict[str, Any]:
    from src.agentrag.agent.service import _is_verbose_followup
    # Trigger planner when:
    #   - feature enabled AND
    #   - question is long enough, OR is a summary/verbose intent (short
    #     "tóm tắt tài liệu" still needs multi-subquery fan-out to gather
    #     enough context for a structured overview).
    long_enough = len(state["question"].strip()) >= settings.AGENT_PLAN_TRIGGER_MIN_CHARS
    summary_intent = _is_verbose_followup(state["question"]) or state.get("verbosity") == "detailed"
    if not (settings.AGENT_PLAN_THEN_EXECUTE_ENABLED and (long_enough or summary_intent)):
        return {"plan_subqueries": [], "plan_latency_ms": 0.0}
    started = time.perf_counter()
    subs = await _INNER._plan_subqueries(state["question"], state.get("document_title"))
    return {
        "plan_subqueries": subs or [],
        "plan_latency_ms": (time.perf_counter() - started) * 1000,
    }


async def bootstrap(state: ChatState) -> dict[str, Any]:
    import asyncio as _asyncio
    tool_trace = list(state.get("tool_trace") or [])
    seen = set(state.get("seen_calls") or [])
    tool_latency_ms = state.get("tool_latency_ms", 0.0)
    doc_title = state.get("document_title")

    # Plan subqueries: parallel (default) or sequential-chained (multi-hop).
    if state.get("plan_subqueries"):
        started = time.perf_counter()
        subqueries = state["plan_subqueries"]
        if settings.AGENT_MULTIHOP_ENABLED:
            prior_out: dict[str, Any] | None = None
            for sq in subqueries:
                chained = _chain_query(sq, prior_out)
                try:
                    sub_in, sub_out = await _INNER.knowledge.bootstrap_search(
                        query=chained, document_title=doc_title,
                    )
                except BaseException:
                    continue
                sub_out = _INNER.security.filter_tool_results(
                    tool_output=sub_out, document_title=doc_title
                )
                prior_out = sub_out
                fp = _INNER.knowledge.fingerprint_call("search_hybrid_kg", sub_in)
                if fp in seen:
                    continue
                seen.add(fp)
                tool_trace.append({
                    "tool_name": "search_hybrid_kg", "tool_input": sub_in,
                    "tool_output": sub_out, "plan_subquery": sq, "multihop": True,
                })
        else:
            results = await _asyncio.gather(
                *[
                    _INNER.knowledge.bootstrap_search(
                        query=sq, document_title=doc_title,
                    )
                    for sq in subqueries
                ],
                return_exceptions=True,
            )
            for sq, res in zip(subqueries, results):
                if isinstance(res, BaseException):
                    continue
                sub_in, sub_out = res
                sub_out = _INNER.security.filter_tool_results(
                    tool_output=sub_out, document_title=doc_title
                )
                fp = _INNER.knowledge.fingerprint_call("search_hybrid_kg", sub_in)
                if fp in seen:
                    continue
                seen.add(fp)
                tool_trace.append({
                    "tool_name": "search_hybrid_kg", "tool_input": sub_in,
                    "tool_output": sub_out, "plan_subquery": sq,
                })
        tool_latency_ms += (time.perf_counter() - started) * 1000

    # Always run a final bootstrap on the original question.
    boot_in, boot_out = await _INNER.knowledge.bootstrap_search(
        query=state["question"], document_title=doc_title,
    )
    started = time.perf_counter()
    boot_out = _INNER.security.filter_tool_results(
        tool_output=boot_out, document_title=doc_title
    )
    tool_latency_ms += (time.perf_counter() - started) * 1000
    fp = _INNER.knowledge.fingerprint_call("search_hybrid_kg", boot_in)
    if fp not in seen:
        tool_trace.append({
            "tool_name": "search_hybrid_kg",
            "tool_input": boot_in,
            "tool_output": boot_out,
        })
        seen.add(fp)

    return {
        "tool_trace": tool_trace,
        "seen_calls": list(seen),
        "tool_latency_ms": tool_latency_ms,
        "step_count": 0,
    }


async def decide(state: ChatState) -> dict[str, Any]:
    started = time.perf_counter()
    decision = await _INNER._decide(
        question=state["question"],
        document_title=state.get("document_title"),
        tool_trace=state["tool_trace"],
        chat_history=state.get("chat_history"),
        memory_context=state.get("memory_context"),
    )
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "decide_decision": decision,
        "decide_latency_ms": state.get("decide_latency_ms", 0.0) + elapsed,
        "step_count": state.get("step_count", 0) + 1,
    }


async def tool_exec(state: ChatState) -> dict[str, Any]:
    decision = state["decide_decision"]
    tool_trace = list(state["tool_trace"])
    seen = set(state.get("seen_calls") or [])
    doc_title = state.get("document_title")

    tool_name, tool_input = _INNER.knowledge.normalize_tool_call(
        tool_name=decision.get("tool_name") or "search_hybrid_kg",
        tool_input=decision.get("tool_input") or {},
        question=state["question"],
        document_title=doc_title,
    )
    fp = _INNER.knowledge.fingerprint_call(tool_name, tool_input)
    if fp in seen:
        # Duplicate call → force done.
        return {"decide_decision": {"done": True}, "tool_trace": tool_trace}
    seen.add(fp)

    started = time.perf_counter()
    _, _, tool_output = await _INNER.knowledge.execute_tool(
        tool_name=tool_name, tool_input=tool_input,
        question=state["question"], document_title=doc_title,
    )
    tool_output = _INNER.security.filter_tool_results(
        tool_output=tool_output, document_title=doc_title
    )
    elapsed = (time.perf_counter() - started) * 1000

    tool_trace.append({
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "tool_latency_ms": round(elapsed, 2),
    })
    return {
        "tool_trace": tool_trace,
        "seen_calls": list(seen),
        "tool_latency_ms": state.get("tool_latency_ms", 0.0) + elapsed,
    }


async def assemble(state: ChatState) -> dict[str, Any]:
    started = time.perf_counter()
    # ContextAssembler.assemble returns a dict; we want the packed list only.
    result = await _INNER.context.assemble(
        state["question"],
        [step["tool_output"] for step in state["tool_trace"]],
    )
    packed = result.get("packed_context", []) if isinstance(result, dict) else result
    return {
        "packed_context": packed,
        "assemble_latency_ms": (time.perf_counter() - started) * 1000,
    }


async def answer_node(state: ChatState) -> dict[str, Any]:
    started = time.perf_counter()
    final = state.get("decide_decision") if state.get("decide_decision", {}).get("done") else None
    out = await _INNER._answer(
        question=state["question"],
        packed_context=state["packed_context"],
        tool_trace=state["tool_trace"],
        final_answer=final,
        chat_history=state.get("chat_history"),
        memory_context=state.get("memory_context"),
        verbosity=state.get("verbosity"),
    )
    return {
        "answer": out.get("answer", ""),
        "citations": out.get("citations", []),
        "highlights": out.get("highlights", []),
        "answer_latency_ms": (time.perf_counter() - started) * 1000,
    }


async def critique(state: ChatState) -> dict[str, Any]:
    if not settings.CRAG_ENABLED:
        return {"critique_decision": {"grounded": True, "reason": "disabled"}}
    started = time.perf_counter()
    decision = _INNER._critique(
        answer=state.get("answer", ""),
        citations=state.get("citations", []),
        packed_context=state.get("packed_context", []),
    )
    elapsed = max((time.perf_counter() - started) * 1000, 0.01)
    return {"critique_decision": decision, "critique_latency_ms": elapsed}


async def corrective_retrieve(state: ChatState) -> dict[str, Any]:
    """One bounded CRAG correction: step-back query rewrite + re-retrieve +
    re-answer. Appends new evidence to the trace, then loops to critique."""
    doc_title = state.get("document_title")
    # Step-back: broaden the query to recover when the first retrieval missed.
    stepback = f"{state['question']} (bối cảnh tổng quát, định nghĩa, nguyên nhân)"
    boot_in, boot_out = await _INNER.knowledge.bootstrap_search(
        query=stepback, document_title=doc_title,
    )
    boot_out = _INNER.security.filter_tool_results(tool_output=boot_out, document_title=doc_title)
    trace = list(state.get("tool_trace") or [])
    trace.append({"tool_name": "search_hybrid_kg", "tool_input": boot_in,
                  "tool_output": boot_out, "corrective": True})

    assembled = await _INNER.context.assemble(
        state["question"], [s["tool_output"] for s in trace]
    )
    packed = assembled.get("packed_context", []) if isinstance(assembled, dict) else assembled
    out = await _INNER._answer(
        question=state["question"], packed_context=packed, tool_trace=trace,
        final_answer=None, chat_history=state.get("chat_history"),
        memory_context=state.get("memory_context"), verbosity=state.get("verbosity"),
    )
    return {
        "tool_trace": trace,
        "packed_context": packed,
        "answer": out.get("answer", ""),
        "citations": out.get("citations", []),
        "highlights": out.get("highlights", []),
        "critique_retries": state.get("critique_retries", 0) + 1,
    }


async def ground(state: ChatState) -> dict[str, Any]:
    # Cite by source number: the answer's inline [n] = packed_context position,
    # so the UI citation list must be the full ordered packed context (tagged
    # with `source` = n), not the model's free-form citation subset.
    grounded = _INNER._build_packed_citations(state.get("packed_context") or [])
    from src.agentrag.agent.service import _should_drop_abstention_citations
    if _should_drop_abstention_citations(state.get("answer", ""), state.get("packed_context"), settings.RETRIEVAL_RELEVANCE_FLOOR):
        grounded = []
    await _attach_source_ids(grounded)
    elapsed = (time.perf_counter() - state["total_started"]) * 1000
    timings = {
        "total": round(elapsed, 2),
        "decide": round(state.get("decide_latency_ms", 0.0), 2),
        "tool": round(state.get("tool_latency_ms", 0.0), 2),
        "assemble": round(state.get("assemble_latency_ms", 0.0), 2),
        "answer": round(state.get("answer_latency_ms", 0.0), 2),
        "plan": round(state.get("plan_latency_ms", 0.0), 2),
        "critique": round(state.get("critique_latency_ms", 0.0), 4),
    }
    return {
        "citations": grounded,
        "reasoning_path": "semantic",
        "timings_ms": timings,
    }


# ── Routers (conditional edges) ──────────────────────────────────────────────

def _route_chitchat(state: ChatState) -> str:
    return "chitchat_answer" if state.get("is_chitchat") else "semantic_plan"


def _route_decide(state: ChatState) -> str:
    decision = state.get("decide_decision") or {}
    if decision.get("done"):
        return "assemble"
    if state.get("step_count", 0) >= max(settings.AGENT_MAX_STEPS - 1, 1):
        return "assemble"
    return "tool_exec"


def _route_critique(state: ChatState) -> str:
    if not settings.CRAG_ENABLED:
        return "ground"
    decision = state.get("critique_decision") or {}
    if decision.get("grounded", True):
        return "ground"
    if state.get("critique_retries", 0) >= settings.AGENT_CRITIQUE_MAX_RETRIES:
        return "ground"
    return "corrective_retrieve"


# ── Build graph ──────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(ChatState)
    g.add_node("validate", validate)
    g.add_node("memory", memory)
    g.add_node("chitchat_check", chitchat_check)
    g.add_node("chitchat_answer", chitchat_answer)
    g.add_node("semantic_plan", semantic_plan)
    g.add_node("bootstrap", bootstrap)
    g.add_node("decide", decide)
    g.add_node("tool_exec", tool_exec)
    g.add_node("assemble", assemble)
    g.add_node("answer", answer_node)
    g.add_node("critique", critique)
    g.add_node("corrective_retrieve", corrective_retrieve)
    g.add_node("ground", ground)

    g.add_edge(START, "validate")
    g.add_edge("validate", "memory")
    g.add_edge("memory", "chitchat_check")
    g.add_conditional_edges("chitchat_check", _route_chitchat,
                            {"chitchat_answer": "chitchat_answer", "semantic_plan": "semantic_plan"})
    g.add_edge("chitchat_answer", END)
    g.add_edge("semantic_plan", "bootstrap")
    g.add_edge("bootstrap", "decide")
    g.add_conditional_edges("decide", _route_decide,
                            {"tool_exec": "tool_exec", "assemble": "assemble"})
    g.add_edge("tool_exec", "decide")
    g.add_edge("assemble", "answer")
    g.add_edge("answer", "critique")
    g.add_conditional_edges("critique", _route_critique,
                            {"ground": "ground", "corrective_retrieve": "corrective_retrieve"})
    g.add_edge("corrective_retrieve", "critique")
    g.add_edge("ground", END)
    return g.compile(checkpointer=_CHECKPOINTER)


_GRAPH = _build_graph()


class GraphAgentService:
    """Drop-in replacement for AgentService with LangGraph orchestrator (v2 nodes)."""

    @observe_chat_turn
    async def chat(
        self,
        question: str,
        document_title: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        conversation_id: str | None = None,
        domain_filter: dict[str, Any] | None = None,
        verbosity: str | None = None,
    ) -> dict[str, Any]:
        update_turn_trace(name=(question or "")[:80], session_id=conversation_id)
        # Verbose / summary follow-ups like "viết dài hơn được không?" have no
        # domain terms → retrieval misses everything. Rewrite the question by
        # prepending the most recent prior user question so the retriever
        # (planner + tool calls) has something to match against. Keep the
        # rewrite simple natural Vietnamese — bracketed payloads confuse 7B
        # JSON models.
        from src.agentrag.agent.service import _is_verbose_followup
        effective_question = question
        if (
            _is_verbose_followup(question)
            and chat_history
            and len(question.strip()) < 80
        ):
            prior_user = next(
                (m.get("content", "") for m in reversed(chat_history)
                 if m.get("role") == "user" and (m.get("content") or "").strip() != question.strip()),
                None,
            )
            if prior_user:
                effective_question = f"{prior_user} (yêu cầu chi tiết hơn)"
        initial: ChatState = {
            "question": effective_question,
            "document_title": document_title,
            "chat_history": chat_history or [],
            "conversation_id": conversation_id,
            "verbosity": verbosity,
        }
        # S5 — propagate domain_filter via ContextVar so AgentTools.search_*
        # picks it up downstream (same mechanism as loop backend).
        from src.agentrag.retrieval.context import set_domain_filter
        set_domain_filter(domain_filter)
        config = {"configurable": {"thread_id": conversation_id or f"anon-{id(initial)}"}}
        # Total wall-clock budget: the per-call LLM timeout does NOT bound the whole
        # decide→tools→rerank→answer loop (each call retryable). Under a gemini 503 storm
        # the total can run away (>120s, 42-min hang observed). Cap it and degrade
        # gracefully instead of hanging the caller.
        import asyncio as _aio
        import logging as _logging
        try:
            state = await _aio.wait_for(
                _GRAPH.ainvoke(initial, config=config),
                timeout=settings.AGENT_TOTAL_TIMEOUT_S,
            )
        except _aio.TimeoutError:
            _logging.getLogger(__name__).warning(
                "agent.chat exceeded total budget %.0fs (question=%r) — graceful timeout",
                settings.AGENT_TOTAL_TIMEOUT_S, (question or "")[:60],
            )
            return {
                "question": question,
                "document_title": document_title,
                "answer": "Hệ thống đang bận, vui lòng thử lại sau giây lát.",
                "citations": [],
                "tool_trace": [],
                "reasoning_path": "timeout",
                "highlights": [],
                "timings_ms": {},
                "langfuse_trace_id": current_trace_id(),
                "context": [],
                "timed_out": True,
            }
        return {
            "question": question,
            "document_title": document_title,
            "answer": state.get("answer", ""),
            "citations": state.get("citations", []),
            "tool_trace": state.get("tool_trace", []),
            "reasoning_path": state.get("reasoning_path", "semantic"),
            "highlights": state.get("highlights", []),
            "timings_ms": state.get("timings_ms", {}),
            "langfuse_trace_id": current_trace_id(),
            # Retrieved+packed passages used to synthesize the answer. Exposed so
            # eval (RAGAS contexts) and clients can inspect grounding evidence.
            "context": state.get("packed_context", []),
            **_message_signals(state.get("tool_trace")),
        }

    async def chat_stream(
        self,
        question: str,
        document_title: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        conversation_id: str | None = None,
        model_override: str | None = None,
        verbosity: str | None = None,
    ) -> AsyncIterator[str]:
        """Run the FULL LangGraph (semantic plan, agent loop, CRAG critique,
        abstain, grounding) then stream the grounded answer. Preserves the SSE protocol
        (status / token / done / error). The router sets domain_filter +
        document_scope via ContextVar before calling this, so graph nodes pick
        them up."""
        def _sse(event: str, data: Any) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        try:
            from src.agentrag.agent.service import _is_verbose_followup
            effective_question = question
            if _is_verbose_followup(question) and chat_history and len(question.strip()) < 80:
                prior_user = next(
                    (m.get("content", "") for m in reversed(chat_history)
                     if m.get("role") == "user" and (m.get("content") or "").strip() != question.strip()),
                    None,
                )
                if prior_user:
                    effective_question = f"{prior_user} (yêu cầu chi tiết hơn)"

            yield _sse("status", {"step": "retrieve"})
            initial: ChatState = {
                "question": effective_question,
                "document_title": document_title,
                "chat_history": chat_history or [],
                "conversation_id": conversation_id,
                "verbosity": verbosity,
            }
            config = {"configurable": {"thread_id": conversation_id or f"anon-{id(initial)}"}}
            state = await _GRAPH.ainvoke(initial, config=config)

            answer = state.get("answer", "") or ""
            yield _sse("status", {"step": "answer"})
            # The graph already produced the whole answer; replay it as ~40-char
            # token frames (typewriter UX) instead of one frame per char, which
            # would emit thousands of SSE writes for a long answer.
            for i in range(0, len(answer), 40):
                yield _sse("token", {"text": answer[i:i + 40]})

            yield _sse("done", {
                "citations": state.get("citations", []),
                "highlights": state.get("highlights", []),
                "reasoning_path": state.get("reasoning_path", "semantic"),
                "tool_trace": [
                    {"tool_name": s.get("tool_name"), "tool_input": s.get("tool_input")}
                    for s in (state.get("tool_trace") or [])
                ],
                "timings_ms": state.get("timings_ms", {}),
                **_message_signals(state.get("tool_trace")),
            })
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})
