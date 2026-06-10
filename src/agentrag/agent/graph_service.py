"""LangGraph backend for AgentService — proper node split (v2).

Each node = one phase of the existing chat() loop. Helper methods on the
shared AgentService instance handle the actual work; the graph supplies
orchestration, conditional branching, and checkpointing.

Nodes:
    validate            → security gate
    memory              → ChatMemoryService.retrieve
    chitchat_check      → conditional: chitchat fast-path vs classify
    chitchat_answer     → end
    classify            → query intent + structured gate
    structured_run      → SQL pipeline (conditional fallback)
    semantic_plan       → optional sub-query plan
    bootstrap           → initial hybrid_kg search
    decide              → LLM: next tool or done
    tool_exec           → run tool, append trace (loop → decide)
    assemble            → ContextAssembler.assemble
    answer              → LLM: final synthesis
    ground              → citation grounding

Conditional edges:
    chitchat_check  → chitchat_answer | classify
    classify        → structured_run | semantic_plan | chitchat_answer
    structured_run  → ground (success) | semantic_plan (fallback)
    decide          → tool_exec (more) | assemble (done or max steps)
    tool_exec       → decide (always loop back)
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.agentrag.agent.service import AgentService, _is_chitchat, _lang_instruction, _CHITCHAT_SYSTEM_PROMPT, _attach_source_ids
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
    classifier_output: Any            # ClassifierOutput | None
    intent: str                       # "chitchat" | "structured" | "semantic"
    structured_result: Optional[dict[str, Any]]
    plan_subqueries: list[str]
    tool_trace: list[dict[str, Any]]
    seen_calls: list[str]             # set serialized as list for state pickle
    step_count: int
    decide_decision: Optional[dict[str, Any]]
    packed_context: list[dict[str, Any]]

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
    sql_query: Optional[str]
    timings_ms: dict[str, float]


_INNER = AgentService()
_CHECKPOINTER = InMemorySaver()


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


async def classify(state: ChatState) -> dict[str, Any]:
    if not settings.STRUCTURED_REASONING_ENABLED:
        return {"classifier_output": None, "intent": "semantic"}
    output = await _INNER.classifier.classify(
        question=state["question"],
        document_title=state.get("document_title"),
        chat_history=state.get("chat_history"),
    )
    return {
        "classifier_output": output,
        "intent": "structured" if output.intent == "structured" else "semantic",
    }


async def structured_run(state: ChatState) -> dict[str, Any]:
    co = state.get("classifier_output")
    result = await _INNER.structured_pipeline.run(
        question=state["question"],
        document_title=state.get("document_title"),
        chat_history=state.get("chat_history"),
        query_type=getattr(co, "query_type", None) or "comparison",
        classifier_confidence=getattr(co, "confidence", 0.0),
    )
    if result.get("_structured_fallback"):
        # Fallthrough to semantic path.
        return {"structured_result": None, "intent": "semantic"}
    return {"structured_result": result}


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
    classifier_output = state.get("classifier_output")

    # Plan subqueries first (parallel).
    if state.get("plan_subqueries"):
        started = time.perf_counter()
        results = await _asyncio.gather(
            *[
                _INNER.knowledge.bootstrap_search(
                    query=sq, document_title=doc_title, intent=classifier_output,
                )
                for sq in state["plan_subqueries"]
            ],
            return_exceptions=True,
        )
        tool_latency_ms += (time.perf_counter() - started) * 1000
        for sq, res in zip(state["plan_subqueries"], results):
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
                "tool_name": "search_hybrid_kg",
                "tool_input": sub_in,
                "tool_output": sub_out,
                "plan_subquery": sq,
            })

    # Always run a final bootstrap on the original question.
    boot_in, boot_out = await _INNER.knowledge.bootstrap_search(
        query=state["question"], document_title=doc_title, intent=classifier_output,
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


async def fast_answer(state: ChatState) -> dict[str, Any]:
    """Adaptive fast path: single retrieve + single-shot answer, skipping the
    plan->decide->tool loop. Used only for high-confidence simple single-domain
    questions (WS4)."""
    doc_title = state.get("document_title")
    co = state.get("classifier_output")
    boot_in, boot_out = await _INNER.knowledge.bootstrap_search(
        query=state["question"], document_title=doc_title, intent=co,
    )
    boot_out = _INNER.security.filter_tool_results(tool_output=boot_out, document_title=doc_title)
    trace = [{"tool_name": "search_hybrid_kg", "tool_input": boot_in, "tool_output": boot_out}]

    assembled = await _INNER.context.assemble(state["question"], [boot_out])
    packed = assembled.get("packed_context", []) if isinstance(assembled, dict) else assembled

    started = time.perf_counter()
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
        "answer_latency_ms": (time.perf_counter() - started) * 1000,
        "reasoning_path": "fast",
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


async def ground(state: ChatState) -> dict[str, Any]:
    # Structured path bypasses ground (already has citations).
    if state.get("structured_result"):
        sr = state["structured_result"]
        elapsed = (time.perf_counter() - state["total_started"]) * 1000
        return {
            "answer": sr.get("answer", ""),
            "citations": sr.get("citations", []),
            "highlights": sr.get("highlights", []),
            "reasoning_path": "structured",
            "sql_query": sr.get("sql_query"),
            "timings_ms": {"total": round(elapsed, 2)},
        }

    # Cite by source number: the answer's inline [n] = packed_context position,
    # so the UI citation list must be the full ordered packed context (tagged
    # with `source` = n), not the model's free-form citation subset.
    grounded = _INNER._build_packed_citations(state.get("packed_context") or [])
    await _attach_source_ids(grounded)
    elapsed = (time.perf_counter() - state["total_started"]) * 1000
    timings = {
        "total": round(elapsed, 2),
        "decide": round(state.get("decide_latency_ms", 0.0), 2),
        "tool": round(state.get("tool_latency_ms", 0.0), 2),
        "assemble": round(state.get("assemble_latency_ms", 0.0), 2),
        "answer": round(state.get("answer_latency_ms", 0.0), 2),
        "plan": round(state.get("plan_latency_ms", 0.0), 2),
    }
    return {
        "citations": grounded,
        "reasoning_path": "semantic",
        "timings_ms": timings,
    }


# ── Routers (conditional edges) ──────────────────────────────────────────────

def _route_chitchat(state: ChatState) -> str:
    return "chitchat_answer" if state.get("is_chitchat") else "classify"


def _route_intent(state: ChatState) -> str:
    if state.get("intent") == "structured":
        return "structured_run"
    co = state.get("classifier_output")
    if (
        settings.ADAPTIVE_ROUTING_ENABLED
        and co is not None
        and getattr(co, "complexity", "complex") == "simple"
        and getattr(co, "single_domain", False)
        and getattr(co, "confidence", 0.0) >= settings.ADAPTIVE_FASTPATH_MIN_CONFIDENCE
    ):
        return "fast_answer"
    return "semantic_plan"


def _route_structured(state: ChatState) -> str:
    # If structured_result is set, we've succeeded → ground (which exports it).
    # If fallback (None), continue semantic path.
    return "ground" if state.get("structured_result") else "semantic_plan"


def _route_decide(state: ChatState) -> str:
    decision = state.get("decide_decision") or {}
    if decision.get("done"):
        return "assemble"
    if state.get("step_count", 0) >= max(settings.AGENT_MAX_STEPS - 1, 1):
        return "assemble"
    return "tool_exec"


# ── Build graph ──────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(ChatState)
    g.add_node("validate", validate)
    g.add_node("memory", memory)
    g.add_node("chitchat_check", chitchat_check)
    g.add_node("chitchat_answer", chitchat_answer)
    g.add_node("classify", classify)
    g.add_node("structured_run", structured_run)
    g.add_node("semantic_plan", semantic_plan)
    g.add_node("bootstrap", bootstrap)
    g.add_node("fast_answer", fast_answer)
    g.add_node("decide", decide)
    g.add_node("tool_exec", tool_exec)
    g.add_node("assemble", assemble)
    g.add_node("answer", answer_node)
    g.add_node("ground", ground)

    g.add_edge(START, "validate")
    g.add_edge("validate", "memory")
    g.add_edge("memory", "chitchat_check")
    g.add_conditional_edges("chitchat_check", _route_chitchat,
                            {"chitchat_answer": "chitchat_answer", "classify": "classify"})
    g.add_edge("chitchat_answer", END)
    g.add_conditional_edges("classify", _route_intent,
                            {"structured_run": "structured_run",
                             "semantic_plan": "semantic_plan",
                             "fast_answer": "fast_answer"})
    g.add_conditional_edges("structured_run", _route_structured,
                            {"ground": "ground", "semantic_plan": "semantic_plan"})
    g.add_edge("semantic_plan", "bootstrap")
    g.add_edge("bootstrap", "decide")
    g.add_conditional_edges("decide", _route_decide,
                            {"tool_exec": "tool_exec", "assemble": "assemble"})
    g.add_edge("tool_exec", "decide")
    g.add_edge("assemble", "answer")
    g.add_edge("answer", "ground")
    g.add_edge("fast_answer", "ground")
    g.add_edge("ground", END)
    return g.compile(checkpointer=_CHECKPOINTER)


_GRAPH = _build_graph()


class GraphAgentService:
    """Drop-in replacement for AgentService with LangGraph orchestrator (v2 nodes)."""

    async def chat(
        self,
        question: str,
        document_title: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        conversation_id: str | None = None,
        domain_filter: dict[str, Any] | None = None,
        verbosity: str | None = None,
    ) -> dict[str, Any]:
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
        state = await _GRAPH.ainvoke(initial, config=config)
        return {
            "question": question,
            "document_title": document_title,
            "answer": state.get("answer", ""),
            "citations": state.get("citations", []),
            "tool_trace": state.get("tool_trace", []),
            "reasoning_path": state.get("reasoning_path", "semantic"),
            "sql_query": state.get("sql_query"),
            "highlights": state.get("highlights", []),
            "timings_ms": state.get("timings_ms", {}),
            # Retrieved+packed passages used to synthesize the answer. Exposed so
            # eval (RAGAS contexts) and clients can inspect grounding evidence.
            "context": state.get("packed_context", []),
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
        """Streaming not yet implemented for v2 — fall back to inner."""
        async for chunk in _INNER.chat_stream(
            question=question,
            document_title=document_title,
            chat_history=chat_history or [],
            conversation_id=conversation_id,
            model_override=model_override,
            verbosity=verbosity,
        ):
            yield chunk
