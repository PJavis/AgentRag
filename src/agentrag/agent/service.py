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


_VERBOSE_TOKENS = (
    "dài hơn", "chi tiết", "kỹ hơn", "kỹ càng", "tỉ mỉ", "đầy đủ hơn",
    "mở rộng", "giải thích thêm", "nói rõ", "rõ hơn", "sâu hơn",
    "thêm chi tiết", "expand", "elaborate", "in detail", "more detail",
    "longer", "go deeper", "explain more",
)

# Summarization / overview intent — needs structured multi-section output even
# without explicit "detailed" keyword. Trigger verbose mode automatically.
_SUMMARY_TOKENS = (
    "tóm tắt", "tổng hợp", "tổng quan", "khái quát", "overview",
    "summary", "summarize", "summarise", "recap",
    "trình bày", "trình bầy", "giới thiệu",
    "nội dung tài liệu", "nội dung của tài liệu", "main points",
    "key points", "the key", "phân tích",
)


def _is_verbose_followup(question: str) -> bool:
    """Detect user asking for a longer / more detailed elaboration of the prior turn,
    OR a summary/overview request that needs structured multi-section output."""
    q = (question or "").lower()
    return (
        any(tok in q for tok in _VERBOSE_TOKENS)
        or any(tok in q for tok in _SUMMARY_TOKENS)
    )


# Shared Markdown formatting rules. Both the semantic _answer prompt and the
# structured AnswerSynthesizer use this so every assistant turn renders with
# bold highlights / blockquote warnings / GFM tables / LaTeX formulas in the
# ReactMarkdown frontend (remark-gfm + remark-math + rehype-katex).
MARKDOWN_FORMAT_RULES = (
    "FORMAT: `answer` field is Markdown. Required: "
    "**bold** key terms (drug names, doses, lab values, diagnoses, percentages); "
    "bullet lists `- ` for parallel facts; `### Heading` between sections; "
    "GFM tables `| a | b |` for comparisons; LaTeX `$...$` or `$$...$$` for formulas "
    "(eGFR, BMI, dose calc, ratios, statistics); "
    "`> blockquote` only for safety warnings or contraindications. "
    "Each section heading on its own line, body below. Never inline heading with body."
)


_CHITCHAT_SYSTEM_PROMPT = (
    "You are a warm, friendly research companion. The user is making small talk "
    "(greeting / thanks / casual remark) — reply briefly and naturally, like a friend. "
    "1–3 sentences max. Do NOT pretend to retrieve documents. Do NOT cite anything. "
    "Match the user's language (Vietnamese ↔ English). Keep it light."
)

from src.agentrag.agent.context import _lost_in_middle_reorder
from src.agentrag.config import settings
from src.agentrag.services import (
    ContextAssemblyService,
    KnowledgeService,
    LLMGateway,
    SecurityService,
)
from src.agentrag.structured.pipeline import StructuredReasoningPipeline
from src.agentrag.structured.query_classifier import QueryIntentClassifier


_PRIMARY_ANSWER_KEYS = (
    "answer", "summary", "response", "output", "text", "content",
    "result", "explanation", "tóm_tắt", "câu_trả_lời",
)


def _find_answer_field(obj: Any, max_depth: int = 6) -> str | None:
    """Find first non-empty answer/summary/response string in nested dict/list.

    Some finetuned models ignore the prompt and wrap output in odd shapes
    like {"search_results":[{"result":[{"answer":"..."}]}]} or simply switch
    `answer` → `summary` when the question is a summarization request. Walk
    the tree instead of giving up.
    """
    if max_depth <= 0:
        return None
    if isinstance(obj, dict):
        for key in _PRIMARY_ANSWER_KEYS:
            v = obj.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, dict):
                nested = _find_answer_field(v, max_depth - 1)
                if nested:
                    return nested
            if isinstance(v, list):
                # Some models return content as list of strings/sections.
                parts = []
                for item in v:
                    if isinstance(item, str) and item.strip():
                        parts.append(item.strip())
                    elif isinstance(item, dict):
                        nested = _find_answer_field(item, max_depth - 1)
                        if nested:
                            parts.append(nested)
                if parts:
                    return "\n\n".join(parts)
        # Fall back: recurse into remaining values.
        for k, v in obj.items():
            if k in _PRIMARY_ANSWER_KEYS:
                continue
            if isinstance(v, (dict, list)):
                nested = _find_answer_field(v, max_depth - 1)
                if nested:
                    return nested
    elif isinstance(obj, list):
        for item in obj:
            nested = _find_answer_field(item, max_depth - 1)
            if nested:
                return nested
    return None


class AgentService:
    def __init__(self):
        # S4: fetch execution-plane services through the container so that
        # all Reasoning-Plane code shares one set of clients per process.
        from src.agentrag.services.container import get_container
        self._container = get_container()
        self.llm_gateway = self._container.llm
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

    async def chat_stream(
        self,
        question: str,
        document_title: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        conversation_id: str | None = None,
        model_override: str | None = None,
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

            assembly = await self.context.assemble(question, [e["tool_output"] for e in tool_trace])

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

            # UI model override (picker) wins for the answer step; else task routing.
            if model_override:
                from src.agentrag.agent.llm import AgentLLM
                client = AgentLLM(model_override=model_override)
            else:
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
        """Trimmed view of past turns for prompt injection.

        Returns empty when Chat StructMem is on — semantic memory replaces
        the sliding window so we avoid double-loading the same context (and
        burning tokens on stale chatter). Postgres `chat_messages` is still
        the canonical audit log and powers the UI / admin trace.
        """
        if settings.CHAT_STRUCTMEM_ENABLED:
            return []
        if not messages:
            return []
        scoped = messages[-limit:]
        return [
            {
                "role": item.get("role"),
                "content": (item.get("content") or "")[:800],
            }
            for item in scoped
        ]

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
            seg_type = ctx.get("segment_type") or citation.get("segment_type") or "text"
            mime = self._mime_for_segment(ctx, seg_type)
            excerpt = (ctx.get("excerpt") or ctx.get("content") or "")[:300]
            entry: dict[str, Any] = {
                "document_title": citation.get("document_title"),
                "section_path": citation.get("section_path"),
                "position": citation.get("position"),
                "content_hash": citation.get("content_hash"),
                "excerpt": excerpt,
                "segment_type": seg_type,
                "mime": mime,
            }
            # Page-aware fields (PDF only)
            page = ctx.get("page") or ctx.get("page_start")
            if page is not None:
                entry["page"] = page
                entry["page_label"] = f"p.{page}"
            if ctx.get("page_start") is not None:
                entry["page_start"] = ctx["page_start"]
            if ctx.get("page_end") is not None:
                entry["page_end"] = ctx["page_end"]
            # Image segments — expose URL so the UI can render <img>
            if seg_type == "image":
                meta = ctx.get("metadata") or {}
                img = meta.get("image_url") or meta.get("image_path") or ctx.get("image_url")
                if img:
                    entry["image_url"] = self._normalize_image_url(img)
            grounded.append(entry)
        return grounded

    @staticmethod
    def _mime_for_segment(ctx: dict[str, Any], seg_type: str) -> str | None:
        """Infer mime from document filename suffix; falls back by segment_type."""
        if seg_type == "image":
            return (ctx.get("metadata") or {}).get("image_mime") or "image/jpeg"
        title = (ctx.get("document_title") or "").lower()
        if title.endswith(".pdf"):
            return "application/pdf"
        if title.endswith(".docx") or title.endswith(".doc"):
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if title.endswith(".md"):
            return "text/markdown"
        if title.endswith(".txt"):
            return "text/plain"
        return None

    @staticmethod
    def _normalize_image_url(raw: str) -> str:
        """Rewrite pipeline image_path → /api/images/<rest> so the frontend can fetch."""
        if not raw:
            return raw
        if raw.startswith("http://") or raw.startswith("https://") or raw.startswith("/api/"):
            return raw
        if raw.startswith("/images/"):
            return f"/api{raw}"
        if raw.startswith("data/images/"):
            return f"/api/{raw[len('data/'):]}"
        if raw.startswith("images/"):
            return f"/api/{raw}"
        return raw

    async def _answer(
        self,
        question: str,
        packed_context: list[dict[str, Any]],
        tool_trace: list[dict[str, Any]],
        final_answer: dict[str, Any] | None,
        chat_history: list[dict[str, Any]] | None,
        memory_context: list[dict[str, Any]] | None = None,
        verbosity: str | None = None,
    ) -> dict[str, Any]:
        if final_answer and final_answer.get("answer"):
            return {
                "answer": final_answer["answer"],
                "citations": final_answer.get("citations", []),
                "highlights": final_answer.get("highlights", []),
            }

        if verbosity == "detailed":
            verbose = True
        elif verbosity == "concise":
            verbose = False
        else:
            verbose = _is_verbose_followup(question)
        length_directive = (
            "LENGTH: thorough, multi-paragraph. STRUCTURE (required): start each "
            "section with a Markdown H2 heading on its own line (e.g. `## Định nghĩa`), "
            "then the body below it. For medical/overview topics use these sections "
            "when relevant: Tổng quan, Định nghĩa, Phân loại, Nguyên nhân, Chẩn đoán, "
            "Điều trị, Tiên lượng. Use NUMBERED lists (`1.` `2.` `3.`) for "
            "classifications, causes, steps and ranked items; bullets only for "
            "non-sequential parallel facts. Bold key terms. "
            if verbose
            else
            "LENGTH: concise — focus on the question. Still use **bold** and bullets when appropriate. "
        )
        system_prompt = (
            f"{_lang_instruction(question)} "
            "Answer ONLY from the provided context. "
            "OUTPUT SCHEMA (strict): {\"answer\": \"<markdown string>\", "
            "\"citations\": [{\"document_title\": str, \"section_path\": str, "
            "\"position\": int, \"content_hash\": str}], "
            "\"highlights\": [\"sentence1\", \"sentence2\", ...]} . "
            "The top-level keys MUST be exactly \"answer\", \"citations\", \"highlights\". "
            "Do NOT name the top-level key after a tool, search method, or topic. "
            "Do NOT output \"summary\", \"search_results\", \"search_hybrid_kg\", \"response\", \"output\". "
            "Each citation must include document_title, section_path, position, content_hash. "
            "highlights: array of 3-5 strings — the most important facts or takeaways from the answer, "
            "each a complete self-contained sentence. "
            # Markdown formatting — frontend renders ReactMarkdown + GFM + KaTeX.
            f"{MARKDOWN_FORMAT_RULES}"
            f"{length_directive}"
            "When context contains multiple documents, focus on the document(s) directly relevant to the question; ignore unrelated documents. "
            "If the question is too vague to give a useful answer (e.g., which item, which document, or which aspect is missing), "
            "set answer to ONE focused clarifying question, citations to [], and highlights to []. "
            "Do not answer and ask at the same time. "
            "If context is insufficient for a specific question, say so explicitly. "
            "Answer in clear, natural sentences and avoid broken wording. "
            "Only cite claims directly supported by the provided context. "
            "INLINE CITATIONS: every context item has a numeric 'source' field. In the "
            "\"answer\" markdown, append the supporting source number(s) in square brackets "
            "immediately after each factual sentence or clause — e.g. 'Hà Nội là thủ đô [1].' "
            "Cite ONLY the source(s) whose content directly supports that claim; never cite a "
            "source that does not support it and never invent source numbers. Multiple supporting "
            "sources → [1][2]. Conversational or clarifying replies need no citations. "
            "Do NOT add examples, field names, or details not explicitly present in the context. "
            # Image-segment context: treat vision LLM descriptions as primary
            # evidence when no text segments are available (scanned PDFs).
            "If the context contains segments with segment_type='image', the 'content' "
            "field of each such segment is the vision-LLM transcription of a slide/figure "
            "from the source PDF. Treat it as legitimate textual evidence and ground your "
            "answer in it — do NOT say the document has no information just because the "
            "evidence comes from image descriptions. Translate to Vietnamese when needed. "
            "You MAY perform simple arithmetic (×, ÷, +, −) on numeric values explicitly stated in the context; "
            "show the calculation briefly (e.g. '10 × 3000 = 30,000 gold')."
        )
        history_view = self.summarize_history(chat_history, limit=10)
        # When verbose follow-up and ChatStructMem suppressed the slidng window,
        # surface the last 2 turns trimmed to 1200 chars each so 7B models don't
        # blow their context budget (was 4 turns × 4000 chars = 16k).
        if verbose and not history_view and chat_history:
            history_view = [
                {
                    "role": item.get("role"),
                    "content": (item.get("content") or "")[:1200],
                }
                for item in chat_history[-2:]
            ]
        # Lost-in-middle is applied to the PROMPT COPY only — best chunks at the
        # start + end for LLM attention — while the returned packed_context stays
        # in relevance order for eval/citation. Each item keeps its stable 'source'
        # number, so inline [n] markers still map to retrieval_context[n-1].
        prompt_context = packed_context
        if settings.AGENT_LOST_IN_MIDDLE_REORDER and len(packed_context) > 2:
            prompt_context = _lost_in_middle_reorder(packed_context)
        answer_payload: dict[str, Any] = {
            "question": question,
            "chat_history": history_view,
            "context": prompt_context,
        }
        if memory_context:
            answer_payload["conversation_memory"] = memory_context
        user_prompt = json.dumps(answer_payload, ensure_ascii=True)
        # Debug: log payload sizes to diagnose summarization failures.
        import logging as _logging
        _ctx_log = _logging.getLogger(__name__)
        # Collect image URLs from packed_context for multimodal grounding —
        # so a vision-capable answer model (Gemini 2.5/3.5 Flash, GPT-4o) can
        # read pixels, not just the vision-LLM caption that fed retrieval.
        image_urls: list[str] = []
        seen_urls: set[str] = set()
        for c in packed_context:
            if c.get("segment_type") == "image":
                url = c.get("image_url")
                if url and url not in seen_urls:
                    # If url is a relative path served by FastAPI /images
                    # static mount, prepend the public base URL when known.
                    if url.startswith("/images/") and getattr(settings, "PUBLIC_BASE_URL", None):
                        url = settings.PUBLIC_BASE_URL.rstrip("/") + url
                    image_urls.append(url)
                    seen_urls.add(url)
                    if len(image_urls) >= 4:
                        break  # cap to keep prompt budget reasonable
        _ctx_log.info(
            "_answer: question=%r packed_context_count=%d packed_chars=%d verbose=%s images=%d",
            question[:80], len(packed_context),
            sum(len(str(c.get("excerpt") or c.get("content") or "")) for c in packed_context),
            verbose, len(image_urls),
        )
        if image_urls:
            try:
                answer, _latency_ms = await self.llm_gateway.json_response_multimodal(
                    system_prompt=system_prompt,
                    user_text=user_prompt,
                    image_urls=image_urls,
                    task="answer",
                )
            except Exception:
                _ctx_log.exception("_answer: multimodal call failed, falling back to text-only")
                answer, _latency_ms = await self.llm_gateway.json_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    task="answer",
                )
        else:
            answer, _latency_ms = await self.llm_gateway.json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                task="answer",
            )
        _ctx_log.info(
            "_answer: raw_answer_keys=%s answer_field_len=%d",
            list(answer.keys()) if isinstance(answer, dict) else type(answer).__name__,
            len(str(answer.get("answer") or "")) if isinstance(answer, dict) else 0,
        )
        # Dump first 800 chars of raw answer for debugging.
        try:
            import json as _json
            _ctx_log.info(
                "_answer: raw_answer_dump=%s",
                _json.dumps(answer, ensure_ascii=False)[:800],
            )
        except Exception:
            pass
        # Recover answer if model returned wrong top-level shape.
        # Finetunes occasionally produce {"search_results":[...]} or
        # decide-shape {"done":false, "reflection":...} despite the answer prompt.
        if isinstance(answer, dict) and not answer.get("answer"):
            recovered = _find_answer_field(answer)
            if recovered:
                answer = {
                    "answer": recovered,
                    "citations": answer.get("citations") or [],
                    "highlights": answer.get("highlights") or [],
                }
            else:
                # Last resort: pull a coherent text field (reflection/reason/summary)
                # or synthesize a graceful "not found" so UI never shows empty bubble.
                fallback_text = ""
                for k in ("reflection", "reason", "summary", "explanation", "response"):
                    v = answer.get(k)
                    if isinstance(v, str) and v.strip():
                        fallback_text = v.strip()
                        break
                if not fallback_text:
                    fallback_text = (
                        "Tôi chưa tìm được thông tin chính xác cho câu hỏi này. Bạn có thể thử:\n\n"
                        "- **\"Tóm tắt tài liệu\"** — xem tổng quan toàn bộ nội dung\n"
                        "- Hỏi cụ thể hơn về một mục, trang, hoặc thuật ngữ\n"
                        "- Dùng từ khoá có trong tài liệu (tên thuốc, chẩn đoán, hệ cơ quan)\n\n"
                        "Tôi không tìm thấy thông tin đủ rõ trong tài liệu để trả lời câu hỏi này. "
                        "Bạn có thể hỏi cụ thể hơn về một mục hoặc trang nào không?"
                    )
                answer = {"answer": fallback_text, "citations": answer.get("citations") or [], "highlights": []}
        return answer
