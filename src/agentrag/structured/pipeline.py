from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agentrag.services.knowledge_service import KnowledgeService
    from src.agentrag.services.llm_gateway import LLMGateway
    from src.agentrag.services.security_service import SecurityService

from src.agentrag.common.tracing import StageTracer
from src.agentrag.structured.extractor import StructuredExtractor
from src.agentrag.structured.schema_discovery import SchemaDiscoveryModule
from src.agentrag.structured.sql_engine import SQLReasoningEngine
from src.agentrag.structured.synthesizer import AnswerSynthesizer


class StructuredReasoningPipeline:
    """
    Orchestrator cho nhánh SQL reasoning (ADR 0002).

    Flow: Retrieval → Schema Discovery → Extraction → SQL → Synthesis.
    Graceful fallback về nhánh semantic tại bất kỳ bước nào thất bại.

    Return schema tương thích với AgentService.chat() + 2 fields mới:
      "reasoning_path": "structured"
      "sql_query": str
    """

    def __init__(
        self,
        knowledge_service: KnowledgeService,
        llm_gateway: LLMGateway,
        security_service: SecurityService,
    ) -> None:
        self._knowledge = knowledge_service
        self._security = security_service
        self._schema_discovery = SchemaDiscoveryModule(llm_gateway)
        self._extractor = StructuredExtractor(llm_gateway)
        self._sql_engine = SQLReasoningEngine(llm_gateway)
        self._synthesizer = AnswerSynthesizer(llm_gateway)

    async def run(
        self,
        question: str,
        document_title: str | None,
        chat_history: list[dict[str, Any]] | None,
        query_type: str,
        classifier_confidence: float,
    ) -> dict[str, Any]:
        """
        Trả về:
        - Nếu thành công: dict tương thích AgentService.chat() với reasoning_path="structured"
        - Nếu fallback cần: {"_structured_fallback": True, "_fallback_reason": str, ...}
        """
        tracer = StageTracer()
        total_started = time.perf_counter()

        # ── Step 1: Retrieval ─────────────────────────────────────────────────
        try:
            tracer.start("retrieve", "KnowledgeService", query=question)
            tool_input, tool_output = await self._knowledge.bootstrap_search(
                query=question,
                document_title=document_title,
            )
            tool_output = self._security.filter_tool_results(tool_output, document_title)
            chunks: list[dict[str, Any]] = tool_output.get("results") or []
            tracer.end("retrieve", chunk_count=len(chunks))
        except Exception as exc:
            tracer.fail("retrieve", exc)
            return self._fallback_result(question, document_title, f"retrieve_failed:{exc}", tracer)

        # ── Step 2: Schema Discovery ──────────────────────────────────────────
        try:
            tracer.start("schema_discovery", "SchemaDiscoveryModule")
            schema = await self._schema_discovery.discover(
                question=question,
                query_type=query_type,
                candidate_chunks=chunks,
                document_title=document_title,
            )
            tracer.end("schema_discovery", tables=[t.name for t in schema.tables], is_empty=schema.is_empty)
            if schema.is_empty:
                return self._fallback_result(question, document_title, "empty_schema", tracer)
        except Exception as exc:
            tracer.fail("schema_discovery", exc)
            return self._fallback_result(question, document_title, f"schema_failed:{exc}", tracer)

        # ── Step 3: Extraction ────────────────────────────────────────────────
        try:
            tracer.start("extraction", "StructuredExtractor")
            extraction = await self._extractor.extract(
                chunks=chunks,
                schema=schema,
                question=question,
            )
            tracer.end(
                "extraction",
                total_rows=sum(s.total_rows for s in extraction.stats),
                valid_rows=sum(s.valid_rows for s in extraction.stats),
                is_empty=extraction.is_empty,
            )
            if extraction.is_empty:
                return self._fallback_result(question, document_title, "empty_extraction", tracer)
        except Exception as exc:
            tracer.fail("extraction", exc)
            return self._fallback_result(question, document_title, f"extract_failed:{exc}", tracer)

        # ── Step 4: SQL Reasoning ─────────────────────────────────────────────
        try:
            tracer.start("sql", "SQLReasoningEngine")
            sql_result = await self._sql_engine.execute(
                question=question,
                schema=schema,
                database=extraction.database,
                query_type=query_type,
            )
            tracer.end("sql", retry_count=sql_result.retry_count, execution_ok=sql_result.execution_ok)
            if not sql_result.execution_ok:
                return self._fallback_result(
                    question, document_title,
                    sql_result.fallback_reason or "sql_failed", tracer,
                )
        except Exception as exc:
            tracer.fail("sql", exc)
            return self._fallback_result(question, document_title, f"sql_exception:{exc}", tracer)

        # ── Step 5: Answer Synthesis ──────────────────────────────────────────
        try:
            tracer.start("synthesize", "AnswerSynthesizer")
            synthesis = await self._synthesizer.synthesize(
                question=question,
                sql_result=sql_result,
                candidate_chunks=chunks,
                query_type=query_type,
                chat_history=chat_history,
            )
            tracer.end("synthesize")
        except Exception as exc:
            tracer.fail("synthesize", exc)
            # Softer fallback: trả raw SQL result thay vì quay về semantic
            return self._raw_sql_result(question, document_title, sql_result, chunks, tracer, total_started)

        total_latency_ms = (time.perf_counter() - total_started) * 1000

        return {
            "question": question,
            "document_title": document_title,
            "answer": synthesis.answer,
            "citations": synthesis.citations,
            "tool_trace": self._build_tool_trace(tracer, tool_input, tool_output, sql_result),
            "context": self._build_packed_context(chunks),
            "reasoning_path": "structured",
            "sql_query": sql_result.sql_query,
            "timings_ms": {
                "total": round(total_latency_ms, 2),
                **tracer.as_timings_dict(),
            },
        }

    # ── Fallback helpers ──────────────────────────────────────────────────────

    def _fallback_result(
        self,
        question: str,
        document_title: str | None,
        reason: str,
        tracer: StageTracer,
    ) -> dict[str, Any]:
        """Sentinel dict — AgentService sẽ detect và route về semantic path."""
        return {
            "_structured_fallback": True,
            "_fallback_reason": reason,
            "question": question,
            "document_title": document_title,
            "_trace": tracer.as_dict(),
        }

    def _raw_sql_result(
        self,
        question: str,
        document_title: str | None,
        sql_result: "SQLEngineOutput",
        chunks: list[dict[str, Any]],
        tracer: StageTracer,
        total_started: float,
    ) -> dict[str, Any]:
        """Softer fallback: trả dữ liệu SQL thô khi synthesis thất bại."""
        from src.agentrag.structured.synthesizer import AnswerSynthesizer as S
        synth = S.__new__(S)
        formatted = synth._format_result(sql_result.result_rows, "comparison")
        total_latency_ms = (time.perf_counter() - total_started) * 1000
        return {
            "question": question,
            "document_title": document_title,
            "answer": f"Kết quả truy vấn:\n{formatted}",
            "citations": [],
            "tool_trace": [],
            "context": self._build_packed_context(chunks),
            "reasoning_path": "structured",
            "sql_query": sql_result.sql_query,
            "timings_ms": {
                "total": round(total_latency_ms, 2),
                **tracer.as_timings_dict(),
            },
        }

    # ── Context helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_packed_context(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Same schema với ContextAssembler._stage_citation_pack."""
        return [
            {
                "document_title": c.get("document_title"),
                "section_path": c.get("section_path"),
                "position": c.get("position"),
                "content_hash": c.get("content_hash"),
                "excerpt": (c.get("content") or "")[:1500],
            }
            for c in chunks
        ]

    @staticmethod
    def _build_tool_trace(
        tracer: StageTracer,
        tool_input: dict[str, Any],
        tool_output: dict[str, Any],
        sql_result: "SQLEngineOutput",
    ) -> list[dict[str, Any]]:
        trace = [
            {
                "tool_name": "bootstrap_search",
                "tool_input": tool_input,
                "tool_output": {
                    "mode": tool_output.get("mode"),
                    "result_count": len(tool_output.get("results") or []),
                },
            },
            {
                "tool_name": "sql_reasoning",
                "sql_query": sql_result.sql_query,
                "result_rows": len(sql_result.result_rows),
                "retry_count": sql_result.retry_count,
            },
        ]
        # Append stage events
        for event in tracer.events:
            trace.append({
                "stage": event.stage,
                "service": event.service,
                "elapsed_ms": event.elapsed_ms,
                "metadata": event.metadata,
                **({"error": event.error} if event.error else {}),
            })
        return trace
