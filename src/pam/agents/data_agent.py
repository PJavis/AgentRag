from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.pam.services.knowledge_service import KnowledgeService
from src.pam.structured.pipeline import StructuredReasoningPipeline
from src.pam.structured.query_classifier import ClassifierOutput, QueryIntentClassifier


@dataclass
class DataTask:
    question: str
    document_title: str | None = None
    query_type: str = "comparison"


@dataclass
class DataResult:
    question: str
    rows: list[dict[str, Any]]           # SQL result rows (structured) hoặc text chunks (semantic)
    sql_query: str | None
    reasoning_path: str                  # "structured" hoặc "semantic"
    source_chunks: list[dict[str, Any]]  # raw chunks cho provenance
    citations: list[dict[str, Any]]


class DataAgent:
    """
    Specialized worker: thu thập data cho một sub-question.

    Nếu câu hỏi có structure → dùng StructuredReasoningPipeline.
    Else → dùng KnowledgeService.bootstrap_search (semantic retrieval).
    """

    def __init__(
        self,
        knowledge_service: KnowledgeService,
        structured_pipeline: StructuredReasoningPipeline,
        classifier: QueryIntentClassifier,
    ) -> None:
        self._knowledge = knowledge_service
        self._pipeline = structured_pipeline
        self._classifier = classifier

    async def run(self, task: DataTask) -> DataResult:
        intent = await self._classifier.classify(
            question=task.question,
            document_title=task.document_title,
        )

        if intent.intent == "structured":
            result = await self._pipeline.run(
                question=task.question,
                document_title=task.document_title,
                chat_history=None,
                query_type=intent.query_type or task.query_type,
                classifier_confidence=intent.confidence,
            )
            if not result.get("_structured_fallback"):
                return DataResult(
                    question=task.question,
                    rows=[],  # SQL rows đã embedded trong answer
                    sql_query=result.get("sql_query"),
                    reasoning_path="structured",
                    source_chunks=result.get("context") or [],
                    citations=result.get("citations") or [],
                )

        # Semantic fallback
        _, tool_output = await self._knowledge.bootstrap_search(
            query=task.question,
            document_title=task.document_title,
        )
        chunks = tool_output.get("results") or []
        return DataResult(
            question=task.question,
            rows=[],
            sql_query=None,
            reasoning_path="semantic",
            source_chunks=chunks,
            citations=[],
        )
