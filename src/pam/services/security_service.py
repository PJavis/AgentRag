from __future__ import annotations

from typing import Any


class SecurityService:
    """
    Query-time policy layer (v1 skeleton).

    Current behavior is intentionally conservative:
    - validate basic input shape
    - keep room for future table/column/document-level authz
    """

    def validate_chat_request(
        self,
        question: str,
        document_title: str | None,
    ) -> None:
        if not question or not question.strip():
            raise ValueError("question must be non-empty")
        if document_title is not None and not str(document_title).strip():
            raise ValueError("document_title must be non-empty when provided")

    def filter_tool_results(
        self,
        tool_output: dict[str, Any],
        document_title: str | None,
    ) -> dict[str, Any]:
        # v1: pass-through with optional hard document filter safeguard.
        if not document_title:
            return tool_output
        results = tool_output.get("results")
        if not isinstance(results, list):
            return tool_output
        filtered = [
            item for item in results
            if item.get("document_title") in (None, document_title)
        ]
        cloned = dict(tool_output)
        cloned["results"] = filtered
        return cloned
