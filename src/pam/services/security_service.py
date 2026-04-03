from __future__ import annotations

from typing import Any

from src.pam.common.security_policy import DocumentPolicy, PolicyRegistry


class SecurityService:
    """
    Query-time policy layer (Phase B).

    Hai lớp bảo vệ:
    1. Basic input validation (validate_chat_request)
    2. Result filtering theo DocumentPolicy (filter_tool_results + apply_document_policy)

    Không cần user model — policy áp dụng theo document_title metadata.
    """

    def __init__(self, policy_registry: PolicyRegistry | None = None) -> None:
        self._registry = policy_registry or PolicyRegistry()

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
        """
        Lớp 1: hard document_title filter (giữ nguyên behavior cũ).
        Lớp 2: áp dụng DocumentPolicy nếu có.
        """
        results = tool_output.get("results")
        if not isinstance(results, list):
            return tool_output

        # Lớp 1: document_title filter (backward-compatible)
        if document_title:
            results = [
                item for item in results
                if item.get("document_title") in (None, document_title)
            ]

        # Lớp 2: policy-based filter
        if document_title and self._registry.has_policy(document_title):
            policy = self._registry.get(document_title)
            if policy is not None:
                results = self.apply_document_policy(results, policy)

        cloned = dict(tool_output)
        cloned["results"] = results
        return cloned

    def apply_document_policy(
        self,
        results: list[dict[str, Any]],
        policy: DocumentPolicy,
    ) -> list[dict[str, Any]]:
        """
        Áp dụng policy rules lên danh sách kết quả:
        - Loại chunk có section_path nằm trong denied list
        - Loại chunk có segment_type nằm trong denied list
        - Giới hạn số kết quả theo max_results
        """
        filtered: list[dict[str, Any]] = []
        for item in results:
            section_path = item.get("section_path")
            segment_type = item.get("segment_type") or item.get("metadata", {}).get("segment_type")

            if not self.validate_section_access(section_path, policy):
                continue
            if policy.matches_denied_segment_type(segment_type):
                continue
            filtered.append(item)

        if policy.max_results is not None:
            filtered = filtered[: policy.max_results]

        return filtered

    def validate_section_access(
        self,
        section_path: str | None,
        policy: DocumentPolicy,
    ) -> bool:
        """True nếu section_path được phép theo policy rules."""
        return not policy.matches_denied_section(section_path)


if __name__ == "__main__":
    from src.pam.common.security_policy import PolicyRegistry

    registry = PolicyRegistry()
    registry.load_from_list([
        {
            "document_title": "CorpDoc",
            "denied_section_prefixes": ["internal/"],
            "denied_segment_types": ["secret"],
            "max_results": 2,
        }
    ])

    svc = SecurityService(policy_registry=registry)

    # Basic validation
    svc.validate_chat_request("Hello?", None)
    try:
        svc.validate_chat_request("", None)
        assert False, "Should raise"
    except ValueError:
        pass

    # filter_tool_results — document filter (backward compat)
    out = svc.filter_tool_results(
        {
            "results": [
                {"document_title": "CorpDoc", "section_path": "public/intro", "content": "ok"},
                {"document_title": "OtherDoc", "section_path": "public/intro", "content": "filtered"},
            ]
        },
        document_title="CorpDoc",
    )
    assert len(out["results"]) == 1
    assert out["results"][0]["content"] == "ok"

    # apply_document_policy — section deny
    chunks = [
        {"section_path": "internal/pricing", "content": "secret"},
        {"section_path": "public/overview", "content": "visible"},
        {"section_path": "public/features", "content": "visible2"},
        {"section_path": "public/extra", "content": "visible3"},
    ]
    policy = registry.get("CorpDoc")
    filtered = svc.apply_document_policy(chunks, policy)
    assert len(filtered) == 2  # max_results=2
    assert all(r["section_path"].startswith("public/") for r in filtered)

    print("[OK] SecurityService smoke test passed")
