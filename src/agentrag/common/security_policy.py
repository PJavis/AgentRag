from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentPolicy:
    """
    Policy kiểm soát truy cập ở mức document/section.
    Không cần user model — hoạt động dựa trên metadata của chunks.
    """

    document_title: str

    # Section path bắt đầu bằng prefix này sẽ bị loại (ví dụ: "internal/", "draft/")
    denied_section_prefixes: list[str] = field(default_factory=list)

    # Regex patterns match với section_path sẽ bị loại
    denied_section_patterns: list[str] = field(default_factory=list)

    # Loại segment bị loại (ví dụ: ["confidential_table"])
    denied_segment_types: list[str] = field(default_factory=list)

    # Giới hạn số kết quả tối đa trả về (None = không giới hạn)
    max_results: int | None = None

    def __post_init__(self) -> None:
        self._compiled_patterns: list[re.Pattern] = [
            re.compile(p, re.IGNORECASE) for p in self.denied_section_patterns
        ]

    def matches_denied_section(self, section_path: str | None) -> bool:
        if not section_path:
            return False
        for prefix in self.denied_section_prefixes:
            if section_path.startswith(prefix):
                return True
        for pattern in self._compiled_patterns:
            if pattern.search(section_path):
                return True
        return False

    def matches_denied_segment_type(self, segment_type: str | None) -> bool:
        if not segment_type or not self.denied_segment_types:
            return False
        return segment_type in self.denied_segment_types


class PolicyRegistry:
    """
    In-memory store các DocumentPolicy.
    Populated tại startup từ config hoặc file JSON.
    Read-only sau khi init — thread-safe.
    """

    def __init__(self) -> None:
        self._policies: dict[str, DocumentPolicy] = {}

    def load_from_list(self, policies: list[dict[str, Any]]) -> None:
        """
        Load policies từ list of dicts. Format mỗi dict:
        {
            "document_title": str,
            "denied_section_prefixes": list[str],   # optional
            "denied_section_patterns": list[str],   # optional
            "denied_segment_types": list[str],      # optional
            "max_results": int | null,              # optional
        }
        """
        for item in policies:
            title = item.get("document_title", "")
            if not title:
                continue
            policy = DocumentPolicy(
                document_title=title,
                denied_section_prefixes=item.get("denied_section_prefixes", []),
                denied_section_patterns=item.get("denied_section_patterns", []),
                denied_segment_types=item.get("denied_segment_types", []),
                max_results=item.get("max_results"),
            )
            self._policies[title] = policy

    def get(self, document_title: str) -> DocumentPolicy | None:
        return self._policies.get(document_title)

    def has_policy(self, document_title: str) -> bool:
        return document_title in self._policies

    def all_titles(self) -> list[str]:
        return list(self._policies.keys())


if __name__ == "__main__":
    registry = PolicyRegistry()
    registry.load_from_list([
        {
            "document_title": "SecretDoc",
            "denied_section_prefixes": ["internal/", "draft/"],
            "denied_section_patterns": [r"confidential"],
            "denied_segment_types": ["secret_table"],
            "max_results": 3,
        }
    ])

    policy = registry.get("SecretDoc")
    assert policy is not None

    assert policy.matches_denied_section("internal/pricing") is True
    assert policy.matches_denied_section("public/overview") is False
    assert policy.matches_denied_section("draft/v2") is True
    assert policy.matches_denied_section("section/confidential_data") is True
    assert policy.matches_denied_segment_type("secret_table") is True
    assert policy.matches_denied_segment_type("text") is False
    assert registry.get("OtherDoc") is None

    print("[OK] PolicyRegistry + DocumentPolicy smoke test passed")
