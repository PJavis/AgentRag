"""Assign system_tag + specialty_tag + canonical_terms to chunks.

Strategy:
  1. Parse section_path into segments split by '/'.
  2. Resolve each segment via TermResolver. Later segments (more specific
     children) overwrite system_tag; specialty_tags + canonical_terms accumulate.
  3. If section_path is generic (e.g. "Tổng quan", "Phần 1") OR didn't yield
     a system_tag, fall back to scanning chunk content with find_in_text().
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.agentrag.ontology.resolver import TermResolver


_GENERIC_HEADINGS = {
    "tong quan", "mo dau", "gioi thieu", "loi noi dau", "loi mo dau",
    "phan 1", "phan 2", "phan 3", "phan 4", "phan 5",
    "muc 1", "muc 2", "muc 3",
    "chuong 1", "chuong 2", "chuong 3", "chuong 4", "chuong 5",
    "tai lieu tham khao", "muc luc", "phu luc",
    "chu giai", "danh sach",
}


def _norm(s: str) -> str:
    d = unicodedata.normalize("NFD", s or "")
    a = "".join(c for c in d if unicodedata.category(c) != "Mn")
    a = a.replace("đ", "d").replace("Đ", "d")
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", a.lower()).split())


class SectionTagger:
    def __init__(self, resolver: TermResolver | None = None) -> None:
        self._resolver = resolver or TermResolver()

    async def tag_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        section_path: str = chunk.get("section_path") or ""
        content: str = chunk.get("content") or ""

        system_tag: str | None = None
        specialty_set: set[str] = set()
        canonical_set: set[str] = set()

        for seg in [s.strip() for s in section_path.split("/") if s.strip()]:
            if _norm(seg) in _GENERIC_HEADINGS:
                continue
            r = await self._resolver.resolve(seg, strict=True)
            if r is None:
                continue
            if r.system_tag:
                system_tag = r.system_tag
            specialty_set.update(r.specialty_tags)
            canonical_set.add(r.canonical)

        # Fall back to chunk content when section_path didn't yield a system tag.
        if system_tag is None:
            content_hits = await self._resolver.find_in_text(content, max_terms=5)
            for h in content_hits:
                if system_tag is None and h.system_tag:
                    system_tag = h.system_tag
                specialty_set.update(h.specialty_tags)
                canonical_set.add(h.canonical)

        return {
            **chunk,
            "system_tag": system_tag,
            "specialty_tag": sorted(specialty_set),
            "canonical_terms": sorted(canonical_set),
        }
