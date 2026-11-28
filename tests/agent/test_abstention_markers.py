"""A clean abstention drops its citations, so it must not keep [n] markers.

Live 2026-11-28: "Thuốc điều trị đầu tay tăng huyết áp ở người trẻ..." answered
"không có nội dung nào ... [1][5], không phải ..." with citations == []. The UI
turns [n] into a link to citation n, so the refusal showed two dead links.
"""
from __future__ import annotations

import asyncio
import time

from src.agentrag.agent import graph_service as gs
from src.agentrag.agent.service import _strip_citation_markers

_THIN = [{"rerank_score": 0.0, "content": "x", "document_title": "d"}]
_STRONG = [{"rerank_score": 0.9, "content": "x", "document_title": "d"}]


def _ground(answer, packed, monkeypatch):
    async def _noop(_c):
        return None

    monkeypatch.setattr(gs, "_attach_source_ids", _noop)
    monkeypatch.setattr(gs.settings, "ANSWER_ABSTAIN_ON_THIN_CONTEXT", True)
    return asyncio.run(gs.ground({"answer": answer, "packed_context": packed,
                                  "total_started": time.perf_counter()}))


def test_dropped_citations_take_their_markers_with_them(monkeypatch):
    out = _ground("Không có thông tin về thuốc này (ví dụ \"tăng huyết áp\") [1][5], không phải hướng dẫn.",
                  _THIN, monkeypatch)
    assert out["citations"] == []
    assert out["answer"] == "Không có thông tin về thuốc này (ví dụ \"tăng huyết áp\"), không phải hướng dẫn."


def test_kept_citations_keep_their_markers(monkeypatch):
    answer = "NMCT là tắc mạch vành [1]."
    out = _ground(answer, _STRONG, monkeypatch)
    assert out["citations"]
    assert "answer" not in out or out["answer"] == answer


def test_strip_leaves_markdown_links_and_other_brackets():
    assert _strip_citation_markers("Xem [tài liệu](http://x/1) và [Hình 2] [3].") == \
        "Xem [tài liệu](http://x/1) và [Hình 2]."
    assert _strip_citation_markers("Không tìm thấy [12] trong tài liệu.") == "Không tìm thấy trong tài liệu."
    assert _strip_citation_markers("Câu một [1].\nCâu hai [2]") == "Câu một.\nCâu hai"
