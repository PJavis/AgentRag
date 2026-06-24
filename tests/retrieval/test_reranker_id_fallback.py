"""Regression: candidates that carry only content_hash (the assemble pipeline's
dedup key) must still be rerankable. Before the fix, maybe_rerank bailed with
'no_candidate_ids' because it keyed solely on item['id'] -> rerank_score never
reached packed_context -> relevance-floor/abstain/answerability-gate all inert."""
from __future__ import annotations

import asyncio

from src.agentrag.config import settings
from src.agentrag.retrieval.reranker import LLMReranker


class _FakeCrossEncoder:
    def predict(self, pairs):
        # Rank the 'react' passage above the unrelated one.
        return [0.95 if "react" in (c or "").lower() else 0.05 for _q, c in pairs]


def _reranker(monkeypatch) -> LLMReranker:
    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_BACKEND", "local_cross_encoder")
    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_MODEL", None)
    r = LLMReranker()
    monkeypatch.setattr(r, "_get_local_cross_encoder", lambda: _FakeCrossEncoder())
    return r


def test_rerank_uses_content_hash_when_id_absent(monkeypatch):
    r = _reranker(monkeypatch)
    items = [
        {"content_hash": "h1", "content": "Bài viết về dòng điện trong dung dịch điện ly."},
        {"content_hash": "h2", "content": "React useState is a hook for state."},
    ]
    out, ok, reason = asyncio.run(r.maybe_rerank("react useState?", items, top_k=2, force=True))
    assert ok is True, reason
    assert reason == "ok_local_cross_encoder"
    # rerank_score attached to the same dict objects (rides through to packed_context)
    assert all(it.get("rerank_score") is not None for it in out)
    # the relevant passage is ranked first
    assert "react" in out[0]["content"].lower()


def test_rerank_still_works_when_id_present(monkeypatch):
    r = _reranker(monkeypatch)
    items = [
        {"id": "a", "content": "Bài viết về dòng điện trong dung dịch điện ly."},
        {"id": "b", "content": "React useState is a hook for state."},
    ]
    out, ok, reason = asyncio.run(r.maybe_rerank("react useState?", items, top_k=2, force=True))
    assert ok is True, reason
    assert all(it.get("rerank_score") is not None for it in out)
    assert "react" in out[0]["content"].lower()
