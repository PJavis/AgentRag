"""tei backend — bge-reranker-v2-m3 served by a TEI container (/rerank), no HTTP."""
from __future__ import annotations

import asyncio
import copy
import pytest

from src.agentrag.config import settings
from src.agentrag.config_validation import _validate_retrieval_reranker_settings
from src.agentrag.retrieval.reranker import LLMReranker


def _tei_reranker(monkeypatch, responses):
    """LLMReranker on the tei backend whose HTTP call returns `responses` and
    records every (url, payload) it was sent."""
    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_BACKEND", "tei")
    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_TEI_URL", "http://tei-rerank:80")
    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_TOP_N", 20)
    r = LLMReranker()
    sent: list[tuple[str, dict]] = []

    async def fake_post(url, payload):
        sent.append((url, payload))
        if isinstance(responses, Exception):
            raise responses
        return responses

    monkeypatch.setattr(r, "_tei_post", fake_post)
    return r, sent


def _cands():
    return [
        {"id": "a", "content": "alpha " * 400},
        {"id": "b", "content": "beta"},
        {"id": "c", "content": "gamma"},
    ]


def test_tei_backend_has_no_api_client(monkeypatch):
    r, _ = _tei_reranker(monkeypatch, [])
    assert r.backend == "tei"
    assert r.client is None


def test_tei_orders_by_score_and_matches_local_rerank_score(monkeypatch):
    # TEI (raw_scores=false) returns sigmoid(logit), the same probability
    # CrossEncoder.predict returns; it is stored as rerank_score unchanged.
    r, _ = _tei_reranker(
        monkeypatch,
        [{"index": 1, "score": 0.9}, {"index": 2, "score": 0.5}, {"index": 0, "score": 0.1}],
    )
    out, ok, reason = asyncio.run(r.maybe_rerank("q", _cands(), top_k=3, force=True))
    assert ok and reason == "ok_tei"
    assert [c["id"] for c in out] == ["b", "c", "a"]
    assert [c["rerank_score"] for c in out] == pytest.approx([0.9, 0.5, 0.1])


def test_tei_request_matches_local_cross_encoder_inputs(monkeypatch):
    r, sent = _tei_reranker(monkeypatch, [{"index": 0, "score": 0.1}])
    asyncio.run(r.maybe_rerank("câu hỏi", _cands(), top_k=3, force=True))
    url, payload = sent[0]
    assert url == "http://tei-rerank:80/rerank"
    assert payload["query"] == "câu hỏi"
    assert payload["raw_scores"] is False
    assert payload["truncate"] is True
    # Same 1600-char cut as the local cross-encoder path.
    assert len(payload["texts"]) == 3
    assert len(payload["texts"][0]) == 1600


def test_tei_failure_falls_back_to_original_order(monkeypatch):
    r, _ = _tei_reranker(monkeypatch, ConnectionError("down"))
    out, ok, reason = asyncio.run(r.maybe_rerank("q", _cands(), top_k=3, force=True))
    assert not ok
    assert reason.startswith("tei_exception:")
    assert [c["id"] for c in out] == ["a", "b", "c"]
    assert all("rerank_score" not in c for c in out)


def test_tei_malformed_response_falls_back(monkeypatch):
    r, _ = _tei_reranker(monkeypatch, {"error": "model not loaded"})
    out, ok, reason = asyncio.run(r.maybe_rerank("q", _cands(), top_k=3, force=True))
    assert not ok
    assert reason == "tei_no_rankable_candidates"
    assert [c["id"] for c in out] == ["a", "b", "c"]


def test_tei_validation_requires_tei_url():
    s = copy.copy(settings)
    s.RETRIEVAL_RERANK_ENABLED = True
    s.RETRIEVAL_RERANK_BACKEND = "tei"
    s.RETRIEVAL_RERANK_TEI_URL = None
    with pytest.raises(ValueError, match="RETRIEVAL_RERANK_TEI_URL"):
        _validate_retrieval_reranker_settings(s)


def test_tei_validation_passes_with_tei_url():
    s = copy.copy(settings)
    s.RETRIEVAL_RERANK_ENABLED = True
    s.RETRIEVAL_RERANK_BACKEND = "tei"
    s.RETRIEVAL_RERANK_TEI_URL = "http://127.0.0.1:8081"
    _validate_retrieval_reranker_settings(s)  # no raise


def test_old_floor_setting_fails_with_converted_value():
    s = copy.copy(settings)
    s.RETRIEVAL_RELEVANCE_FLOOR = 0.55
    s.ANSWERABILITY_GRAY_MARGIN = 0.13
    from src.agentrag.config_validation import _validate_relevance_scale_settings

    with pytest.raises(ValueError) as exc:
        _validate_relevance_scale_settings(s)
    assert "RETRIEVAL_RELEVANCE_MIN_PROB=0.2007" in str(exc.value)
    assert "ANSWERABILITY_GRAY_MARGIN_PROB=0.5531" in str(exc.value)


def test_new_scale_settings_pass_validation():
    s = copy.copy(settings)
    s.RETRIEVAL_RELEVANCE_FLOOR = None
    s.ANSWERABILITY_GRAY_MARGIN = None
    from src.agentrag.config_validation import _validate_relevance_scale_settings

    _validate_relevance_scale_settings(s)  # no raise
