"""local_cross_encoder backend — defaults + validation (no model download)."""
from __future__ import annotations

import copy

import pytest

from src.agentrag.config import settings
from src.agentrag.config_validation import _validate_retrieval_reranker_settings
from src.agentrag.retrieval.reranker import LLMReranker


def test_local_backend_rejects_api_model_name():
    # backend=local but model is an API/chat name (the silent OSError trap):
    # CrossEncoder would try to load "gemini-2.5-flash-lite" from HuggingFace.
    s = copy.copy(settings)
    s.RETRIEVAL_RERANK_ENABLED = True
    s.RETRIEVAL_RERANK_BACKEND = "local_cross_encoder"
    s.RETRIEVAL_RERANK_MODEL = "gemini-2.5-flash-lite"
    with pytest.raises(ValueError, match="cross-encoder"):
        _validate_retrieval_reranker_settings(s)


def test_local_backend_accepts_hf_cross_encoder():
    s = copy.copy(settings)
    s.RETRIEVAL_RERANK_ENABLED = True
    s.RETRIEVAL_RERANK_BACKEND = "local_cross_encoder"
    s.RETRIEVAL_RERANK_MODEL = "dengcao/bge-reranker-v2-m3"
    _validate_retrieval_reranker_settings(s)  # no raise


def test_local_backend_defaults_to_bge(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_BACKEND", "local_cross_encoder")
    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_MODEL", None)
    r = LLMReranker()
    assert r.backend == "local_cross_encoder"
    assert r.model == "dengcao/bge-reranker-v2-m3"
    assert r.client is None  # no API client for local backend


def test_local_backend_validation_passes_without_model():
    s = copy.copy(settings)
    s.RETRIEVAL_RERANK_ENABLED = True
    s.RETRIEVAL_RERANK_BACKEND = "local_cross_encoder"
    s.RETRIEVAL_RERANK_MODEL = None
    _validate_retrieval_reranker_settings(s)  # no raise — reranker defaults the model


def test_local_backend_disabled_short_circuits():
    s = copy.copy(settings)
    s.RETRIEVAL_RERANK_ENABLED = False
    _validate_retrieval_reranker_settings(s)  # no raise


def test_local_backend_scores_numpy_array(monkeypatch):
    # CrossEncoder.predict returns a numpy array; the shared scoring helper must
    # not test it for truthiness ("truth value of an array is ambiguous").
    import asyncio

    import numpy as np

    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_BACKEND", "local_cross_encoder")
    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_TOP_N", 20)
    r = LLMReranker()

    class _Model:
        def predict(self, pairs):
            return np.array([-1.0, 2.0], dtype=np.float32)

    monkeypatch.setattr(r, "_get_local_cross_encoder", lambda: _Model())
    out, ok, reason = asyncio.run(
        r.maybe_rerank("q", [{"id": "a", "content": "x"}, {"id": "b", "content": "y"}], top_k=2, force=True)
    )
    assert ok and reason == "ok_local_cross_encoder"
    assert [c["id"] for c in out] == ["b", "a"]


def test_local_backend_stores_predict_probability_unchanged(monkeypatch):
    # CrossEncoder.predict already returns sigmoid(logit); rerank_score must be
    # that probability, not sigmoid of it (the pre-2026-10-04 0.5–0.731 squeeze).
    import asyncio

    import numpy as np

    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_BACKEND", "local_cross_encoder")
    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_TOP_N", 20)
    r = LLMReranker()

    class _Model:
        def predict(self, pairs):
            return np.array([0.000016, 0.853], dtype=np.float32)

    monkeypatch.setattr(r, "_get_local_cross_encoder", lambda: _Model())
    out, ok, _ = asyncio.run(
        r.maybe_rerank("q", [{"id": "a", "content": "x"}, {"id": "b", "content": "y"}], top_k=2, force=True)
    )
    assert ok
    assert out[0]["rerank_score"] == pytest.approx(0.853, abs=1e-4)
    assert out[1]["rerank_score"] == pytest.approx(0.000016, abs=1e-6)
