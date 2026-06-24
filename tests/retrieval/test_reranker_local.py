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
