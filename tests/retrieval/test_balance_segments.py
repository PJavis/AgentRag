"""S10 — image-segment cap in retrieval top_k."""
from __future__ import annotations

import pytest

from src.agentrag.config import settings
from src.agentrag.retrieval.elasticsearch_retriever import ElasticsearchRetriever


def _h(seg: str, name: str) -> dict:
    return {"segment_type": seg, "content_hash": name, "rank": 0}


def test_demotes_excess_images_when_text_available(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_MAX_IMAGE_RATIO", 0.3)
    hits = [_h("image", f"i{i}") for i in range(1, 6)] + [
        _h("text", f"t{i}") for i in range(1, 5)
    ]
    out = ElasticsearchRetriever._balance_segment_types(hits, top_k=5)
    head = out[:5]
    types = [h["segment_type"] for h in head]
    images = types.count("image")
    assert images <= 1, f"max 1 image with ratio 0.3 × top_k 5; got {types}"
    text_hashes = [h["content_hash"] for h in head if h["segment_type"] == "text"]
    assert text_hashes[0] == "t1"


def test_keeps_images_when_no_text_in_tail():
    hits = [_h("image", f"i{i}") for i in range(1, 8)]
    out = ElasticsearchRetriever._balance_segment_types(hits, top_k=5)
    assert all(h["segment_type"] == "image" for h in out[:5])


def test_noop_below_cap():
    hits = (
        [_h("text", "t1")]
        + [_h("image", "i1")]
        + [_h("text", f"t{i}") for i in range(2, 5)]
    )
    out = ElasticsearchRetriever._balance_segment_types(hits, top_k=5)
    assert [h["content_hash"] for h in out] == [
        h["content_hash"] for h in hits
    ]


def test_empty_input():
    assert ElasticsearchRetriever._balance_segment_types([], 5) == []
