"""ContextAssembler must inject a raw-question retrieval into the candidate pool so the
agent's decide-step rewrites (which retrieve via hybrid_kg + variants) can only ADD chunks,
never drop the best raw-question chunk below the abstain floor (2026-06-26 row21 flaky-abstain:
raw hybrid scores the relevant chunk 0.716, but the agent's pool topped out at 0.619/<floor)."""
import pytest

from src.agentrag.agent.context import ContextAssembler
from src.agentrag.agent import service as svc


def _mk_assembler(monkeypatch, raw_hit):
    a = ContextAssembler()

    class _FakeRetriever:
        async def search(self, query, mode="hybrid", top_k=8):
            return {"results": [dict(raw_hit)]}

    async def _fake_rerank(query, items, top_k, force=False):
        # score the raw-question's gold chunk high, everything else neutral
        for it in items:
            it["rerank_score"] = 0.716 if it.get("content_hash") == "GOLD" else 0.50
        return items, True, "ok_local_cross_encoder"

    a._retriever = _FakeRetriever()
    a._reranker.maybe_rerank = _fake_rerank  # type: ignore[assignment]
    return a


@pytest.mark.asyncio
async def test_raw_query_chunk_injected_when_tools_missed_it(monkeypatch):
    monkeypatch.setattr(svc.settings, "RETRIEVAL_INCLUDE_RAW_QUERY", True, raising=False)
    # the agent's tool_results MISSED the gold chunk (only distractors)
    tool_results = [{"results": [
        {"content_hash": "D1", "content": "distractor one", "score": 30.0},
        {"content_hash": "D2", "content": "distractor two", "score": 25.0},
    ]}]
    gold = {"content_hash": "GOLD", "content": "Bob Marley died of melanoma at 36.", "score": 40.0}
    a = _mk_assembler(monkeypatch, raw_hit=gold)

    out = await a.assemble("Bob Marley đã qua đời vì bệnh gì?", tool_results)
    packed = out["packed_context"]
    hashes = {c.get("content_hash") for c in packed}
    assert "GOLD" in hashes, "raw-question gold chunk not injected into the pool"
    # and it scores above the abstain floor → not thin
    assert svc._is_thin_context(packed, svc.settings.RETRIEVAL_RELEVANCE_FLOOR) is False


@pytest.mark.asyncio
async def test_injection_disabled_by_flag(monkeypatch):
    monkeypatch.setattr(svc.settings, "RETRIEVAL_INCLUDE_RAW_QUERY", False, raising=False)
    tool_results = [{"results": [{"content_hash": "D1", "content": "distractor", "score": 30.0}]}]
    gold = {"content_hash": "GOLD", "content": "the gold chunk", "score": 40.0}
    a = _mk_assembler(monkeypatch, raw_hit=gold)
    out = await a.assemble("q?", tool_results)
    assert "GOLD" not in {c.get("content_hash") for c in out["packed_context"]}
