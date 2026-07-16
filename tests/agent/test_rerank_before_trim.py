"""Answer-context membership must be decided by the cross-encoder, not the weak
pre-rerank rrf heuristic. 2026-07-16 agent-path dig: gold ranks deep on the rrf
score but high on the reranker; the old pipeline trimmed by rrf THEN reranked the
survivors, so a wide recall pool let high-rrf distractors crowd gold out of the
packed context (recovered 3 misses but regressed 2). Fix: rerank the full deduped
pool, then trim/pack the top-K by rerank_score."""
import pytest

from src.agentrag.agent.context import ContextAssembler
from src.agentrag.agent import service as svc


def _mk_assembler():
    a = ContextAssembler()

    class _FakeRetriever:
        async def search(self, query, mode="hybrid", top_k=8, **kw):
            return {"results": []}  # no raw-query injection needed here

    async def _fake_rerank(query, items, top_k, force=False):
        # GOLD reranks high, the high-rrf distractor reranks low.
        for it in items:
            it["rerank_score"] = 0.80 if it.get("content_hash") == "GOLD" else 0.40
        return items, True, "ok_local_cross_encoder"

    a._retriever = _FakeRetriever()
    a._reranker.maybe_rerank = _fake_rerank  # type: ignore[assignment]
    return a


@pytest.mark.asyncio
async def test_trim_keeps_high_rerank_over_high_rrf_distractor(monkeypatch):
    monkeypatch.setattr(svc.settings, "RETRIEVAL_INCLUDE_RAW_QUERY", False, raising=False)
    # Budget fits ONE ~small chunk → the pipeline must choose. GOLD has a LOW rrf
    # score (would lose the old rrf-keyed trim) but the HIGHEST rerank score.
    monkeypatch.setattr(svc.settings, "AGENT_MAX_CONTEXT_TOKENS", 40, raising=False)
    tool_results = [{"results": [
        {"content_hash": "DISTRACT", "content": "w " * 30, "score": 99.0},  # high rrf, low rerank
        {"content_hash": "GOLD", "content": "w " * 30, "score": 1.0},        # low rrf, high rerank
    ]}]
    a = _mk_assembler()
    out = await a.assemble("q?", tool_results)
    packed = out["packed_context"]
    hashes = [c.get("content_hash") for c in packed]
    assert hashes and hashes[0] == "GOLD", f"top packed chunk should be GOLD (highest rerank), got {hashes}"
