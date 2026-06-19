"""Diagnostic: for each out-of-corpus refusal question, run the live agent and
record what _is_thin_context actually sees — max rerank_score, the thin verdict,
and whether citations survived. Decides why refusal_rate stayed 0.0.

Run: ANSWER_ABSTAIN_ON_THIN_CONTEXT=true uv run python scripts/eval/probe_thin_context.py
"""
import asyncio
import json
from pathlib import Path

from src.agentrag.agent import service as svc
from src.agentrag.config import settings
from src.agentrag.agent.factory import get_agent_service

records = []
_orig = svc._is_thin_context


def _spy(packed_context, floor):
    scores = [c.get("rerank_score") for c in (packed_context or [])
              if c.get("rerank_score") is not None]
    verdict = _orig(packed_context, floor)
    records.append({
        "n_items": len(packed_context or []),
        "n_scored": len(scores),
        "max_score": round(max((float(s) for s in scores), default=-1), 4),
        "floor": floor,
        "thin": verdict,
    })
    return verdict


svc._is_thin_context = _spy


async def main():
    print(f"abstain={settings.ANSWER_ABSTAIN_ON_THIN_CONTEXT} floor={settings.RETRIEVAL_RELEVANCE_FLOOR} "
          f"rerank={settings.RETRIEVAL_RERANK_ENABLED}/{settings.RETRIEVAL_RERANK_BACKEND}")
    cases = json.loads(Path("data/eval/refusal_set.json").read_text(encoding="utf-8"))
    agent = get_agent_service()
    for c in cases:
        q = c.get("question", "")
        records.clear()
        out = await agent.chat(question=q, document_title=None, conversation_id=f"probe-{c.get('id')}")
        cits = out.get("citations") or []
        # the answer-node _is_thin_context call is the relevant one (first record)
        rec = records[0] if records else {"max_score": None, "thin": None, "n_scored": 0}
        print(f"[{rec['thin']!s:5}] max={rec['max_score']!s:>7} scored={rec['n_scored']:>2} "
              f"cits={len(cits):>2} | {q[:46]}")


asyncio.run(main())
