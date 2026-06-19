"""Companion to probe_thin_context: same spy, but on IN-CORPUS (relevant) suite
questions. Compares relevant max rerank_score vs the ~0.50 floor seen on
out-of-corpus, to decide whether a recalibrated RETRIEVAL_RELEVANCE_FLOOR can
separate them.
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
    v = _orig(packed_context, floor)
    records.append({"n_scored": len(scores),
                    "max": round(max((float(s) for s in scores), default=-1), 4)})
    return v


svc._is_thin_context = _spy


async def main():
    print(f"IN-CORPUS · abstain={settings.ANSWER_ABSTAIN_ON_THIN_CONTEXT} floor={settings.RETRIEVAL_RELEVANCE_FLOOR}")
    # full in-corpus questions from the already-run benchmark report (no HF load)
    pc = json.loads(Path("data/eval/abstain_ab_A_off.json").read_text(encoding="utf-8"))["per_case"]
    qs = [c["question"] for c in pc][:10]
    agent = get_agent_service()
    maxes = []
    for i, q in enumerate(qs):
        records.clear()
        await agent.chat(question=q, document_title=None, conversation_id=f"probein-{i}")
        ex = type("E", (), {"question": q})()
        rec = records[0] if records else {"max": None, "n_scored": 0}
        if isinstance(rec["max"], float) and rec["max"] >= 0:
            maxes.append(rec["max"])
        print(f"max={rec['max']!s:>7} scored={rec['n_scored']:>2} | {ex.question[:50]}")
    if maxes:
        maxes.sort()
        print(f"\nIN-CORPUS max rerank_score: min={maxes[0]} median={maxes[len(maxes)//2]} max={maxes[-1]} (n={len(maxes)})")
        print("OUT-OF-CORPUS was ~0.50 flat. Separable floor exists iff in-corpus min > ~0.51.")


asyncio.run(main())
