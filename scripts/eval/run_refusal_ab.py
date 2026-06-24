"""Answerability-gate A/B on the out-of-corpus refusal set (P1 Task 5.9).

Reuses the already-ingested corpus (no re-ingest). Runs each refusal question
through the live agent twice — gate OFF then gate ON — toggling
`settings.ANSWERABILITY_GATE_ENABLED` at runtime (the answer prompt reads it per
call). classify_refusal is rule-based, so this costs only the agent answer calls
(no LLM judge). Captures the max rerank_score per case to show why the gate fires.

Run: uv run python scripts/eval/run_refusal_ab.py
Writes: docs/eval/benchmark_answerability_ab_2026-06-24_vi.md
"""
import asyncio
import json
import os
from pathlib import Path

from src.agentrag.agent import service as svc
from src.agentrag.config import settings
from src.agentrag.agent.factory import get_agent_service
from src.agentrag.eval.refusal import classify_refusal

_max_score = {"v": None}
_orig_thin = svc._is_thin_context


def _spy(packed_context, floor):
    scores = [c.get("rerank_score") for c in (packed_context or [])
              if c.get("rerank_score") is not None]
    _max_score["v"] = round(max((float(s) for s in scores), default=-1.0), 4)
    return _orig_thin(packed_context, floor)


svc._is_thin_context = _spy


def _answer_of(out: dict) -> str:
    return (out.get("answer") or out.get("response") or "").strip()


async def _run_arm(cases: list[dict], gate: bool) -> dict:
    settings.ANSWERABILITY_GATE_ENABLED = gate
    agent = get_agent_service()
    counts = {"abstained": 0, "hedged_cited": 0, "hallucinated": 0, "empty": 0}
    per_case = []
    for c in cases:
        q = c.get("question", "")
        _max_score["v"] = None
        try:
            out = await agent.chat(question=q, document_title=None,
                                   conversation_id=f"refab-{int(gate)}-{c.get('id')}")
            ans, cits = _answer_of(out), (out.get("citations") or [])
        except Exception as exc:  # an error is not a refusal
            ans, cits = "", []
            print(f"  [err] {c.get('id')}: {exc}")
        verdict = classify_refusal(ans, cits)
        counts[verdict] += 1
        per_case.append({"id": c.get("id"), "verdict": verdict,
                         "max_score": _max_score["v"], "n_cits": len(cits)})
        print(f"  gate={int(gate)} [{verdict:12}] max={_max_score['v']!s:>7} cits={len(cits)} | {q[:48]}")
    n = len(cases) or 1
    return {
        "n": len(cases),
        "counts": counts,
        "refusal_rate": round(counts["abstained"] / n, 3),
        "hedged_cited_rate": round(counts["hedged_cited"] / n, 3),
        "hallucination_rate": round(counts["hallucinated"] / n, 3),
        "per_case": per_case,
    }


def _md(off: dict, on: dict) -> str:
    L = []
    L.append("# Answerability-gate A/B — out-of-corpus refusal set (2026-06-24)\n")
    L.append(f"- Set: `data/eval/refusal_set.json` · n={off['n']} · corpus reused (no re-ingest)")
    L.append(f"- Gate: `ANSWERABILITY_GATE_ENABLED` · margin `ANSWERABILITY_GRAY_MARGIN={settings.ANSWERABILITY_GRAY_MARGIN}`")
    L.append(f"- Floor `RETRIEVAL_RELEVANCE_FLOOR={settings.RETRIEVAL_RELEVANCE_FLOOR}` · rerank `{settings.RETRIEVAL_RERANK_BACKEND}`\n")
    L.append("| Metric | OFF (baseline) | ON (gate) |")
    L.append("|---|---|---|")
    L.append(f"| refusal_rate (clean abstain, ideal ↑) | {off['refusal_rate']:.3f} | {on['refusal_rate']:.3f} |")
    L.append(f"| hedged_cited_rate (soft ↓) | {off['hedged_cited_rate']:.3f} | {on['hedged_cited_rate']:.3f} |")
    L.append(f"| **hallucination_rate (DANGEROUS ↓)** | **{off['hallucination_rate']:.3f}** | **{on['hallucination_rate']:.3f}** |")
    L.append(f"| counts (abstain/hedged/halluc/empty) | {off['counts']['abstained']}/{off['counts']['hedged_cited']}/{off['counts']['hallucinated']}/{off['counts']['empty']} | {on['counts']['abstained']}/{on['counts']['hedged_cited']}/{on['counts']['hallucinated']}/{on['counts']['empty']} |\n")
    L.append("## Per-case (id · OFF→ON verdict · max rerank score)\n")
    on_by = {c["id"]: c for c in on["per_case"]}
    L.append("| id | OFF | ON | max_score(ON) |")
    L.append("|---|---|---|---|")
    for c in off["per_case"]:
        o = on_by.get(c["id"], {})
        L.append(f"| {c['id']} | {c['verdict']} | {o.get('verdict','?')} | {o.get('max_score')} |")
    return "\n".join(L) + "\n"


async def main():
    print(f"abstain={settings.ANSWER_ABSTAIN_ON_THIN_CONTEXT} floor={settings.RETRIEVAL_RELEVANCE_FLOOR} "
          f"margin={settings.ANSWERABILITY_GRAY_MARGIN} rerank={settings.RETRIEVAL_RERANK_BACKEND}")
    cases = json.loads(Path("data/eval/refusal_set.json").read_text(encoding="utf-8"))
    if os.environ.get("REFUSAL_SINGLE_ARM"):
        gate = settings.ANSWERABILITY_GATE_ENABLED
        label = f"gate{int(gate)}_floorgate{int(settings.RETRIEVAL_RELEVANCE_GATE_ENABLED)}"
        print(f"\n=== SINGLE ARM ({label}) n={len(cases)} ===")
        res = await _run_arm(cases, gate=gate)
        out = Path(f"docs/eval/refusal_singlearm_{label}.json")
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{label}: refusal={res['refusal_rate']} hedged={res['hedged_cited_rate']} "
              f"halluc={res['hallucination_rate']} counts={res['counts']}")
        print(f"wrote {out}")
        return
    print(f"\n=== ARM A: gate OFF (n={len(cases)}) ===")
    off = await _run_arm(cases, gate=False)
    print(f"\n=== ARM B: gate ON (n={len(cases)}) ===")
    on = await _run_arm(cases, gate=True)
    out_path = Path("docs/eval/benchmark_answerability_ab_2026-06-24_vi.md")
    out_path.write_text(_md(off, on), encoding="utf-8")
    print(f"\nOFF hallucination_rate={off['hallucination_rate']:.3f}  ->  ON={on['hallucination_rate']:.3f}")
    print(f"OFF refusal_rate={off['refusal_rate']:.3f}  ->  ON={on['refusal_rate']:.3f}")
    print(f"wrote {out_path}")


asyncio.run(main())
