"""Slice C Phase 4 — measure answer-time vision lift (ON vs OFF) on the image eval set.

Settings is an import-time singleton, so each arm runs as its own process with a
different VISION_ANSWER_MODEL env. Mode --arm runs one arm; mode --compare diffs two
per-arm json files into a markdown report.

  # OFF arm (answer-time vision disabled)
  VISION_ANSWER_MODEL="" PYTHONPATH=. uv run python scripts/eval/run_vision_e2e_ab.py --arm off
  # ON arm (answer-time vision = gemini)
  VISION_ANSWER_MODEL="gemini-2.5-flash" PYTHONPATH=. uv run python scripts/eval/run_vision_e2e_ab.py --arm on
  # compare
  PYTHONPATH=. uv run python scripts/eval/run_vision_e2e_ab.py --compare data/eval/vision_e2e_on.json data/eval/vision_e2e_off.json
"""
import argparse, asyncio, json, sys
sys.path.insert(0, ".")


async def _run_arm(arm, eval_path, n, out_path):
    import logging
    logging.basicConfig(level=logging.INFO)  # surface _answer "images=N" so we can confirm multimodal fires
    from src.agentrag.config import settings
    from src.agentrag.eval.benchmark_datasets import load_local_jsonl
    from src.agentrag.agent.factory import get_agent_service
    from src.agentrag.eval.correctness_judge import score_correctness
    from src.agentrag.services.llm_gateway import LLMGateway
    # config.py env_ignore_empty=True makes VISION_ANSWER_MODEL="" a NO-OP override, so set the
    # (mutable, call-time-read) singleton directly to guarantee a real ON/OFF contrast.
    settings.VISION_ANSWER_MODEL = "gemini-2.5-flash" if arm == "on" else ""
    print(f"ARM={arm} VISION_ANSWER_MODEL={settings.VISION_ANSWER_MODEL!r}")
    examples = load_local_jsonl(eval_path, n)
    agent, gateway = get_agent_service(), LLMGateway()
    rows = []
    for ex in examples:
        out = await agent.chat(question=ex.question, document_title=None,
                               conversation_id=f"vis-eval-{ex.id}")
        if out.get("timed_out"):
            rows.append({"id": ex.id, "score": None, "timed_out": True})
            print(f"  {ex.id} TIMED_OUT")
            continue
        ans = out.get("answer", "") or ""
        e = await score_correctness(ex.question, ans, ex.reference_answer,
                                    "\n".join(ex.gold_contexts), gateway)
        rows.append({"id": ex.id, "question": ex.question, "answer": ans[:400],
                     "score": e.mean, "low_confidence": e.low_confidence})
        print(f"  {ex.id} score={e.mean:.2f}")
    scored = [r["score"] for r in rows if isinstance(r.get("score"), (int, float))]
    mean = sum(scored) / len(scored) if scored else 0.0
    report = {"vision_answer_model": settings.VISION_ANSWER_MODEL, "n": len(rows),
              "scored": len(scored), "mean_correctness": round(mean, 4), "rows": rows}
    json.dump(report, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"ARM vision_answer_model={settings.VISION_ANSWER_MODEL!r} MEAN={mean:.4f} scored={len(scored)}/{len(rows)} -> {out_path}")


def _compare(on_path, off_path, out_md):
    on, off = json.load(open(on_path)), json.load(open(off_path))
    delta = on["mean_correctness"] - off["mean_correctness"]
    lines = ["# Vision Slice C — answer-time e2e A/B (2026-07-19)", "",
             "Eval set: `data/eval/vision_evalset_2026-07-19.jsonl` (image-dependent Qs).", "",
             "| arm | VISION_ANSWER_MODEL | n | scored | mean correctness |",
             "|---|---|---|---|---|",
             f"| OFF | `{off['vision_answer_model']}` | {off['n']} | {off['scored']} | {off['mean_correctness']:.4f} |",
             f"| ON | `{on['vision_answer_model']}` | {on['n']} | {on['scored']} | {on['mean_correctness']:.4f} |",
             "", f"**Delta (ON - OFF) = {delta:+.4f}**", "",
             ("**Recommendation:** default-ON answer-time vision" if delta >= 0.03
              else "**Recommendation:** keep answer-time vision OFF (no material lift on this set)"), ""]
    open(out_md, "w").write("\n".join(lines))
    print(f"delta={delta:+.4f} -> {out_md}")


async def _amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["on", "off"])
    ap.add_argument("--eval", default="data/eval/vision_evalset_2026-07-19.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("ON_JSON", "OFF_JSON"))
    ap.add_argument("--out-md", default="docs/eval/vision_e2e_2026-07-19.md")
    args = ap.parse_args()
    if args.compare:
        _compare(args.compare[0], args.compare[1], args.out_md)
    else:
        await _run_arm(args.arm, args.eval, args.n, args.out or f"data/eval/vision_e2e_{args.arm}.json")


asyncio.run(_amain())
