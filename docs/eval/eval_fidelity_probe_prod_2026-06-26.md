# Eval-fidelity probe — prod corpus (2026-06-26)

Phase-1 oracle probe + new **ensemble correctness judge** (nugget-recall + reference-guided
rubric), run against a **prod-corpus eval set** synthesised over the live index (not the public
`vn_bkai`/`vn_legal` sets). This is the "fix the eval" diagnostic: does a trustworthy ruler still
plateau at ~0.74, or does it resolve real correctness?

- eval set: `data/eval/prod_corpus_evalset.jsonl` (gitignored) — synthetic Q + gold source-chunk +
  LLM-grounded gold answer over `Segment.content`, no re-ingest.
- judge: ensemble, `eval_judge=gemini-2.5-flash`, `eval_judge2=gemini-2.5-pro` (real cross-model floor).
- oracle: `gold_gen`/`oracle_gen=gemini-2.5-pro` (strong model + gold context = perfect retrieval).
- n=30 requested → **4 skipped on transient gemini 503** → **26 scored**.

## Result

| metric | value |
|---|---|
| system avg (live agent, ensemble) | **0.842** |
| oracle avg (pro + gold context) | **0.976** |
| **oracle − system** | **+0.134** |
| judge-noise floor — pearson(flash judge, pro judge) | **0.965** |

System-score distribution (26 scored): `17× 1.00`, `2× 0.85`, `0.83`, `0.78`, `0.70`, `0.64`,
`0.25`, `2× 0.00`. Oracle is `≈1.00` on all but a few (`0.93`, `0.85`, `0.60`).

## Read — the new ruler breaks the 0.74 ceiling

The earlier conclusion (correctness ~0.74 is eval-fidelity-bound) was measured with **RAGAS
`answer_correctness` (claim-F1) on terse public gold**. On this prod-corpus eval with the **new
ensemble judge + realistic gold answers**, that cap is gone:

1. **Ruler is not capped.** The oracle (perfect retrieval + strong generator) reaches **0.976** —
   the ensemble judge credits a correct, complete answer ~perfectly. The 0.74 plateau was the *old
   metric*, not a universal eval-fidelity wall.
2. **Ruler is trustworthy.** The flash and pro judge models agree at **pearson 0.965** — the
   correctness number is judge-stable, not noise. A low pearson would invalidate everything; 0.965
   validates it.
3. **Ruler now resolves real system signal.** The **+0.134** gap is not a uniform ceiling — it is a
   **retrieval-failure tail**: 3 of 26 questions (`0.00`, `0.00`, `0.25`) where the live system
   retrieved the wrong chunk / answered wrong while the oracle, *given the gold chunk*, nailed it.
   ~17/26 score 1.0. The old eval could not surface these misses; this one does, and they are
   actionable (retrieval, not the metric).

**So: the eval is measurably better.** It credits correctness up to ~0.98 (vs the old 0.74 cap),
agrees across judge models, and exposes the system's true headroom as identifiable retrieval
failures rather than an inscrutable plateau.

## Caveats (honest scope)

- **n=26 effective** — small; 4 questions dropped on transient gemini 503 (high-demand). Directional,
  not a precision number.
- **Corpus is mixed Vietnamese** (motorbikes, securities/financial law) — **not the medical corpus**
  memory/docs assumed. The eval is over whatever is actually indexed (176 segments, 134 usable).
- **v1 single-chunk synth-Q is easy** for the majority (17× 1.0). The discriminating signal lives in
  the retrieval-failure tail. **Multi-chunk / multi-hop gold (spec v1.1)** would harden the eval and
  spread the bulk of scores below 1.0.
- **Oracle ≈ 0.976, not 1.0** — the LLM-written gold answers carry a "Dựa vào ngữ cảnh được cung
  cấp…" preamble; the oracle phrases differently and the judge occasionally docks it. A gold-prompt
  cleanup (drop the preamble) would tighten the ceiling toward 1.0.
- **Synthetic gold is a proxy.** The spec's **human calibration step (§5, κ ≥ 0.7)** is still required
  to trust *absolute* numbers. The *relative* signal here (oracle > system, judge agreement 0.965) is
  already strong.

## Gate decision

**The eval-fidelity goal is met: the new ensemble ruler resolves correctness where the old one
plateaued.** Adopt it. The system's measured headroom is **retrieval** (3 hard misses), not the
metric — so resuming retrieval tuning is now worthwhile *and measurable*.

**Next:**
1. Tighten gold-answer prompt (drop the "based on the context" preamble) → oracle → ~1.0.
2. Add the multi-chunk subset (v1.1) so the eval discriminates across the whole set, not just the tail.
3. Human calibration spot-check (25–30 Q, κ ≥ 0.7) to trust absolute numbers.
4. Re-run at higher n with retry/backoff on 503 (or off-peak) to cut the skip rate.
5. Investigate the 3 retrieval misses (Q with sys=0.00/0.25) — the first concrete, eval-surfaced
   system bug.

## Files
`scripts/eval/build_prod_evalset.py`, `scripts/eval/oracle_probe.py`,
`src/agentrag/eval/correctness_judge.py`. Raw eval set + per-Q log are gitignored.
