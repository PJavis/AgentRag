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
   small tail of failures: 3 of 26 questions (`0.00`, `0.00`, `0.25`). ~17/26 score 1.0. The old
   eval could not surface these; this one does, and they are actionable. **Investigated (see
   addendum) — the misses are NOT retrieval failures: they are flaky false-abstentions at the
   relevance floor.**

## Addendum — root cause of the 3 misses (2026-06-26, systematic-debugging)

Rows 11/18/21 (pest-risk law, China factory accidents, Bob Marley). Evidence gathered at each
component boundary:

- **Retrieval is fine.** Hybrid search returns the gold source-chunk at **rank 0** for all three.
- **Generation is fine.** When not gated, the agent answers correctly (e.g. row 21 → "Bob Marley
  died at 36 of a malignant tumor [1]…", matches gold).
- **The gate is the culprit — and it's flaky.** `_is_thin_context` (service.py:67) abstains when
  `max(rerank_score) < RETRIEVAL_RELEVANCE_FLOOR (0.6)`. The bge cross-encoder
  (`dengcao/bge-reranker-v2-m3`) scores the one relevant chunk at **~0.61–0.73 and every other
  chunk a flat 0.5** (it only meaningfully scores the top hit). The agent fires the question +3 LLM
  query-rewrites, so the max rerank score wobbles **right around 0.6** run-to-run: one run scored
  the top chunk **0.619 → answered correctly**; other runs (incl. the probe) dipped **< 0.6 →
  abstained** ("Tài liệu hiện có không có thông tin để trả lời câu hỏi này") → judge scores 0.

**Root cause:** borderline false-abstention at the `RETRIEVAL_RELEVANCE_FLOOR=0.6` boundary. The
floor (T7-calibrated at "in-corpus ~0.73" on a *different* set) sits inside the score mass of
genuinely-relevant-but-paraphrased Vietnamese chunks, so normal rewrite/rerank variance flips
answer↔abstain. Not retrieval, not generation.

**Side finding (separate bug):** `agent.chat` can **hang indefinitely** under a stalled gemini
connection (no per-call timeout) — observed a 42-minute hang on one question. Worth a request
timeout on the gateway client.

**Fix options (safety-sensitive — lowering the floor re-admits OOC hallucination, the exact thing
the gate was added to prevent):**
1. Lower `RETRIEVAL_RELEVANCE_FLOOR` modestly (e.g. 0.6 → 0.55) for borderline margin — re-validate
   OOC abstention doesn't regress (T7 set: OOC ~0.50, so 0.55 keeps separation).
2. Make the gate less knife-edge: require the best chunk to clear the floor by a margin, or average
   the top-k real scores, instead of a single-chunk `max < floor` flip.
3. Make query-rewrite deterministic (temperature 0) so the same question doesn't flip between runs.
4. Recalibrate the floor against the *prod* corpus relevance distribution (the synthetic prod-Q mass
   centers lower than T7's validation set).

### Fix applied + revalidation (commit `f5dfe76`)

Applied: floor `0.6→0.55`, deterministic `QueryRewriter` (temp 0), per-request gateway timeout (60s).

Revalidation (6 live Qs, floor 0.55):
- **row18** (was abstaining) → **answers correctly** ✅. Floor fix worked here.
- **row21** → **still abstains** ❌. But raw-question retrieve+rerank gives **max 0.716, stable across
  3 runs, >> 0.55** — so the floor is NOT row21's problem. The agent's *decide-step* generates
  rewritten/sub-queries (tool_trace showed 4 variants) whose assembled packed-context top score
  drops below the floor (0.619 one run, <0.55 another). The `QueryRewriter` temp-0 fix did NOT touch
  the decide-step tool-query generation — **residual flakiness lives there.**
- **OOC safety holds:** "capital of France" and "first US president" still abstain ✅ at 0.55
  (genuinely out-of-corpus). ("sulfuric acid" answered — but it is in-corpus: top hit is a lab
  sample-processing chunk, so not an OOC regression.)
- **Hang partially mitigated:** the 60s per-call timeout doesn't bound *total* agent.chat (4 rewrites
  × retries) — row11 still exceeded 120s. A total/step budget is the fuller fix.

**Residual root cause (next fix):** the agent decide-step tool-query generation is non-deterministic
and can retrieve worse chunks than the raw question → packed-context max rerank dips below floor →
false abstain. Candidate fixes: (a) make the decide-step deterministic (temp 0) like QueryRewriter;
(b) **always keep the raw-question retrieval in the rerank candidate pool** so rewrites can only ADD,
never degrade, the best chunk (raw row21 = 0.716 would always be present → never thin). (b) is more
robust and doesn't depend on determinism, and doesn't loosen OOC (raw OOC query still ~0.50).

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
plateaued.** Adopt it. The first concrete, eval-surfaced system bug is **flaky false-abstention at
the relevance floor** (see addendum) — NOT retrieval. Fixing it is a safety-gate tuning decision.

**Next:**
1. ✅ Tightened gold-answer prompt (commit `872a863`, drops the "based on the context" preamble) →
   regenerate eval set to take effect.
2. **Fix the floor flakiness** (addendum fix options) — re-validate OOC abstention doesn't regress.
3. Add the multi-chunk subset (v1.1) so the eval discriminates across the whole set, not just the tail.
4. Human calibration spot-check (25–30 Q, κ ≥ 0.7) to trust absolute numbers.
5. Re-run at higher n with retry/backoff on 503 (or off-peak) to cut the skip rate.
6. Add a per-call request timeout on the gateway client (the 42-min `agent.chat` hang).

## Files
`scripts/eval/build_prod_evalset.py`, `scripts/eval/oracle_probe.py`,
`src/agentrag/eval/correctness_judge.py`. Raw eval set + per-Q log are gitignored.
