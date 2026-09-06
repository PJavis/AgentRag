# Table Probe — Unblock Plan (2026-09-06)

**Status:** Road 0 EXECUTED 2026-09-06 — results in §7. Roads 1 and 2 remain open.
**Blocks on:** `docs/eval/table_probe_power_analysis_2026-09-06.md` §5 — the probe as
designed is a confirmatory GO/NO-GO, and this corpus cannot support one.
**Scope note:** `PDF_PRESERVE_TABLES` stays default-`False` through every road below.
No road ships a table feature; each one only buys information.

## 0. The block, in one paragraph

Tasks 1–3 and 6 are done and merged. Tasks 4, 5, 7, 8, 9 (preflight, gated parser arm,
evalset authoring, chunk-integrity precondition, paired A/B runner) are unstarted and
would cost the expensive part of the probe — hand-authoring 54–81 questions with
reference answers, two full ingests, and a judge pass. The power analysis says the
answer that spend buys cannot be confirmatory: the usable pool is 27 tables, the
shipped rule cannot return GO below 6 discordant tables, 80% power needs arm B to help
42% of tables, and 14 of the 27 tables sit in a single document — so to the extent
outcomes cluster per document the effective n approaches 7, where **GO is unreachable
at any effect size**.

## 1. The asymmetry that decides the sequencing

Clustering is not a statistics problem. No test, no alpha, no extra question fixes 14
tables sharing one template. What follows from that:

| claim the probe might make | credible on this corpus? |
|---|---|
| "arm B does not change what tables look like downstream" | **yes** — mechanical, no sampling involved |
| "arm B does not move retrieval of the gold row" | **mostly** — continuous outcome, but the n_eff≈7 caveat still bounds it |
| "arm B does not improve answers" (NO-GO) | weakly — means only *"no effect large enough for 27 clustered tables to see"* |
| "arm B improves answers" (GO) | **no** — cannot be established here |

So: **spend on cheap falsification, not on expensive confirmation.** A negative result
is reachable for hours of compute; a positive one is not reachable at any spend without
a bigger corpus. Every road below is ordered by that.

## 2. Road 0 — mechanical pre-probes (recommended, do first)

Judge-free, author-free, no reference answers, no evalset. Both run on already-merged
code (`table_quality.render_markdown`, `table_probe_corpus_survey.py`).

### 0a. Structure-delta check (hours)

For each of the 27 usable tables, compare what arm A actually indexes (PyMuPDF
`get_text("text", sort=True)` over the table's bbox) against arm B's
`render_markdown` output, and measure **row/column adjacency preserved** — for every
(row label, column header, cell) triple in `extract()`, is the cell still adjacent to
both of its keys in the emitted text?

- Output: per-table adjacency-restored fraction, arm A vs arm B.
- **Kill condition:** if arm B does not materially restore adjacency on most of the 27,
  the probe is over — NO-GO for free, no questions authored, no indices built. The
  causal chain the whole spec rests on is broken at step one.
- **Pass condition:** arm B restores adjacency broadly → the mechanism exists, continue
  to 0b. This is *not* evidence that answers improve.

### 0b. Retrieval-only probe (≈1 day)

Two ingests (flag off / flag on) into isolated indices, then measure **rank of the gold
row's chunk** for a mechanically generated query per table — `"<row label> <column
header>"` built from `extract()`, no hand authoring.

- Outcome per table is **continuous** (rank / recall@k), not the binary correctness the
  sign test consumes, so it carries magnitude and needs a far smaller n than 27 to see
  the same effect. Analyse with Wilcoxon signed-rank, and report a **document-clustered
  bootstrap CI** alongside.
- **Kill condition:** arm B does not improve retrieval of the gold row → e2e answers
  cannot improve either (the answer model never sees the row). NO-GO, cheaply.
- **Limitation to state in the report, not hide:** synthetic queries are not user
  questions. This measures the mechanism, not the user-visible outcome. It can kill the
  probe; it cannot green-light a build on its own.

Cost of Road 0 in total: two ingests and no judge spend. This is the whole reason to run
it before Task 7.

## 3. Road 1 — estimation run (only if Road 0 passes)

Execute Tasks 4, 5, 7, 8, 9 as planned, with **one contract change to Task 9's report**:

- Report per-table wins / losses / ties with a **document-clustered bootstrap interval**.
- State the **MDE (0.42 at N=27, independent) and the n_eff≈7 clustering bound** on the
  same page as the result.
- `decide_paired` still runs and its verdict is still printed — labelled
  **non-confirmatory**. A NO-GO from this corpus means *"no effect large enough for 27
  clustered tables to see"*, and the report must say those words.
- Everything else in the plan's Global Constraints stands unchanged: 29 unique docs,
  gate via `is_safe_to_markdown`, render via `render_markdown` (never `to_markdown()`),
  table append after the OCR block, timeouts excluded not zeroed, `corpus_docs_sha`
  guard, full accounting of every question that does not reach the comparison.

Question authoring stays at 2–3 per table — for **outcome reliability per table**, not
for power. n is the table count either way; the plan already says so and the power
analysis §4 makes it explicit.

## 4. Road 2 — grow the corpus (the only real fix)

Required if a confirmatory GO is genuinely wanted. Needs **documents, not questions**,
and documents that are not templated copies of each other — the current 27 tables come
from 7 docs with 52% in one.

Rough target: 80% power at a plausible `p_help` of 0.25 needs roughly N≈60 tables, and
those tables must come from enough distinct source documents that document-level
clustering stops dominating — call it ≥30 distinct docs. That is a corpus-acquisition
task with its own scope, cost and PHI review, not something the probe can do to itself.

## 5. Rejected levers (recorded so they are not re-proposed)

| lever | why rejected |
|---|---|
| author more questions per table | power is driven by the table count; adds cost, adds nothing |
| include the 13 thin tables (N=40) | they cannot carry a row/column-alignment question — would manufacture ties |
| loosen alpha or the 2:1 win rule | discards the false-positive control the rev-2 design exists to provide |
| treat `is_data_grid` as the eligible set | it **ranks** targets; it has never gated. Arm B rewrites all 40 gate-passing tables |
| gate the run with `eval/corpus_fingerprint.py` | it hashes `(document_title, segment_count)`, and arm B changes segment counts by design → would flag a correct run. Use `corpus_docs_sha` |

## 6. Decision required from a human

1. **Run Road 0a + 0b?** (recommended — cheap, and the only way a NO-GO gets bought at a
   sane price)
2. **If Road 0 passes, run Road 1 as an estimation** — accepting that the output is an
   estimate with a stated MDE, not a gate?
3. **Or go straight to Road 2** (corpus growth) because only a confirmatory answer is
   acceptable?
4. **Or stop the probe here** and record "unpowered on the available corpus" as the
   final state — the question stays open but is no longer re-litigated on a hunch.

Option 4 is a legitimate outcome. The probe's purpose was to stop the table question
being re-argued from intuition; a written "we measured what we could measure and the
corpus cannot answer it" achieves that, and costs nothing more.


---

## 7. Results — Road 0 executed (2026-09-06)

Road 0 ran, plus a step it did not anticipate. Reports:
`docs/eval/table_probe_structure_delta_2026-09-06.md`,
`table_probe_retrieval_ab_2026-09-06.md`,
`table_probe_comprehension_ab_2026-09-06.md`.

### 0a — structure delta: the premise holds

Same 27-table / 7-document pool the power analysis reasoned about.

| quantity | value |
|---|---|
| median arm-A row adjacency | **17%** |
| mean (mean of per-document means) | 29% (30%) |
| tables fully intact under arm A | **0 of 27** |
| tables fully destroyed | 12 of 27 |

79 of 101 failed rows are `cell_fragmented_across_columns` — the cell's own words
cut apart and interleaved with the neighbouring column. 16 are wrapped cells, 6 are
reading-order scrambling proper, 0 are missing text. **Continue.**

### 0b — retrieval: not where tables hurt

19 askable rows, mechanical queries, both arms through the production parse+chunk
path, ranked corpus-wide.

| retriever | MRR A | MRR B | mean ΔRR | 95% CI (doc-clustered) |
|---|---|---|---|---|
| bm25 | 0.869 | 0.921 | +0.052 | [0.000, +0.211] |
| dense | 0.401 | 0.354 | −0.046 | [−0.110, +0.006] |
| rrf | 0.730 | 0.715 | −0.015 | [−0.048, +0.042] |

No retriever shows arm B raising the gold row — **and the kill condition still does
not fire**, because its premise is false: arm A already retrieves the gold row at
recall@10 = 0.95. The row reaches the model under both arms. 0a's adjacency collapse
does not propagate here because lexical retrieval is bag-of-words: a shredded row
still matches all its own terms. So retrieval is removed as the mechanism, and arm B
has no retrieval benefit to offer.

### 0c — comprehension: this is where tables hurt (a step Road 0 did not plan)

0b left exactly one hypothesis: once the row is in context, can the model bind a cell
to its column? Ground truth is `extract()`, so there is no judge and no authored
reference answer — a few LLM calls per table.

| quantity | arm A (flat) | arm B (+markdown) | arm C (flat ×2) |
|---|---|---|---|
| cells read correctly | 0.72 | **0.84** | 0.68 |
| mean token-F1 | 0.695 | **0.843** | 0.619 |
| abstained | 5 | 2 | 6 |
| surplus tokens beyond the gold cell | 52 | 3 | — |
| — borrowed from OTHER cells of the table | **47 (90%)** | 0 (0%) | — |

Paired B vs A: **3 better / 0 worse / 16 same**, mean ΔF1 +0.148, doc-clustered 95% CI
[+0.000, +0.269], sign test p = 0.250 → the shipped rule says INCONCLUSIVE, exactly as
the power analysis predicted it would (3W/0L cannot clear a bar needing 6 discordant).

**The mechanism is the last row.** Under arm A the model usually finds the value and
cannot see where it *stops* — 90% of its surplus tokens are borrowed from other cells
of the same table. A containment-style correctness check scores "right cell with the
next row glued on" as correct; as an answer to a real question it is not.

**Duplication is ruled out.** Arm B's context is a superset of arm A's, so the win
could have been the second copy. Arm C is the page text twice with no structure added:
it scores 0.68 — *no better than arm A* — while B beats C 3/0/16 at ΔF1 +0.224. The
gain is the structure.

### What the chain now says

1. Flattening genuinely destroys table structure (0a).
2. That damage does **not** cost retrieval (0b) — so any table feature justified as a
   recall fix is justified on a false premise.
3. It **does** cost comprehension, by a measured mechanism — cell-boundary bleed (0c),
   and the fix is structure, not repetition.
4. The effect is directionally consistent and never once negative across 19 paired
   comparisons in 7 documents, and still **cannot clear the shipped confirmatory bar**.
   That is the corpus limit the power analysis described, arriving exactly as predicted.

### Recommended next step

Road 0 has bought more than Road 1 would have: a mechanism, a control, and zero losses,
for a few hundred LLM calls. Two honest options remain, and the choice is a human's:

- **Ship arm B behind `PDF_PRESERVE_TABLES` and measure on production traffic.** The
  change is additive and gated, it never lost here, and production is the only place
  the corpus is big enough to settle it. This treats 0c as sufficient evidence to try,
  not as proof.
- **Run Road 1 as estimation** (Tasks 4/7/8/9 — Task 5 is now done) with real authored
  questions. It tests user-shaped questions rather than cell lookup, but it remains
  underpowered by exactly the margin the power analysis computed and will most likely
  return another INCONCLUSIVE.

What is no longer worth doing: arguing the table question from intuition. The mechanism
is measured, the retrieval claim is falsified, and the confound is controlled.
