# Table Probe — Unblock Plan (2026-09-06)

**Status:** decision plan. Nothing here is executed yet.
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
