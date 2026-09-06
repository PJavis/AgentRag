# Table Probe — Grid Accounting and Power Analysis (2026-09-06)

**Status:** blocking. Settles whether the probe can answer its question before any
eval questions are authored.

**Answer up front: no, not as a confirmatory go/no-go.** The corpus supports at most
27 independent units, the shipped decision rule cannot return GO below 6 discordant
units, and 52% of the units live in one document. Recommendation in §5.

## 1. Two numbers that looked contradictory

An earlier summary of this work gave two accounts that appeared to move in opposite
directions. They describe **different stages** and both are correct:

| stage | quantity | before | after |
|---|---|---|---|
| **1. Classification** — which detections are `real_data` | data grids | 35 | **19** |
| **2. Chunk-window eligibility** — of those, which survive chunking | eligible grids | 12 of 19 | **19 of 19** |

Stage 1 fell because the classifier was corrected. Stage 2 rose because the filter was
reading the wrong field (`est_tokens`, the whole-table total, instead of
`max_block_tokens`, the largest emitted block). Nothing was double-counted; the
summary simply presented a stage-1 number and a stage-2 number as if they described
the same set.

## 2. Where the 16 grids went

Every one of the 35 is accounted for. Attribution is by **ablation**: revert one new
rule at a time and see which restores `real_data`.

| doc | page | tbl | new kind | attributed to | still gate-passing? |
|---|---|---|---|---|---|
| 2f499de4 | 9 | 0 | `nonnumeric` | ordinal-column exclusion | yes |
| 2f499de4 | 13 | 0 | `nonnumeric` | digit-density + ordinal | yes |
| 2f499de4 | 14 | 1 | `nonnumeric` | digit-density | yes |
| 2f499de4 | 15 | 0 | `nonnumeric` | digit-density + ordinal | yes |
| 2f499de4 | 16 | 2 | `nonnumeric` | digit-density | yes |
| 2f499de4 | 17 | 1 | `nonnumeric` | digit-density | yes |
| 2f499de4 | 18 | 1 | `nonnumeric` | digit-density | yes |
| 2f499de4 | 19 | 0 | `nonnumeric` | digit-density | yes |
| 5164c4af | 7 | 0 | `nonnumeric` | digit-density | yes |
| 81273a0e | 23 | 1 | `nonnumeric` | digit-density | yes |
| 81273a0e | 24 | 0 | `nonnumeric` | digit-density | yes |
| 881e7c3a | 136 | 0 | `single_column` | structured-rows rule | **no** |
| 881e7c3a | 140 | 1 | `single_column` | structured-rows rule | **no** |
| 881e7c3a | 163 | 4 | `single_column` | structured-rows rule | **no** |
| 881e7c3a | 163 | 6 | `single_column` | structured-rows rule | **no** |
| 881e7c3a | 189 | 0 | `single_column` | structured-rows rule | **no** |

**None was dropped by a defect.** Inspected cell-by-cell:

- The 5 `single_column` cases have **0 or 1 rows with two populated cells**. `p163 t4`,
  `p163 t6` and `p189` have literally zero. They are prose fragments inside a border;
  the old classifier passed them only because it counted PyMuPDF's `None` column
  placeholders as real columns. Correctly excluded, and correctly rejected by the gate.
- The 11 `nonnumeric` cases are **real, structured, multi-column tables** — a 4-column
  procedure table, a 3-column comparison matrix, a drug side-effect table. They lost
  `real_data` because their only digits were a row counter (`STT`) or a stray year.
  Two sit just under the 0.15 numeric floor at 0.12 — a threshold call, not a bug.

## 3. The consequential correction: `real_data` was never the eligible set

`is_data_grid` **ranks** targets; it has never gated anything. `table_quality.py` says
so directly: *"Numeric density must NOT gate arm B: a text-only comparison matrix is
still a real table whose columns carry meaning."* Arm B rewrites all **40**
gate-passing tables, `real_data` and `nonnumeric` alike.

So describing the target set as having "shrunk 35 → 19" was wrong in a way that
matters — it conflated the ranking class with the eligible set. Of the 40 gate-passing
tables, those with enough structure to support a row/column-alignment question
(≥3 structured rows **and** ≥3 columns):

| | count |
|---|---|
| gate-passing tables | 40 |
| — with ≥3 structured rows and ≥3 columns | **27** (16 `real_data`, 11 `nonnumeric`) |
| — too thin to ask an alignment question | 13 |

**The usable pool is 27 tables, not 19.**

## 4. Unit of analysis, and what that does to power

**The unit is the table.** Questions authored against one table are not independent
draws: arm B either restores that table's row/column binding or it does not, and every
question on it inherits that single fact. Feeding per-question outcomes into the sign
test would inflate n and deflate p — the same independence violation that motivated
the round-robin fix in `rank_targets`.

**Stated explicitly, because it inverts the spec's stated remedy:** adding more
questions per table **does not increase power**. §4 of the design says *"the remedy is
more questions, not a lower bar."* That is only true across *tables*. Within a table,
2–3 questions are worth authoring — they reduce the chance of misclassifying that
table's outcome — but the sign test's n stays at the table count either way.

The spec **states no minimum detectable effect and no power target.** It fixes a
decision rule (B wins ≥2× as often as it loses, exact two-sided sign test p < 0.05)
and never asks what that rule can see. Derived from the shipped `decide_paired`:

**The rule cannot return GO on fewer than 6 discordant tables.** 5W/0L gives p = 0.0625
→ INCONCLUSIVE; 6W/0L gives p = 0.0312 → GO.

Power = P(GO), each table independently helped with probability `p_help`, hurt with
probability 0.05, else tied:

| `p_help` | N=7 (docs) | N=19 | **N=27** | N=40 |
|---|---|---|---|---|
| 0.15 | 0.00 | 0.03 | 0.08 | 0.17 |
| 0.20 | 0.00 | 0.08 | 0.19 | 0.36 |
| 0.25 | 0.00 | 0.17 | 0.35 | 0.57 |
| 0.30 | 0.00 | 0.29 | 0.51 | 0.74 |
| 0.40 | 0.02 | 0.55 | 0.77 | 0.94 |
| 0.50 | 0.06 | 0.76 | 0.92 | 0.99 |

| pool | `p_help` needed for 80% power |
|---|---|
| N=19 (`real_data` only) | 0.53 |
| **N=27 (usable pool)** | **0.42** |
| N=40 (all gate-passing) | 0.33 |
| N=7 (if outcomes cluster by document) | **unreachable at any effect size** |

**The clustering caveat is decisive.** 14 of the 27 tables (52%) are in `2f499de4`,
spread over 7 documents total. Tables within a document share a template, so if arm B
helps one skills table there it very likely helps all of them. To the extent outcomes
cluster at document level, the effective n approaches **7** — and at n=7 the rule
cannot reach GO even if arm B helps every single table, because 6 discordant units out
of 7 requires nearly perfect separation.

## 5. Decision

**The probe is not adequately powered for a confirmatory GO/NO-GO on this corpus.**
It can detect only a large effect — arm B helping ~42% of tables — and only if
outcomes are independent across tables, which the 52% single-document concentration
makes doubtful.

What each available lever actually buys:

| lever | effect on power | verdict |
|---|---|---|
| Use the 27-table pool instead of 19 `real_data` | MDE 0.53 → 0.42 | **do it** — free, and a correction of a misreading, not a loosened standard |
| Author more questions per table | **none** — n is the table count | do 2–3 for outcome reliability only |
| Include the 13 thin tables (N=40) | MDE → 0.33, but they cannot carry an alignment question | reject — would manufacture ties |
| Loosen the decision rule | would trade the false-positive control the rev-2 design exists to provide | reject |
| Add documents to the corpus | the only lever that genuinely fixes both n and clustering | the real fix, out of scope here |

**Recommendation: run it as an estimation exercise, not a gate.** Report per-table
wins / losses / ties with an interval, and state the MDE alongside. A NO-GO from this
corpus would mean "no effect large enough for 27 clustered tables to see", which is a
much weaker claim than "tables don't matter" — and the probe's whole purpose was to
stop that question being re-litigated on a hunch. Recording the number honestly still
achieves that; pretending it is confirmatory does not.

**If a confirmatory answer is required, the corpus must grow** — more documents, not
more questions, and ideally documents that are not templated copies of each other.

## Reproduce

```bash
PYTHONPATH=. uv run python scripts/eval/table_probe_corpus_survey.py --corpus data/originals \
    --json data/eval/table_probe_corpus_survey.json
# grid accounting, ablation attribution, pool characterisation and the power table
# were produced by the scripts recorded in this session's scratchpad; all figures
# derive from data/eval/table_probe_corpus_survey.json plus the shipped
# scripts/eval/table_probe_lib.decide_paired.
```
