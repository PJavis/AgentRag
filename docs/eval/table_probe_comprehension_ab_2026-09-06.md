# Table Probe — 0c Comprehension Probe

**Question:** 0b showed the gold row reaches the model's context under both
arms (arm A recall@10 = 0.95), so retrieval is not where flattening hurts.
This asks the only hypothesis left: once the row is in context, can the model
still bind a cell to its column when the flat text has shredded the row?

**No judge.** Ground truth is `extract()` itself, so this costs two LLM calls
per table — no reference answers, no gold-context curation, no LLM grader.

## Result

| quantity | arm A (flat) | arm B (+ markdown) |
|---|---|---|
| cells read correctly | 13.7/19 (0.72) | 16.0/19 (0.84) |
| questions whose 3 samples disagreed | 1 | 0 |
| mean token-F1 | 0.695 | 0.843 |
| abstained (`UNKNOWN`) | 5 | 2 |
| surplus tokens beyond the gold cell | 52 | 3 |
| — of those, borrowed from OTHER cells of the same table | 47 (0.90) | 0 (0.00) |

| paired outcome | value |
|---|---|
| B better / worse / same | 3 / 0 / 16 |
| excluded (call errored in one arm) | 0 |
| mean ΔF1 (B − A) | 0.148 |
| 95% CI, resampling documents | [+0.000, +0.269] |
| sign test p | 0.250 |
| shipped rule says | **INCONCLUSIVE** — B leads 3W/0L but p=0.250 >= 0.05 — need more questions |

## Duplication control (arm C)

Arm C is the page text **twice**: the same duplication arm B gets, with
no structure added. B − C is therefore the structure, separated from the
second copy.

| quantity | arm C (duplicated flat) |
|---|---|
| cells read correctly | 0.68 |
| mean token-F1 | 0.619 |
| abstained | 6 |
| B vs C — better/worse/same | 3 / 0 / 16 |
| B vs C — mean ΔF1 | 0.224 |

## The mechanism

The binary row barely moves (one table). The interesting number is the last
one. Under arm A the model usually *finds* the value and then cannot see
where it **stops**: its surplus tokens come overwhelmingly from other cells
of the same table. Under arm B that surplus is essentially gone.

This is why the binary column understates arm A's failures. A containment
check scores "right cell plus the next row glued on" as correct, and in an
answer to a real question it is not — it attributes one row's content to
another. The token-F1 column and the bleed row are measuring that; the
correctness column is blind to it.

## How to read this

- **The shipped verdict above is NOT confirmatory.** The unit is the table,
  52% of the pool is one document, and the rule cannot return GO below 6
  discordant tables. It is printed because the plan says to print it, and
  labelled because the power analysis says it cannot carry a decision.
- **The interval resamples documents, not tables** — the only interval this
  corpus can honestly support.
- **Arm B's context is a superset of arm A's**: the flat text plus the
  markdown. That is exactly what arm B does in production, so it is the right
  comparison — but a win can come from the second copy as easily as from the
  structure, and nothing here separates those two.
- **Temperature is pinned to 0 and every question is asked 3 times.** At the default 0.3 two runs of
  this same code disagreed 15/19 vs 13/19 — variance the size of the effect.
  Residual sample disagreement is counted in the table above, not averaged
  away silently.
- The question is mechanical cell lookup, not a user question. A model that
  can look a cell up may still answer a real question badly, and vice versa.
