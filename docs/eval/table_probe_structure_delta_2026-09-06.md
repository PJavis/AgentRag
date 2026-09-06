# Table Probe — 0a Structure Delta

**What this measures:** for every row carrying two or more populated cells,
do those cells still sit together on one line, in document order?
Arm A is the page text the pipeline indexes today
(`get_text("text", sort=True)`); arm B is `table_quality.render_markdown`.

**Read arm A only.** Arm B scores 1.0 by construction — `render_markdown`
emits one row per line — so its column is tautological and is printed only
as a sanity check that the renderer did what it claims.

## Result

| quantity | value |
|---|---|
| tables scored | 27 |
| documents | 7 |
| **median arm-A row adjacency** | **17%** |
| mean arm-A row adjacency | 29% |
| mean of per-document means (unweights the big doc) | 30% |
| median delta (arm B − arm A) | 83% |
| tables fully intact under arm A | 0 |
| tables fully destroyed under arm A | 12 |
| headers still on one line under arm A | 12 |

## Why arm-A rows failed

| reason | rows |
|---|---|
| `cell_fragmented_across_columns` | 79 |
| `ordered_but_split_across_lines` | 16 |
| `out_of_document_order` | 6 |

- `cell_fragmented_across_columns` — the cell's own words were cut apart and
  interleaved with the neighbouring column. The worst case: neither the row
  nor the cell survives, and a phrase query for the cell cannot match it.
- `ordered_but_split_across_lines` — a wrapped cell. No single line could have
  held that row; duller than the reading-order scrambling the spec describes,
  but it costs the reader the row binding just the same.
- `out_of_document_order` — the scrambling proper, as the spec named it.
- `cell_text_absent` — the words are not in the page text at all.

Arm B's markdown puts the whole cell back on one line in every one of these
cases, so the delta is real regardless of which reason dominates. Which one
does dominate still matters: it is the difference between a measurement and a
talking point.

## Decision rule for this step

- **NO-GO for the probe** if arm A already preserves most rows: there is no
  adjacency left for arm B to restore, and the premise the spec rests on is
  false for this corpus. Stop before authoring a single question.
- **Continue to 0b** (retrieval-only rank probe) if arm A is broadly damaged.
  That is evidence the mechanism exists — never evidence that answers improve.

## Caveats that bound any reading

- 25% of corpus pages have no text layer; `find_tables()` is blind there, so
  arm B ≡ arm A on those pages and this measurement never sees them.
- Document clustering is unchanged by this step: the pool is concentrated in a
  few documents, so n_eff is far below the table count. The per-document column
  above is the honest view; the overall mean is not.
- Row adjacency is a proxy for retrievability, not for answer correctness.
  A table can survive flattening and still be answered badly, and vice versa.

## Per document

| doc | tables | mean arm A | mean delta |
|---|---|---|---|
| `2f499de4` | 14 | 32% | 68% |
| `81273a0e` | 5 | 21% | 79% |
| `0bc89e50` | 2 | 40% | 60% |
| `3ffd647b` | 2 | 21% | 79% |
| `5164c4af` | 2 | 0% | 100% |
| `28f3ad1c` | 1 | 71% | 29% |
| `29e99aa0` | 1 | 22% | 78% |

## Per table

| doc | page | rows scored | arm A | arm B | header intact |
|---|---|---|---|---|---|
| `0bc89e50` | 8 | 3 | 0% | 100% | yes |
| `0bc89e50` | 13 | 5 | 80% | 100% | yes |
| `28f3ad1c` | 4 | 7 | 71% | 100% | no |
| `29e99aa0` | 1 | 9 | 22% | 100% | yes |
| `2f499de4` | 8 | 13 | 54% | 100% | no |
| `2f499de4` | 9 | 8 | 0% | 100% | no |
| `2f499de4` | 10 | 5 | 0% | 100% | no |
| `2f499de4` | 10 | 2 | 0% | 100% | yes |
| `2f499de4` | 11 | 8 | 0% | 100% | no |
| `2f499de4` | 12 | 3 | 0% | 100% | yes |
| `2f499de4` | 13 | 9 | 11% | 100% | no |
| `2f499de4` | 14 | 6 | 17% | 100% | yes |
| `2f499de4` | 15 | 7 | 0% | 100% | no |
| `2f499de4` | 16 | 13 | 92% | 100% | no |
| `2f499de4` | 17 | 12 | 83% | 100% | no |
| `2f499de4` | 17 | 10 | 90% | 100% | no |
| `2f499de4` | 18 | 3 | 33% | 100% | no |
| `2f499de4` | 18 | 14 | 71% | 100% | no |
| `3ffd647b` | 4 | 4 | 25% | 100% | yes |
| `3ffd647b` | 5 | 6 | 17% | 100% | yes |
| `5164c4af` | 7 | 5 | 0% | 100% | yes |
| `5164c4af` | 16 | 3 | 0% | 100% | no |
| `81273a0e` | 13 | 7 | 71% | 100% | no |
| `81273a0e` | 24 | 2 | 0% | 100% | yes |
| `81273a0e` | 25 | 3 | 33% | 100% | no |
| `81273a0e` | 25 | 2 | 0% | 100% | yes |
| `81273a0e` | 26 | 2 | 0% | 100% | yes |
