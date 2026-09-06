# Table Probe — 0b Retrieval-Only Rank Probe

**Question:** does arm B make the gold row easier to *retrieve*? If not, the
answer model never sees the row and no end-to-end gain is possible — a NO-GO
bought for a day of compute instead of the weeks the full probe costs.

**Queries are mechanical** (`"<row label> <column header>"` straight out of
`extract()`), so nothing is tuned toward either arm. They are therefore not
user questions: this step can kill the probe, never green-light a build.

Questions scored: **19** — ranked against the whole corpus.
Dropped: 8 tables with no askable column, 0 rows too
generic to identify by their own words.

| retriever | MRR arm A | MRR arm B | mean ΔRR | 95% CI (doc-clustered) | recall@10 A→B | W/L/T | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| bm25 | 0.869 | 0.921 | 0.052 | [+0.000, +0.211] | 0.95 → 1.00 | 1/0/18 | 1.000 |
| dense | 0.401 | 0.354 | -0.046 | [-0.110, +0.006] | 0.53 → 0.53 | 8/3/8 | 0.577 |
| rrf | 0.730 | 0.715 | -0.015 | [-0.048, +0.042] | 0.95 → 0.95 | 2/4/13 | 0.688 |

## How to read this

- **ΔRR > 0 means arm B retrieved the gold row higher.** The outcome is
  continuous, so it carries magnitude the binary sign test throws away.
- **The interval resamples documents, not tables.** 52% of the pool sits in
  one file; per-table resampling would report precision that is not there.
- **A ceiling bounds the lexical column.** The query is built from the row's
  own words, and BM25 is bag-of-words — so a row shredded across lines still
  matches every term. Arm A therefore starts near the top of the scale and
  there is almost nothing left for arm B to add. Row adjacency, which 0a
  showed collapsing to 17%, is simply not what lexical retrieval consumes.
  **Dense is where the headroom is**, and dense is where to read the answer.
- **BM25 here is a simplified lexical baseline** (whitespace + lowercase), not
  Elasticsearch's Vietnamese analyzer. Identical across arms, so the delta is
  fair; the absolute MRR is not production retrieval.
- The gold chunk is bag-of-words overlap with the row, never a contiguous
  substring — arm A shreds cells, and a substring gold would have defined arm
  A out of the comparison instead of measuring it. No page filter either: the
  chunker labels a chunk by the markers inside it, so a chunk holding one
  page's tail and the next page's marker is labelled with the next page. Arm
  B's longer pages shift those boundaries more often, so filtering on page
  deleted arm B's gold chunks — measured, not hypothesised.
- Vision OCR fallback is forced off in both arms: it calls a remote model, so
  it is neither deterministic nor available offline.

## Verdict

**No retriever shows arm B raising the gold row** — every interval spans
zero.

But the kill condition this step was written with does **not** fire, and
saying it did would be wrong. That condition was *"arm B does not improve
retrieval, therefore the answer model never sees the row"*. Its premise is
false here: arm A already retrieves the gold row at recall@10 = 0.95. The model sees the row under **both** arms.

What this step actually settles:

- **Retrieval is not where flattening hurts.** 0a measured row adjacency
  collapsing to 17%, and that damage does not propagate here, because
  lexical retrieval is bag-of-words: a row shredded across lines still
  matches every one of its own terms.
- **Arm B has no retrieval benefit to offer.** Not a wash to be re-run
  bigger — there is no headroom on the lexical side and no movement on the
  dense side.

What remains untested is **comprehension**: once the row is in context, can
the answer model still bind a cell to its column when the text is shredded?
0b cannot answer that, and must not be read as if it had. Any remaining
gain for arm B has to come from comprehension, not recall.

## What this step cannot say

- Nothing about answer correctness. The queries are the row's own words, not
  user questions.
- Nothing about the 25% of pages with no text layer: `find_tables()` is blind
  there and arm B is byte-identical to arm A.
- Nothing that repairs the clustering bound. 52% of the pool is one document;
  n_eff stays far below the table count no matter how continuous the outcome.
