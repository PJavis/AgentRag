# Table-Data Probe — Design

**Date:** 2026-07-24
**Revised:** 2026-08-02 — corpus survey invalidated four assumptions; instrument corrected.
**Status:** Design approved (rev 2), pending execution
**Author:** dungnq (with Claude)

## 1. Problem & Motivation

Source documents (esp. the real medical PDF corpus) contain tables — dosage grids,
lab reference ranges, comparison matrices. Today tables are **flattened to plain text
at parse time** and are indistinguishable from prose downstream:

- PDF (dominant path): `PDFParser.parse` uses `page.get_text("text", sort=True)`
  (`src/agentrag/ingestion/parsers/pdf_parser.py:83`). Reading-order flattening
  destroys row/column alignment. OCR + vision-LLM fallbacks
  (`pdf_parser.py:86-111`) also return free-form text.
- Chunking is heading/token-window based with no row/table awareness: an oversized
  paragraph goes to `_chunk_section_by_tokens`
  (`src/agentrag/ingestion/chunkers/hybrid_chunker.py:138`, under the branch at :133),
  a blind token window, so a table larger than the chunk window is sliced mid-row.
  (:142 is the paragraph-boundary append, which cannot cut mid-row — three documents
  carried that wrong anchor, which made cross-checking reinforce the error.)
- Retrieval is table-blind. `segment_type="table"` is stamped on Excel/CSV chunks
  only (`src/agentrag/ingestion/pipeline.py:195-197`) and consumed nowhere — it was
  meant for a structured SQL module that has been deleted.
- Structured table extraction (MinerU) was removed as "unproven to matter"
  (`docs/superpowers/specs/2026-06-27-remove-mineru-design.md:19,104`).

**Critical constraint:** there is still **no evidence** tables cause answer misses. No
eval doc under `docs/eval/**` attributes any miss to a mangled table. Rebuilding a
table-extraction subsystem on a hunch repeats the mistake MinerU's removal corrected.

**Therefore this spec is NOT a table-feature build.** It is a cheap, disposable
**measurement probe** whose sole output is a go/no-go decision: *does preserving
table structure at ingest time improve answers enough to justify a real build?*

## 1a. Measured corpus reality (2026-08-02)

Full evidence: `docs/eval/table_probe_corpus_survey_2026-09-06.md`
(supersedes `table_probe_corpus_survey_2026-08-02.md`, whose table-kind counts came
from a classifier that read `extract()` while arm B would have emitted
`to_markdown()`).
Reproduce: `scripts/eval/table_probe_corpus_survey.py`.

| finding | number | effect on this probe |
|---|---|---|
| 116 PDF files → **29 unique** by content hash | 87 redundant copies | every per-doc statistic must run over unique docs |
| **25%** of pages have no text layer (169/684) | OCR/vision path | arm B ≡ arm A on those pages; probe is blind to scanned tables |
| 171 `find_tables()` detections, only **19** are data grids | 11% | target set is small |
| **77%** of detections are not structurally grids | 131/171 | an ungated arm B corrupts them — see below |
| data grids live in **5** unique docs, 9 of 19 in one | — | original "5–8 docs" scope barely exists |

> Counts re-measured 2026-09-06. The rev-2 figures (35 grids, 6 docs, 47% unsafe) came
> from a classifier that read `extract()` while arm B would have emitted
> `to_markdown()`, counted `None` column placeholders as real columns, and treated any
> cell containing a digit as numeric. See
> `docs/eval/table_probe_corpus_survey_2026-09-06.md` §0.

On non-grid detections PyMuPDF's `to_markdown()` produces output strictly worse than the
flat text it replaces (invented `Col2` headers, cells mirrored across columns, one row
per visual line). An ungated arm B would therefore measure
*(gain on real tables) − (damage on layout artifacts)*, and a null result would be
uninterpretable. Arm B is gated **and** renders the gate's own input via
`table_quality.render_markdown`, so that renderer is never invoked.

Where a real grid does exist the hypothesis holds: arm A loses the column binding
entirely (`1 1 1 0` floating free of shattered headers) and arm B restores it. The
probe is worth running — with the corrections in §3.

## 2. Goals & Non-Goals

### Goals
- Produce a directional, **paired** signal (table-preserved vs flat-text ingest) on
  questions that actually depend on a preserved table.
- Gate the decision: build a real table pipeline only if the signal clears the bar.
- Reuse the existing eval/A-B harness; add no production surface area beyond one
  default-off flag and one small pure helper module, both deletable together.
- Be runnable at home whether or not the full stack (ES/TEI/DeepSeek) is up.

### Non-Goals (YAGNI)
- No MinerU / docling / camelot / pdfplumber. The table-preserved variant is the
  cheapest possible (PyMuPDF built-in `find_tables()`).
- No table-aware retrieval, no `segment_type="table"` boosting. Those are candidate
  *outcomes* of a future build, not part of the probe.
- No production config, no default-on behavior. The probe flag is throwaway.
- Not a statistically rigorous benchmark. Small n → directional signal only.
- **Not** a fix for scanned/OCR pages. 25% of the corpus is out of scope by
  construction; the report must say so.

### Explicitly in scope as of rev 2
- **Table-atomic chunking under the flag only.** Not a general chunking feature — a
  precondition without which the probe measures chunking rather than tables (§3 Step 4).

## 3. Design

### Step 0 — Preflight (dedup + branch selector)

Two checks, both cheap, both blocking:

**0a. Corpus dedup.** Run `scripts/eval/table_probe_corpus_survey.py` with
`--unique-list` and `--dedupe-dir`. Every later step consumes the deduplicated
directory, never the raw file listing: `run_table_probe_ab.py --corpus` is pointed at
`--dedupe-dir`'s output. Ingesting 4–7 copies of a document would let duplicate chunks
occupy top-k and swamp the arm difference.

The survey originally emitted only the integer `unique_documents` count and no
directory, so the `<UNIQUE_DOCS_DIR>` the runner asks for did not exist and the only
directory on disk was the raw one with 87 redundant copies — the exact failure this
step exists to prevent, with the report printing "deduplicated to 29 unique documents"
regardless.

**0b. Track selection.** Ping Elasticsearch (host from `elasticsearch_store` config)
and the TEI embedder; check judge key availability (`DEEPSEEK_API_KEY`, or the
configured judge provider key — cross-provider judge validated at pearson 0.921).

- **All up → FULL track** (end-to-end retrieve + answer + judge).
- **Partial → OFFLINE track** (parse + retrieval-recall only; no answer-gen, no judge).

Deliverables: `scripts/eval/table_probe_corpus_survey.py` (done),
`scripts/eval/table_probe_preflight.py` printing `TRACK=full|offline` and the reason.

### Step 1 — Select target tables (not "table-heavy docs")

The unit of analysis is a **table**, not a document. From the survey JSON take the
the survey's **`candidate_tables`** — every gate-passing table, `real_data` and
`nonnumeric` alike (40 total; 27 with ≥3 structured rows and ≥3 columns, the ones that
can carry an alignment question). Do **not** use `data_grids` (19) as the eligible set:
that is the *ranking* class only. Arm B rewrites `nonnumeric` tables too, and a
text-only comparison matrix carries column meaning exactly as a numeric grid does —
this spec says so under Step 3. `per_doc[].candidate_tables` gives page, kind,
structured-row count, columns and measured `est_tokens` / `max_block_tokens`. Rank with `rank_targets`, which interleaves
documents round-robin — a flat cell-count sort put 8 of the top 10 in a single file,
a worse concentration than the doc-level selection this table-level unit replaced,
while the sign test assumes independent draws over the corpus. Filter with
`max_tokens` on **`max_block_tokens`** — the largest rendered block, not the whole-table
`est_tokens` total and certainly not cell count. Cell count predicts rendered size badly
(20 cells → 955 tokens, 104 cells → 429), and filtering on the table total discards the
large multi-row grids that render as several under-budget blocks: at 512 tokens it keeps
12 of 19 grids where the block figure keeps all 19.

PDF-only. DOCX/Excel already emit markdown-ish tables so they would show ~zero A/B
contrast, and OCR pages produce no detections at all.

Deliverable: a target list of `(doc, page, rows, cols)` derived from the survey.

### Step 2 — Author eval questions against the target tables

For each target grid, hand-write **1–2 questions whose answer requires row-column
alignment** (e.g. "for step 1.4, what is the value in the *làm thành thạo* column?").
Author against as many of the 19 grids as time allows — **more questions is the single
highest-leverage change available**, because the discriminating power of the whole probe
is set by n (see §4).

Each row carries gold answer, gold context, **and the source `(doc, page)`** in the
schema `load_local_jsonl` expects (see `src/agentrag/eval/benchmark_datasets.py` and
`vision_evalset_2026-07-19.jsonl`: `id`, `question`, `reference_answer`,
`gold_contexts`), extended with `source_doc` and `source_page`.

`source_page` is **required**, not decorative: the two arms produce different chunk
text for the same table, so gold matching must key on `(doc, page)`. Matching on text
would compare different objects across arms.

**Guardrail:** author fresh against the actual corpus docs. Do NOT reuse
`prod_corpus_evalset_v3.jsonl` gold — the v3 residue set is flagged INVALID vs the real
corpus in project memory.

Deliverable: `data/eval/table_probe_evalset.jsonl`.

### Step 3 — Two ingest variants, with arm B gated

Throwaway, default-off flag `PDF_PRESERVE_TABLES` on `settings` (config.py, following
the existing `env_ignore_empty` singleton pattern used by `VISION_ANSWER_MODEL`).

- **Arm A (control, flag off):** current behavior — `get_text("text", sort=True)`.
- **Arm B (flag on):** in `PDFParser.parse`, **after the OCR/vision block**, call
  `page.find_tables()` and append `table_quality.render_markdown(table.extract(),
  max_tokens=settings.SEARCH_CHUNK_MAX_TOKENS)` to the page text. Tables failing
  `is_safe_to_markdown` render to `""` and are left as flat text only.
  **Never `table.to_markdown()`.** The gate judges `extract()` cells; PyMuPDF's
  renderer invents `ColN` headers and mirrors a single populated cell across
  columns, so gating one representation while emitting the other leaves the
  corruption invisible to the gate — the rev-2 defect this rev exists to close.

The gate is structural, not numeric: a text-only comparison matrix is a real table and
passes; a mirrored prose box or a degenerate 1×N strip does not. Gating on numeric
density instead would silently drop legitimate tables.

Ingest the 29 unique docs **both ways**, one arm at a time. Arms are isolated by
sequence — wipe the corpus indices and Postgres rows, ingest, score, repeat — not by
index name: `get_agent_service()` takes no arguments and `ElasticsearchStore` reads
`settings.ELASTICSEARCH_INDEX_NAME` at construction, so there is no supported way to
point ingest and retrieval at a per-arm index. The wipe is not optional:
`save_document_and_segments` dedupes on `(source_id, content_hash)`, so a re-ingest of
the same bytes returns "skipped" and leaves ES empty.

Arm B emits `table_quality.render_markdown`, **not** `table.to_markdown()`. The gate
judges the `extract()` cells; PyMuPDF's renderer invents `ColN` headers and mirrors a
single populated cell across columns, corruption that does not exist until that renderer
runs and so is invisible to any gate reading `extract()`.

The append happens **after** the OCR/vision block in `PDFParser.parse`. Before it, the
appended markdown would lengthen `stripped` past `PDF_OCR_MIN_TEXT_CHARS` and stop arm B
taking an OCR fallback arm A takes — arm B losing whole pages for reasons unrelated to
tables, and falsifying the "byte-identical on the 25%" claim this document makes below.

Deliverables: `PDF_PRESERVE_TABLES` in config, the gated branch in `pdf_parser.py`,
`table_quality.py` (done, tested).

### Step 4 — Chunk-integrity precondition (blocking)

`SEARCH_CHUNK_MAX_TOKENS=512`; `hybrid_chunker.py:138` slices an oversized paragraph on
a blind token window. A 17×8 markdown table exceeds the window and is cut mid-row, which
destroys arm B's advantage *before retrieval sees it*.

**Solved in the renderer, not in the chunker (rev 3).** `render_markdown` packs rows
into header-repeating blocks sized to `SEARCH_CHUNK_MAX_TOKENS` and separates them with
a blank line, so every block is an ordinary paragraph that already fits the window and
the oversized branch is never reached. Measured on the corpus: 40 gate-passing tables →
55 blocks, 0 over budget, worst 500 tokens. A fixed row count cannot do this — tokens per cell
range 4.1–47.8 (median 9.4), and the `ROWS_PER_BLOCK=8` fallback gives 53–1090-token
blocks, 13% of which overflow.

This is deliberately *not* a table-aware branch in `hybrid_chunker`. That would put
probe-scoped behaviour on the hot path for every ingested document, and the flag it
would read is not imported there.

So before any scoring the check still runs: for each target row, assert one arm-B chunk
holds it together with its header. If rows are being cut, the block packing is not
reaching the chunker — fix that rather than proceeding. A row that still cannot survive
intact means **that question is excluded from the comparison** and the exclusion is
recorded in the report.

Skipping this check does not make the probe cheaper; it makes the result mean something
other than what the report will claim.

Deliverable: `scripts/eval/table_probe_chunk_integrity.py` → per-target intact yes/no.

### Step 5 — Run the A/B, paired

Model the runner on `scripts/eval/run_vision_e2e_ab.py` (two arms as separate
processes, each mutating the settings singleton; `--arm` runs one, `--compare` diffs
two json reports into markdown).

Both tracks score **the same question against both arms** and compare per question.
Only questions whose gold table (a) passed the arm-B gate and (b) survived the chunk
integrity check enter the comparison — questions where the arms produce identical
input cannot inform the decision and only add noise.

- **FULL track:** for each probe question run `agent.chat(...)` against arm A's index
  then arm B's, score with `score_correctness`
  (`src/agentrag/eval/correctness_judge.py`), grounded with gold context. Also record
  gold-table recall@k and whether the answer cites the gold table chunk.
- **OFFLINE track:** skip `agent.chat`. Per-question gold-table **recall@k matched on
  `(doc, page)`** — never on chunk text — for arm A vs arm B, with chunk integrity as
  supporting evidence.

`k` must equal production retrieval `top_k` or the recall number is not comparable to
anything — read it from config (`settings.RETRIEVAL_TOP_K` = 10; the agent's retrieval
tool requests `settings.AGENT_TOOL_TOP_K` = 30). Do not hard-code `8`: rev 1 raised
"confirm production top_k" as an open question and rev 2 answered it with a number that
matches neither setting.

Deliverables: `scripts/eval/run_table_probe_ab.py` (`--arm a|b`, `--compare`), per-arm
json reports under `data/eval/`.

### Step 6 — Decide

See §4 for the decision rule. Either way, log the number and the n. Report:
`docs/eval/table_probe_<date>.md`.

## 4. Metrics & Decision Rule

The rev-1 rule — mean correctness delta ≥ +0.10 — is **withdrawn**. At n=8–12 one
question flipping *is* 0.10, and measured A/B noise on this stack is ~0.3
(project memory, gate/abstain A/Bs). That rule decides a build on a coin flip.

Rev 2 uses a **paired per-question** comparison over the eligible subset:

| track | per-question outcome | statistic | decision |
|---|---|---|---|
| FULL | `score_correctness` B vs A, same question | wins / losses / ties + two-sided sign test on the non-ties | see below |
| OFFLINE | gold table in top-k, B vs A, matched on `(doc,page)` | same | same |

- **GO** — B wins at least twice as often as it loses, **and** the sign test on the
  discordant pairs gives p < 0.05.
- **NO-GO** — B does not win more than it loses.
- **INCONCLUSIVE** — B wins more, but the sign test does not reach p < 0.05. This is
  the honest outcome at small n and must be reported as such. It is *not* a GO.

**Unit of analysis: the TABLE** (added rev 3, 2026-09-06). Questions authored against
one table are not independent draws — arm B either restores that table's row/column
binding or it does not, and every question on it inherits that one fact. Per-question
outcomes must be aggregated to one win/loss/tie per table before the sign test.

This inverts the remedy stated above: **more questions per table does not increase
power.** 2–3 per table are worth authoring because they make that table's outcome less
likely to be misclassified, but n is the table count either way. The remedy for low
power is more *tables*, which on this corpus means more *documents*.

**Minimum detectable effect** (this spec previously stated none). The rule cannot
return GO on fewer than **6 discordant tables** (5W/0L → p = 0.0625; 6W/0L → p =
0.0312). With the corpus's 27 structured candidate tables, 80% power requires arm B to
help ~42% of them; 52% of those tables live in one document, so if outcomes cluster by
document the effective n approaches 7 and GO is unreachable at any effect size.

**Consequence: this corpus does not support a confirmatory go/no-go.** See
`docs/eval/table_probe_power_analysis_2026-09-06.md` for the accounting, the power
table, and the recommendation (report as estimation, not as a gate).

Report must state, alongside the result: n eligible, n excluded for chunk-splitting,
n excluded because the arms produced identical text, and the 25% OCR blind spot.

A mean delta may be reported as colour. It must not drive the decision.

## 5. Deliverables

| # | artifact | status |
|---|---|---|
| 1 | `src/agentrag/ingestion/parsers/table_quality.py` + tests | **done** |
| 2 | `scripts/eval/table_probe_corpus_survey.py` + tests | **done** |
| 3 | `docs/eval/table_probe_corpus_survey_2026-09-06.md` | **done** |
| 4 | `scripts/eval/table_probe_preflight.py` — `TRACK=full\|offline` | to do |
| 5 | `data/eval/table_probe_evalset.jsonl` — questions w/ `source_doc`+`source_page` | to do |
| 6 | `PDF_PRESERVE_TABLES` flag + gated branch in `pdf_parser.py` | to do |
| 7 | Two-arm ingest over unique docs, isolated indices | to do |
| 8 | `scripts/eval/table_probe_chunk_integrity.py` | to do |
| 9 | `scripts/eval/run_table_probe_ab.py` (`--arm`, `--compare`) | to do |
| 10 | `docs/eval/table_probe_<date>.md` — paired result + decision | to do |

## 6. Risks & Mitigations

- **Small n / weak power.** The binding constraint. Mitigate by authoring against as
  many of the 19 grids as possible, reporting INCONCLUSIVE honestly, and never
  restating a mean delta as a decision. No ranking or gating change can manufacture
  more n.
- **Instrument damage (was unmitigated in rev 1, mis-mitigated in rev 2).** Arm B is
  gated on `is_safe_to_markdown` so the treatment does not touch the 77% of detections
  that are layout artifacts — *and* it emits `render_markdown`, built from the same
  cells the gate judged. Gating `extract()` while emitting `to_markdown()` left the
  corruption invisible to the gate: 69 of the 90 detections rev 2 would have passed
  carried an invented `ColN` header.
- **Chunking masks the effect (was unmitigated in rev 1).** Blocking integrity check
  in Step 4; `render_markdown` packs header-repeating blocks to
  `SEARCH_CHUNK_MAX_TOKENS` so blocks fit by construction (measured 0 of 55 over
  budget); exclude and report otherwise. No production chunker change.
- **Cross-arm gold mismatch (was unmitigated in rev 1).** Recall matched on
  `(doc, page)`, not chunk text, because arms produce different text for one table.
- **Evalset outliving its corpus (the 2026-07-13 landmine, re-created here by the
  wipe-and-re-ingest).** Rows carry `corpus_docs_sha` from the survey and the runner
  refuses to score on a mismatch. Note this is NOT `eval/corpus_fingerprint.py`,
  which hashes `(document_title, segment_count)`: arm B changes segment counts by
  design, so that fingerprint would report a mismatch on a *correct* run.
  `corpus_docs_sha` hashes deduplicated document content and is arm-independent.
- **Duplicate corpus (was unknown in rev 1).** Dedup in Step 0a; all steps consume the
  29 unique docs.
- **Embedding bias against markdown.** Pipe-table markdown may embed worse with e5 than
  prose, so arm B can lose recall while carrying better content. This is why FULL track
  (answer correctness) is the preferred track and recall is supporting evidence.
- **OCR blind spot.** 25% of pages produce no detections; arm B ≡ arm A there. Stated
  as a scope limit in the report, not mitigated.
- **Judge grounding.** Use the grounded cross-provider judge (`score_correctness`,
  pearson 0.921); pass gold context so the judge is grounded, not free-scoring.
- **Corpus validity.** Author fresh table Qs; do NOT lean on `prod_corpus_evalset_v3`
  gold (flagged INVALID vs real corpus in memory).
- **False negative from a weak variant.** Arm B is intentionally minimal
  (`find_tables()` only). If the signal is near-but-below the bar, note that a stronger
  extractor *might* clear it — but the burden stays on evidence; do not build on "might".
- **Arm isolation leak.** Ingesting both arms into the same index would contaminate
  retrieval. Arms are isolated by **sequence** — wipe the corpus ES indices and the
  Postgres rows, ingest one arm, score it, repeat — because per-arm index names are not
  reachable: `get_agent_service()` takes no arguments and `ElasticsearchStore` reads
  `settings.ELASTICSEARCH_INDEX_NAME` at construction. The wipe is mandatory:
  `save_document_and_segments` dedupes on `(source_id, content_hash)`, so re-ingesting
  the same bytes returns "skipped" and leaves ES empty.
- **Arm B fails to score a question the other arm scored** (timeout, judge error).
  Silently intersecting the arms drops exactly those questions from both numerator and
  denominator — survivorship bias in the direction that favours GO, and arm B is the arm
  more likely to time out. Every non-compared question is counted in the report.
- **Stack unavailable at run time.** Preflight downgrades FULL→OFFLINE automatically.

## 7. Open Questions

- Whether the corpus duplication should be fixed at the source (ingest-time content
  hash dedup) rather than worked around per-analysis. Out of scope here, but it is the
  higher-value thread — it affects every eval run, not just this probe.
- Whether `29e99aa0` (107 pages, 1 with a text layer) is worth including at all.

## 8. Changelog

**rev 3 (2026-09-06)** — after a code review of the rev-2 instrument, re-measured in
`docs/eval/table_probe_corpus_survey_2026-09-06.md`: the gate classified `extract()`
while arm B would have emitted `to_markdown()`, so 77% of gate-passing detections
carried an invented `ColN` header the gate could not see — arm B now renders with
`table_quality.render_markdown` from the cells the gate judged. Column counting no
longer treats `None` placeholders as columns (new `single_column` kind, 50 detections);
mirroring is detected per adjacent column pair; numeric density uses digit density
rather than "contains a digit" AND excludes a leading ordinal index column, so a
procedure checklist whose only digits are the row counter is no longer a probe target
(data grids 35 → 19 across 5 docs). The table append
moves *after* the OCR block so arm B cannot change OCR routing. Table-atomic chunking is
withdrawn — `render_markdown` packs header-repeating blocks to
`SEARCH_CHUNK_MAX_TOKENS` (measured 0 of 55 blocks over budget), so no production
chunker change ships. Step 0a now writes the deduplicated corpus to disk. Timeouts are
excluded rather than scored 0.0; `k` is read from `settings.RETRIEVAL_TOP_K` rather than
hard-coded to 8; the report carries its track so an OFFLINE NO-GO is labelled as the
weaker evidence this document already said it was.

**rev 2 (2026-08-02)** — after `docs/eval/table_probe_corpus_survey_2026-08-02.md`:
corpus dedup added as Step 0a; arm B gated on `is_safe_to_markdown`; chunk-integrity
promoted to a blocking precondition with table-atomic chunking under the flag; recall
matching keyed on `(doc,page)`; +0.10 mean-delta gate replaced with a paired sign test
plus an explicit INCONCLUSIVE outcome; unit of analysis changed from document to table.
