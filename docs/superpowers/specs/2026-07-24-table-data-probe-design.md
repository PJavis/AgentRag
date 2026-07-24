# Table-Data Probe — Design

**Date:** 2026-07-24
**Status:** Design approved, pending spec review
**Author:** dungnq (with Claude)

## 1. Problem & Motivation

Source documents (esp. the real medical PDF corpus) contain tables — dosage grids,
lab reference ranges, comparison matrices. Today tables are **flattened to plain text
at parse time** and are indistinguishable from prose downstream:

- PDF (dominant path): `PDFParser.parse` uses `page.get_text("text", sort=True)`
  (`src/agentrag/ingestion/parsers/pdf_parser.py:83`). Reading-order flattening
  destroys row/column alignment. OCR + vision-LLM fallbacks
  (`pdf_parser.py:97`, `:211`) also return free-form text.
- Chunking is heading/token-window based with no row/table awareness
  (`src/agentrag/ingestion/chunkers/hybrid_chunker.py:156`), so a table larger than
  `max_tokens` is sliced mid-row.
- Retrieval is table-blind. `segment_type="table"` is stamped on Excel/CSV chunks
  only (`src/agentrag/ingestion/pipeline.py:195-197`) and consumed nowhere — it was
  meant for a structured SQL module that has been deleted.
- Structured table extraction (MinerU) was removed as "unproven to matter"
  (`docs/superpowers/specs/2026-06-27-remove-mineru-design.md:19,104`).

**Critical constraint:** there is currently **no evidence** tables cause answer
misses. No eval doc under `docs/eval/**` attributes any miss to a mangled table.
Rebuilding a table-extraction subsystem on a hunch repeats the mistake MinerU's
removal corrected.

**Therefore this spec is NOT a table-feature build.** It is a cheap, disposable
**measurement probe** whose sole output is a single go/no-go number: *does preserving
table structure at ingest time improve answers enough to justify a real build?*

## 2. Goals & Non-Goals

### Goals
- Produce a directional lift number (table-preserved vs flat-text ingest) on a small
  set of table-dependent questions.
- Gate the decision: build a real table pipeline only if lift clears a threshold.
- Reuse the existing eval/A-B harness; add no production surface area.
- Be runnable at home whether or not the full stack (ES/TEI/DeepSeek) is up.

### Non-Goals (YAGNI)
- No MinerU / docling / camelot / pdfplumber. The table-preserved variant is the
  cheapest possible (PyMuPDF built-in `find_tables()`).
- No table-aware retrieval, no `segment_type="table"` boosting, no table-aware
  chunking. Those are candidate *outcomes* of a future build, not part of the probe.
- No production config, no default-on behavior. The probe flag is throwaway.
- Not a statistically rigorous benchmark. Small n → directional signal only.

## 3. Design

### Step 0 — Preflight (branch selector)

A small check decides which track runs, so the probe never hard-blocks on infra:

- Ping Elasticsearch (host from `elasticsearch_store` config) and the TEI embedder.
- Check judge key availability (`DEEPSEEK_API_KEY`, or the configured judge provider
  key — cross-provider judge validated at pearson 0.921).

Result:
- **All up → FULL track** (end-to-end retrieve + answer + judge).
- **Partial → OFFLINE track** (parse + retrieval-recall only; no answer-gen, no judge).

Deliverable: `scripts/eval/table_probe_preflight.py` (or reuse an existing health
check) that prints `TRACK=full|offline` and the reason.

### Step 1 — Find table-heavy docs

Scan the real prod corpus (the source behind `data/eval/prod_corpus_evalset*.jsonl`).
Count tables per doc:
- PDF: `page.find_tables()` count per page (PyMuPDF, already a dependency).
- DOCX/HTML: count GFM pipe-table blocks after MarkItDown.
- Excel/CSV: trivially tabular (likely excluded — already handled well).

Rank docs by table density; pick **5–8 PDF docs** with substantive **numeric** tables
(doses, lab ranges, comparisons — where cell alignment carries the answer). The probe
targets PDF specifically because that is where structure is lost; DOCX/Excel already
emit markdown-ish tables so they would show ~zero A/B contrast.

Deliverable: `scripts/eval/table_probe_find_docs.py` → prints ranked
`(doc_path, page, table_preview)` list.

### Step 2 — Author eval questions

For each selected table, hand-write **1–2 questions whose answer lives in a specific
cell / requires row-column alignment** (e.g., "What is the maintenance dose of X for a
patient aged Y?"). Target **8–12 questions total**.

Each row carries: gold answer + gold context (the table text) + source doc id, in the
**same schema `load_local_jsonl` expects** (see `src/agentrag/eval/benchmark_datasets.py`
and the `vision_evalset_2026-07-19.jsonl` shape: `id`, `question`, `reference_answer`,
`gold_contexts`).

**Guardrail:** author fresh against the actual corpus docs. Do NOT reuse
`prod_corpus_evalset_v3.jsonl` gold — the v3 residue set is flagged INVALID vs the real
corpus in project memory.

Deliverable: `data/eval/table_probe_evalset.jsonl`.

### Step 3 — Two ingest variants

Introduce a throwaway, default-off flag `PDF_PRESERVE_TABLES` on `settings`
(config.py, following the existing `env_ignore_empty` singleton pattern used by
`VISION_ANSWER_MODEL`).

- **Arm A (control, flag off):** current behavior — `get_text("text", sort=True)`.
- **Arm B (flag on):** in `PDFParser.parse`, call `page.find_tables()`; for each
  detected table emit `table.to_markdown()` (GFM) inline at its position, and take the
  remaining non-table text as today. No other change.

Ingest the same 5–8 docs **both ways** into two isolated targets (two index names or a
namespace/index suffix per arm) so retrieval can be pointed at each arm independently.

Deliverable: the `PDF_PRESERVE_TABLES` branch in `pdf_parser.py` + an ingest helper
that stamps the arm into the index name.

### Step 4 — Run the A/B

Model the runner directly on `scripts/eval/run_vision_e2e_ab.py` (two arms as separate
processes, each mutating the settings singleton; `--arm` runs one, `--compare` diffs
two json reports into markdown).

- **FULL track:** for each probe question, run `agent.chat(...)` against arm A index
  then arm B index; score with `score_correctness`
  (`src/agentrag/eval/correctness_judge.py`). Also record gold-table recall@k and
  whether the answer cites the gold table chunk.
- **OFFLINE track:** skip `agent.chat`. Metrics only:
  1. **Chunk integrity** — does the gold table survive chunking as one intact chunk
     (not split mid-row)? (inspect chunks post-ingest per arm)
  2. **Retrieval recall@k** of the gold table chunk for each probe query, arm A vs B.

Deliverables: `scripts/eval/run_table_probe_ab.py` (`--arm a|b`, `--compare`),
per-arm json reports under `data/eval/`.

### Step 5 — Decide (go/no-go gate)

Primary metric:
- FULL track → mean judge correctness on the probe subset.
- OFFLINE track → gold-table recall@k (with chunk-integrity as supporting evidence).

Lift = B − A.

- **Lift ≥ +0.10** on the primary metric → tables materially help → **greenlight** a
  real table build; write a follow-up design spec (table-aware parse + chunk +
  retrieval scope decided then).
- **Lift < +0.10** → tables do not move the needle → **close it**, keep flat-text,
  record the number so this is not re-litigated on a hunch.

Either way, log the number. Report: `docs/eval/table_probe_<date>.md` (mirror the
vision A/B report format at the tail of `run_vision_e2e_ab.py`).

## 4. Metrics & Thresholds Summary

| Track   | Primary metric                  | Supporting                    | Go threshold |
|---------|---------------------------------|-------------------------------|--------------|
| FULL    | mean judge correctness (B−A)    | gold-table recall@k, cite hit | ≥ +0.10      |
| OFFLINE | gold-table recall@k (B−A)       | chunk integrity (intact?)     | ≥ +0.10      |

n = 8–12 questions → **directional only**, stated explicitly in the report.

## 5. Deliverables (home coding checklist)

1. `scripts/eval/table_probe_preflight.py` — prints `TRACK=full|offline`.
2. `scripts/eval/table_probe_find_docs.py` — ranked table-heavy docs.
3. `data/eval/table_probe_evalset.jsonl` — 8–12 hand-authored table Qs.
4. `PDF_PRESERVE_TABLES` flag (config.py) + `find_tables()/to_markdown()` branch in
   `src/agentrag/ingestion/parsers/pdf_parser.py`.
5. Two-arm ingest (isolated indices per arm).
6. `scripts/eval/run_table_probe_ab.py` (`--arm`, `--compare`), modeled on
   `run_vision_e2e_ab.py`.
7. `docs/eval/table_probe_<date>.md` — lift number + go/no-go decision.

## 6. Risks & Mitigations

- **Small n / noisy signal.** Mitigate: treat as directional go/no-go, state n in the
  report, do not present as a benchmark.
- **Judge grounding.** Use the grounded cross-provider judge (`score_correctness`,
  pearson 0.921); pass gold context so the judge is grounded, not free-scoring.
- **Corpus validity.** Author fresh table Qs; do NOT lean on `prod_corpus_evalset_v3`
  gold (flagged INVALID vs real corpus in memory).
- **False negative from a weak variant.** Arm B is intentionally minimal
  (`find_tables()` only). If lift is near-but-below threshold, note that a stronger
  extractor *might* clear it — but the burden stays on evidence; do not build on
  "might".
- **Arm isolation leak.** Ingesting both arms into the same index would contaminate
  retrieval. Enforce distinct index names per arm; assert in the runner.
- **Stack unavailable at run time.** Preflight downgrades FULL→OFFLINE automatically;
  the probe still yields a recall-based signal.

## 7. Open Questions (resolve at home)

- Exact real-corpus source path for Step 1 (the docs behind `prod_corpus_evalset`).
- `recall@k` — which `k` matches production retrieval top_k (probe fidelity note
  referenced top_k=8).
- Whether any non-PDF table docs are worth including (default: PDF-only).
