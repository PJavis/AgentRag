# Audit — Roadmap Items Justified by "Better Recall" (2026-09-06)

Triggered by 0b (`table_probe_retrieval_ab_2026-09-06.md`). The instruction was:
*any table feature justified as a recall improvement rests on a false premise —
audit the whole list, not just arm B.*

## 1. What 0b licenses, precisely

Overclaiming here would repeat the error being audited, so the scope is stated
before the findings.

| | |
|---|---|
| measured on | 19 mechanical row lookups (`"<row label> <column header>"`) over 27 gate-passing tables in 7 documents |
| recall@10, arm A | **bm25 0.95, rrf 0.95**, dense 0.53 |
| effect of arm B | none on any retriever (every interval spans zero) |

**What this licenses:** for *table-row lookups on this corpus*, the fused retrieval
path is at ceiling. A feature whose payoff is "the table row will be retrieved more
often" has ~5 points of headroom on the fused path and cannot pay for itself.

**What it does not license:**
- Nothing about non-table questions. This is 19 table lookups.
- Nothing about **dense** retrieval. Dense sits at 0.53 — but these queries are
  keyword-shaped by construction, which is the worst case for a bi-encoder and the
  best case for BM25. That gap is expected and is **not** evidence of a dense
  weakness on natural questions.
- Nothing about precision, ranking quality above rank 10, or answer quality.

## 2. Findings

| # | item | where it is justified | verdict | action |
|---|---|---|---|---|
| 1 | "Retrieval is table-blind" as motivation for the table probe | `specs/2026-07-24-table-data-probe-design.md:24` | **Observation true, implication falsified.** `segment_type="table"` really is consumed nowhere, but fixing that cannot buy recall on table rows — the fused path already returns them at 0.95 | keep the observation, strike the recall implication |
| 2 | Candidate follow-ups: "table-aware retrieval", `segment_type="table"` boosting | `specs/2026-07-24-table-data-probe-design.md:84` | **Premise falsified.** Both are recall plays on exactly the lookups measured at ceiling | do not fund as recall work. Re-propose only with a comprehension or precision justification and a stated headroom |
| 3 | Reviving structured table extraction (MinerU-class) | `specs/2026-06-27-remove-mineru-design.md` | **Premise falsified for recall.** The live justification is now comprehension (0c: 90% → 0% cross-cell bleed), which is a much narrower and cheaper claim — arm B already covers it | keep MinerU removed. Arm B is the cheap form of the same benefit |
| 4 | Table-atomic chunking so rows are not sliced | plan Task 6 | **Already closed** in the renderer (blocks packed under the chunk window, header repeated), and 0b shows row slicing was not costing recall anyway | no action |
| 5 | Multi-Query / query translation "tăng recall" | `report/ch2.md:117,259` | **Not falsified.** General-purpose, aimed at vocabulary mismatch on natural questions, which 0b did not measure | out of scope. If it is ever justified *by table cases*, that justification is now dead |
| 6 | FT embedding / dense retrieval investment | `eval/finetune_gate_2026-06-27.md:131` | **Not falsified.** Dense's 0.53 here is a keyword-query artifact, not a measured user-facing gap | no change. Do not cite 0b's 0.53 as a dense deficiency |
| 7 | "Wants better gold, not more retrieval" | `eval/recall_gap_diagnosis_2026-07-17.md:66` | **Corroborated.** Independently reached the same conclusion on a different question set | none — worth citing together |

## 3. The transferable lesson

Items 1–3 shared one unexamined step: *tables are mangled → therefore they retrieve
badly → therefore fix the mangling*. The middle step was never measured, and it is
false. Lexical retrieval is bag-of-words, so a row shredded across lines still
matches every one of its own terms — the damage 0a measured (median 17% row
adjacency) simply does not reach the retriever.

**Standing rule this establishes:** before funding anything justified as a recall
improvement, measure the current recall on the cases it targets. If the baseline is
already at ceiling, the feature cannot pay for itself no matter how real the defect
it fixes. 0a's defect was entirely real; it was just real somewhere else.
