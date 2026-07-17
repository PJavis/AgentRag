# Recall-gap diagnosis (HOME-RUN-recall-gaps Phase 0) — 2026-07-17

Classifies the 2 recall gaps as Lever A (chunking) / B (query-side/HyDE) / C (embedding),
per `docs/HOME-RUN-recall-gaps-2026-07-17.md`. Measured, not assumed.

## Findings

| row | gold indexed? | split? | query↔gold word overlap | gold rank on the question (hybrid / dense / sparse) | class |
|---|---|---|---|---|---|
| prod_corpus-23 | yes (1.00 by gold-text probe) | **WHOLE** (one segment) | **0.000** | absent from top-100 (best hit 0.15) in all 3 modes | **B/C — vocabulary/embedding** |
| prod_corpus-15 | yes (1.00) | WHOLE | 0.032 | absent from top-100 (best 0.09–0.15) | vague Q — accept |

Method: `scratchpad/phase0_recall.py` — probe the gold TEXT (confirms single-segment
indexing) vs probe the QUESTION (gold's rank per mode). "rank" columns show only
distractors (overlap ≤0.15) surface for the question — the true gold (1.00) is **absent
from the top-100 in every mode**.

## Interpretation

**Not Lever A (chunking).** Both golds are indexed as ONE whole segment (the gold-text
probe returns 1.00). Nothing is split across chunk boundaries.

**Not breadth.** Gold is absent from the top-100, not sitting at rank 51–100. Raising
`RETRIEVAL_RAW_QUERY_TOP_K` further won't reach it.

**It's a vocabulary/semantic gap (Lever B, fallback C).** The question is broad and general
("Các nguyên nhân thường gặp gây tiểu không tự chủ" = common causes of incontinence) while
the gold is a bare list that **never names the topic** — "tiểu không tự chủ" does not appear
in it (query↔gold word overlap = 0.000). So:
- **BM25 cannot match** — zero shared vocabulary.
- **Dense embedding does not bridge it** — the broad query embeds far from the specific,
  unlabelled list (gold absent from dense top-100).
- CR was supposed to add the missing topic label, but the earlier 1-doc CR test
  (`rerank_before_trim` follow-up) showed the contextualizer **mislabelled** this chunk
  ("male infertility / ejaculation") — CR is not a reliable fix here.

## Lever choice

- **prod_corpus-23 → Lever B (HyDE) first.** HyDE (`QueryRewriter.make_hyde_text`, wired via
  `knowledge_service.py`, gated by `QUERY_REWRITE_ENABLED`, default OFF) generates a
  hypothetical answer that WOULD contain the specific terms the gold list uses (drugs,
  alcohol, bladder-neck dysfunction…), closing the vocabulary gap on the dense side. Cheapest
  lever — one flag, no re-ingest, no train. Fallback = Lever C (embedding retrain) only if
  HyDE misses the bar.
- **prod_corpus-15 → accept.** Vague/underspecified question (no disease anchor; gold is the
  incontinence list but the question asks generic "history-taking factors"). Already removed
  from the eval set (clean_v2 now 41 rows). Not a retrieval-build target.

## Lever B result — FAILED → accept the tail

Tested HyDE on -23:
- HyDE **was already enabled** in the eval config (`.env QUERY_REWRITE_ENABLED=true`, applied via
  `KnowledgeService.bootstrap_search` in the graph-agent path) — so every clean-v2 probe,
  including the 0.904 baseline, already ran with HyDE, and -23 missed *with it on*.
- Isolated HyDE check: the generated hypothetical is topically correct but names **generic**
  incontinence causes (hormonal, diabetes, obesity); the gold is a **hyper-specific
  surgical/structural list** (bladder-neck dysfunction, hemitrigone, TURP damage).
  hyde↔gold word overlap = 0.15, and with the hypothetical as `dense_query` the gold is **still
  absent** from the top-100 (best hit 0.13). HyDE cannot bridge a query→gold gap this specific.

**Verdict: accept the 1-row tail.** Lever A ruled out (whole segment, not split); Lever B failed
(HyDE already on, doesn't surface the gold); Lever C (embedding retrain + full re-ingest, dim
change) is disproportionate for 1/41 at the +0.044 eval ceiling — and prior evidence
(2026-06-28) shows recall lift did not propagate to correctness. The gold is a pathological bare
list with no topic label (same class as the removed vague -15); the honest read is that closing
it wants **better gold, not more retrieval** — exactly the sheet's accept-the-tail case. No
config change.
