# Clean re-measure after the eval-quality finding — 2026-07-15

Executes `docs/HOME-RUN-eval-cleanup-2026-07-15.md`. The 2026-07-14 CRAG A/B
green-lit HippoRAG-2 off a `retrieval_miss`-dominant bucket split — but that split
was contaminated: ~6/9 misses were broken synthetic questions (dangling
demonstratives / exam-item meta-references) that no retriever can answer. This run
re-measures on a **question-quality-filtered** set to get the true headroom and a
genuine bucket split.

Eval set: `data/eval/c2_evalset_n40_clean.jsonl` (n=40 of 44 clean rows: 35
prod_corpus + 9 multihop; context-dependent questions dropped at build + a
second filter pass). Judges: gemini-2.5-pro / deepseek-v4-pro. CRAG OFF (decision
already made; a cleaner base won't flip a +0.015 gap).

## Numbers vs the dirty run

| signal | dirty (2026-07-14) | clean (2026-07-15) | read |
|---|---|---|---|
| system avg | 0.740 | **0.787** | ↑ — broken Qs were dragging it down |
| oracle avg | 0.911 | 0.950 | ↑ |
| oracle − system | +0.171 | **+0.163** | barely moved — the headroom is REAL, not eval noise |
| misses | 9/40 | 8/40 | |
| bucket split (rm / gm / fa) | 6 / 2 / 1 | **3 / 3 / 2** | retrieval_miss NO LONGER dominant |
| judge-noise pearson | 0.942 | 0.934 | healthy |

## Decision — HippoRAG-2 gate: NOT green (shelve for now)

The pre-registered branches:
- **Branch 1 (shelve):** gap shrinks to ~0.05 AND misses not retrieval. Gap did
  NOT shrink (+0.163) → not a clean fit.
- **Branch 2 (build HippoRAG-2):** retrieval_miss dominant. It is NOT — 3/8 overall,
  tied with generation_miss.

Neither fires cleanly, so the verdict rests on inspecting the actual rows (as the
sheet instructs for branch 2):

**The 3 retrieval_miss are single-hop coverage gaps, not multi-hop reasoning:**
- prod_corpus-7 "Các bước đặt ống thông tiểu?" (catheter insertion steps) — gold_overlap 0.24
- prod_corpus-15 "Những yếu tố cần khai thác trong tiền sử bệnh?" (history-taking factors) — 0.09
- prod_corpus-23 "Nguyên nhân gây tiểu không tự chủ?" (incontinence causes) — 0.13

All three are single-hop factual medical questions where the right chunk simply
wasn't retrieved well (low gold overlap, decent rerank on the wrong passages).
**The 9 multi-hop questions nearly all passed (8/9 correct; the one miss,
multihop-2, is a false_abstention with gold already packed).** HippoRAG-2's
multi-hop/entity-graph traversal targets reasoning chains the system is NOT failing
— building it would not obviously close these single-hop coverage misses.

**→ Do NOT green-light HippoRAG-2.** The 2026-07-14 green-light was a dirty-data
artifact. The real, distributed headroom (+0.163) wants:
1. **single-hop retrieval coverage** (chunking / embedding / rerank on the 3 rm rows) —
   the largest genuine bucket, but not a graph problem;
2. **false-abstention tuning** (2 rows: prod_corpus-12 hedged an answerable
   withdrawal-symptoms Q; multihop-2 abstained with gold packed at rerank 0.58) —
   abstain-prompt / floor territory;
3. **answer-prompt work** (1 genuine generation_miss: prod_corpus-25 vignette, gold
   packed, answer wrong).

## Follow-up: filter still leaks 2 broken questions

Two misses bucketed as generation_miss are actually residual broken questions the
filter missed — they inflate the gm count and the headroom:
- **prod_corpus-39 "Whose screen is being viewed?"** — an English image/OCR-caption
  artifact (all real questions are Vietnamese medical). Easy high-precision drop.
- **prod_corpus-33 "Các câu hỏi về đám rối thần kinh cánh tay là gì?"** — a meta
  "what are the questions about X" phrasing that dodges the exam-item patterns.

Next filter pass should add a non-Vietnamese / image-caption guard and a
"các câu hỏi (về|trong) … là gì" meta pattern, then one more clean re-measure.
With those removed the genuine bucket split is ≈ rm 3 / fa 2 / gm 1 — retrieval
coverage is the plurality of *genuine* misses but still not a multi-hop story.

## Notes
- Flywheel re-seeded from the clean run: +63 citation triplets (44 → 107 total) in
  `data/finetune/citation_pairs.jsonl`.
- Clean set is a fresh question sample → aggregates are comparable to the dirty run,
  per-qid is not.
