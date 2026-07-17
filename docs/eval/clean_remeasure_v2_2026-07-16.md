# Clean re-measure v2 — both filter leaks removed (2026-07-16)

Follow-up to `clean_remeasure_2026-07-15.md`, which found the "clean" set still
leaked 2 broken questions (an English OCR/image-caption artifact and a
"các câu hỏi về X là gì" enumerate-the-exam-questions meta). Both are now dropped
by `question_quality.py` (pure-ASCII guard + `(các|những) câu hỏi` pattern,
commit d111cf1). This run re-probes the fully-clean set.

Eval set: `data/eval/c2_evalset_n40_clean_v2.jsonl` (n=42: 33 prod_corpus + 9
multihop). Judges: gemini-2.5-pro / deepseek-v4-pro. CRAG OFF.

## Headroom shrinks as the eval set gets cleaner

| run | eval set | system avg | oracle−system | misses | bucket split (rm/gm/fa) |
|---|---|---|---|---|---|
| dirty (07-14) | c2 n40 (unfiltered) | 0.740 | +0.171 | 9 | 6 / 2 / 1 |
| clean v1 (07-15) | filtered, 2 leaks left | 0.787 | +0.163 | 8 | 3 / 3 / 2 |
| **clean v2 (07-16)** | **fully filtered** | **0.802** | **+0.118** | **7** | **4 / 3 / 0** |

Each cleaning pass raises system avg and shrinks oracle−system: the "headroom" was
partly broken-question noise, exactly as the eval-quality finding predicted. The
residual **+0.118 is real** (still > the ~0.05 metric-ceiling band).

## Decision — HippoRAG-2 gate: NOT green (confirmed, stronger)

The 4 retrieval_miss are all **single-hop coverage gaps** — direct factual
questions where the gold chunk was simply not retrieved (gold_overlap 0.09–0.24):
- prod_corpus-7  "Các bước đặt ống thông tiểu?" (catheter steps) — 0.24
- prod_corpus-12 "Triệu chứng cai nghiện thuốc phiện?" (opium withdrawal) — 0.10
- prod_corpus-15 "Yếu tố cần khai thác trong tiền sử bệnh?" (history-taking) — 0.09
- prod_corpus-23 "Nguyên nhân gây tiểu không tự chủ?" (incontinence causes) — 0.13

The 3 generation_miss have gold_overlap = 1.00 (retrieval worked, answer wrong) —
and **both multi-hop misses land here**, not in retrieval:
- prod_corpus-25       clinical vignette — gold packed, answer wrong
- prod_corpus_multihop-2 "Hạt tophi" gout crystal — gold packed @ rerank 0.58, uncited
- prod_corpus_multihop-6 osteoclast-surface receptors — gold packed, cited, still wrong

**Every multi-hop question that missed failed at GENERATION, not retrieval** —
multi-hop retrieval is already working. HippoRAG-2's entity-graph / multi-hop
traversal targets a failure mode that is not occurring. Building it would not close
the single-hop coverage misses (the largest bucket) nor the generation misses.

→ **Shelve HippoRAG-2.** The real, distributed +0.118 headroom wants:
1. **single-hop retrieval coverage** — 4/7 misses; chunking / embedding / rerank on
   direct factual questions (NOT a graph problem);
2. **answer generation on packed context** — 3/7 misses incl. both multi-hop;
   answer-prompt / model work (gold is right there, uncited or misused).

## Notes
- false_abstention went 2→0: the two v1 fa rows (opium withdrawal, tophi) re-scored
  as hallucinated/generation this run (fresh judge draws + system answered instead
  of hedging). No systematic abstention problem remains on the clean set.
- Fresh question sample vs v1 → compare aggregates, not per-qid.
- Flywheel NOT re-seeded from v2 (it re-probes ~the same questions as v1; appending
  would duplicate triplets). Seed stands at 107 (`data/finetune/citation_pairs.jsonl`).
- judge-noise pearson 0.940 — healthy.
