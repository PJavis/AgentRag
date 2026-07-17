# Generation-miss diagnostic — clean v2 set (2026-07-16)

The clean re-measure (`clean_remeasure_v2_2026-07-16.md`) leaves **7 real misses** on
the fully-filtered eval set, split **4 retrieval / 3 generation**. HippoRAG-2 is shelved
(both multi-hop misses are generation, not retrieval). This note pins each of the 3
`generation_miss` rows — gold WAS packed (gold_overlap = 1.00), the answer still scored
0 — to a specific lever, so the answer-side work is targeted rather than a blanket
"tune the prompt."

## The 3 rows

| qid | max_rerank of gold | answer behaviour | judged | lever |
|---|---|---|---|---|
| prod_corpus_multihop-2 | **0.58** (barely > floor 0.55) | **false-abstained** ("không đủ thông tin") | 0.0 | trust-low-rerank-context |
| prod_corpus-25 | 0.73 | terse wrong ("Denervation") | 0.2 | answer-completeness |
| prod_corpus_multihop-6 | 0.73 | plausible, **cited [1][3][10]**, still wrong | 0.0 | answer-synthesis / judge |

These are **three different failure modes**, not one prompt bug.

### 1. multihop-2 — false abstention on low-but-present gold (the actionable one)

Q: *"Loại tinh thể nào trong dịch khớp … 'Hạt tophi'?"* (which crystal → tophi = **urate**).
The gold passage was retrieved and packed, but at **rerank 0.58** — just above the 0.55
abstain floor. The answer LLM refused: *"Thông tin trong ngữ cảnh hiện tại không đủ …"*.

This is the mirror image of the abstain-safety win. The floor gate (0.55) correctly
keeps the passage IN the context, but the **answer prompt's own uncertainty instinct**
still discards a passage it deems weak. So a passage that survives the retrieval gate can
be ignored by generation — a second, hidden abstain threshold living in the LLM.

**Lever:** the answer prompt already says "if context is thin, say so." For context that
*passed the floor*, it should instead be instructed to **use the retrieved passages even
when they read as tangential** — the floor already made the keep/drop decision. This is a
prompt change (strengthen "ground in the provided passages; do not second-guess their
relevance") + possibly feeding rerank_score into the prompt so the model knows a passage
cleared the bar. Low risk, directly recoverable. This also connects to the retrieval
lever: a reranker retrain (flywheel) that pushes gold like this from 0.58 → 0.70 removes
the ambiguity at the source.

### 2. prod_corpus-25 — terse-wrong on a vignette

Q: a clinical vignette (nerve injury → muscle biopsy at 4 months). Gold packed, answer =
one word **"Denervation"**, scored 0.2. The answer is directionally near but far too thin
for a nugget/rubric judge — no supporting features, no grounding, no citation.

**Lever:** answer-completeness on vignette/"đặc điểm tốt nhất" questions — the verbose vs
concise heuristic (`_is_verbose_followup`) treats this as concise. Diagnostic-feature
questions want the structured multi-point answer even without a "chi tiết" keyword. Modest
prompt/routing tweak; smaller payoff than #1.

### 3. multihop-6 — plausible, cited, still judged 0

Q: *"Các thụ thể nào trên bề mặt tế bào hủy xương … cơ chế cận tiết …?"* Answer named
**vitronectin receptor (αvβ3) + calcitonin receptor**, cited [1][3][10], gold_overlap 1.00.
Scored 0.0 by both judges.

This one is **ambiguous between answer and judge**. Either the gold answer names different
receptors (RANK/RANKL/OPG axis for the paracrine osteoclast↔osteoblast signalling, vs the
adhesion/calcitonin receptors the system gave) — a genuine answer error — or the judge is
penalising a partially-correct receptor list. **Action:** eyeball the gold_contexts for
this row before touching anything; if the gold is the RANK/RANKL axis, it is a real
retrieval-of-the-wrong-fact-within-a-packed-chunk error (the chunk was there, the model
pulled the wrong sentence), which is answer-synthesis, not prompt length.

## Takeaway / ordering

1. **multihop-2 (trust-low-rerank-context)** is the clean, recoverable one and it
   **double-benefits from the reranker retrain** already on deck — fix it there first, then
   re-check whether the false-abstention persists at the higher rerank score.
2. **prod_corpus-25** is a small answer-completeness/routing tweak.
3. **multihop-6** needs a human look at the gold before it is even classified as answer vs
   judge — do not prompt-tune on it blind.

Net: the 3-row generation bucket is **1 real prompt lever + 1 minor routing tweak + 1
needs-eyeballing**, and the biggest of them is absorbed by the reranker-retrain path. No
separate large answer-model effort is justified by this evidence.
