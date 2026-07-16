# Rerank-before-trim + retrieval breadth — 2026-07-16

Closes the agent-retrieval-path dig that followed the clean re-measure
(`clean_remeasure_v2_2026-07-16.md`). That run left 7 misses; a per-row probe of
the retrieval path found the dominant failure was neither the reranker nor the
answer prompt.

## Root cause

Gold passages for factual questions rank **deep in the hybrid fusion (rank 8–17)**,
but two things hid them from the answer node:

1. **Breadth too shallow.** `AGENT_TOOL_TOP_K=5–8` and `RETRIEVAL_RAW_QUERY_TOP_K=8`
   pulled only ~4–8 fused hits, so deep gold never entered the candidate pool.
2. **Membership decided pre-rerank.** `context.assemble` ran
   `_stage_rank_trim` (a weak rrf + token-overlap + source-boost heuristic, capped
   to a 12k-token budget) to pick the packed set, and the cross-encoder rerank
   only **reordered the survivors**. So simply widening recall let high-rrf
   distractors win packed slots and crowd gold out — and flooded the answer.

Evidence (raw-question retrieval, pre-rerank pool):

| miss | gold in pool? | reranker on gold | agent packed gold? |
|---|---|---|---|
| prod_corpus-7  | yes @ rank 8  | 0.72 (clears floor) | no |
| prod_corpus-12 | yes @ rank 13 | 0.67 | no |
| multihop-2     | yes @ rank 7  | 0.51–0.58 | no |
| prod_corpus-15 | **no** | — | no (true recall gap) |
| prod_corpus-23 | **no** | — | no (true recall gap) |

## Fix

1. **Widen recall:** `AGENT_TOOL_TOP_K 5→30`, `RETRIEVAL_RAW_QUERY_TOP_K 8→50`
   (config.py) — deep gold now enters the pool.
2. **Rerank before trim** (context.assemble): rerank the FULL deduped pool, then
   trim/pack the top-K **by rerank_score** (the cross-encoder decides membership;
   the weak rrf heuristic is now only the rerank-disabled fallback). Keeps the
   existing token-budget + per-bucket diversity + structmem-inclusion behaviour.

A naive breadth bump ALONE (step 1 without step 2) recovered 3 misses but
**regressed 2** previously-correct rows (prod_corpus-0 1.00→0.20, multihop-5
0.58→0.20) because the wider pool flooded the rrf-keyed trim with distractors.
Step 2 is what makes the breadth safe.

## Result — clean win, no regressions

Eval set `c2_evalset_n40_clean_v2.jsonl` (n=42, CRAG off, gemini/deepseek judges):

| metric | before | after | 
|---|---|---|
| system avg | 0.802 | **0.884** (+0.082) |
| oracle − system | +0.121 | **+0.044** (at the metric ceiling) |
| misses | 7 | **3** |

- **Recovered 4:** prod_corpus-7, -12, -25, multihop-2. **Regressions: 0.**
- Remaining 3: prod_corpus-15, -23 (**true recall gaps** — gold not retrievable at
  all → chunking/embedding, a separate lever), multihop-6 (needs a human look at
  the gold: answer-vs-judge).
- oracle−system +0.044 means perfect retrieval + a strong generator now barely
  beats the live system on clean questions — the system is at the eval ceiling.

## Safety gate — OOC abstention unchanged

Refusal A/B on the 15-question out-of-corpus set with the wider retrieval, prod
config (answerability gate OFF): **15/15 clean abstain, 0 hallucinated** — identical
to the pre-breadth baseline. The wider net does not leak: raw OOC queries still
rerank ~0.50 (below the 0.55 floor) → context stays thin → deterministic refusal.
So the recall increase is safe.

## Verdict
Ship both changes. The retrieval-coverage bucket that survived every earlier
re-measure was mostly an answer-context-membership bug, not a model-quality gap —
no reranker retrain or answer-model work was needed. Next real lever is the 2 true
recall gaps (chunking/embedding), which is a different investigation.
