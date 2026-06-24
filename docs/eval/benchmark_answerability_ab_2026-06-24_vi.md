# Answerability-gate A/B — out-of-corpus refusal set (2026-06-24)

- Set: `data/eval/refusal_set.json` · n=15 · corpus reused (no re-ingest)
- Gate: `ANSWERABILITY_GATE_ENABLED` · margin `ANSWERABILITY_GRAY_MARGIN=0.13`
- Floor `RETRIEVAL_RELEVANCE_FLOOR=0.6` · rerank `local_cross_encoder`

| Metric | OFF (baseline) | ON (gate) |
|---|---|---|
| refusal_rate (clean abstain, ideal ↑) | 0.000 | 0.000 |
| hedged_cited_rate (soft ↓) | 0.667 | 0.400 |
| **hallucination_rate (DANGEROUS ↓)** | **0.267** | **0.600** |
| counts (abstain/hedged/halluc/empty) | 0/10/4/1 | 0/6/9/0 |

## Per-case (id · OFF→ON verdict · max rerank score)

| id | OFF | ON | max_score(ON) |
|---|---|---|---|
| tech-react-usestate | hallucinated | hallucinated | -1.0 |
| tech-docker-vs-vm | hedged_cited | hallucinated | -1.0 |
| tech-python-gil | hedged_cited | hallucinated | -1.0 |
| tech-tcp-handshake | empty | hallucinated | -1.0 |
| fab-zxylopraxin | hedged_cited | hedged_cited | -1.0 |
| fab-glor-syndrome | hallucinated | hallucinated | -1.0 |
| fab-quzium | hedged_cited | hallucinated | -1.0 |
| fab-vextra-clinic | hallucinated | hallucinated | -1.0 |
| fab-blorbocide | hedged_cited | hallucinated | -1.0 |
| fab-flebotrin | hedged_cited | hedged_cited | -1.0 |
| fab-karnak | hallucinated | hedged_cited | -1.0 |
| fab-zenithol | hedged_cited | hedged_cited | -1.0 |
| fab-xq7 | hedged_cited | hallucinated | -1.0 |
| fab-mendoza | hedged_cited | hedged_cited | -1.0 |
| fab-hypercardin | hedged_cited | hedged_cited | -1.0 |

## Conclusion — gate FAILS decision rule; deeper root cause found

**Decision:** keep `ANSWERABILITY_GATE_ENABLED=False`. The gate did NOT cut
hallucination (0.267→0.600 is run-to-run LLM noise, NOT the gate working).

**Why the gate never fired:** every case shows `max_score=-1.0` (no `rerank_score`
on packed_context), under BOTH `llm_chat` and `local_cross_encoder` backends. The
gate (and the pre-existing thin-context floor) only act when a cross-encoder
`rerank_score` is present — so neither can fire. `refusal_rate=0.000` in both arms:
the system NEVER cleanly abstains; it cites 8–22 distractors confidently on
out-of-corpus questions.

**Root cause (the real bug):** `LLMReranker.maybe_rerank` builds candidates keyed on
`item["id"]` and bails with `reason="no_candidate_ids"` when no item carries `id`
(`retrieval/reranker.py:71-81`). The assemble pipeline's candidates carry
`content_hash`, not `id` (`agent/context.py:_stage_retrieve` keys dedupe on
`content_hash`/`id`). So `ContextAssembler._stage_global_rerank` silently skips
(exceptions/`ok=False` are swallowed) → no `rerank_score` → floor/abstain/gate all
inert. The 19/06 benchmark's working abstain (refusal 0→7/15) implies scores were
present then → likely a regression from the `e4eb895` checkpoint (same commit that
broke `bootstrap_search(intent=)`).

**Secondary finding:** `gemini-2.5-flash-lite` frequently returns malformed JSON for
the agent's decide/reflect/HyDE steps (`json_response parse failed` — "Extra data" /
"Expecting ',' delimiter"), degrading the decision loop.

**Next (proposed, beyond original T5.9):** make assemble candidates carry `id`
(e.g. `id = content_hash`) OR have `maybe_rerank` fall back to `content_hash`; then
re-run this A/B — the gate can only be fairly evaluated once `rerank_score` reaches
packed_context.
