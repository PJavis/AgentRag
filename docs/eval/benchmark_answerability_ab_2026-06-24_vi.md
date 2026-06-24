# Safety re-validation — out-of-corpus refusal set (2026-06-24)

Set: `data/eval/refusal_set.json` · n=15 fabricated/out-of-corpus questions
(drugs, diseases, software topics — none in any ingested corpus). Corpus reused
(no re-ingest). Judge-free: `classify_refusal` is rule-based (uncertainty marker +
citation presence). Agent = `GraphAgentService`, LLM = gemini-2.5-pro/flash-lite,
embed = ollama nomic-embed-text.

This run started as an A/B of the new answerability gate (P1 Task 5) but uncovered
two upstream regressions; the table below is the corrected picture.

## Headline numbers

| Config | refusal_rate ↑ | hedged_cited ↓ | hallucination ↓ | distractor citations |
|---|---|---|---|---|
| **A. broken rerank** (`llm_chat`, no scores) | 0.000 | 0.667 | 0.267 | **8–22 per answer** |
| **B. rerank fixed, prompt-abstain only** (floor-gate OFF) | 0.267 | 0.000 | 0.667 | 8–15 |
| **C. rerank fixed + hard floor-gate ON** | **0.400** | 0.000 | 0.533 | **0** |

(`ANSWERABILITY_GATE_ENABLED` gray-band gate left OFF throughout — see §3.)

## 1. Two regressions made the abstain safety totally inert (config A)

Every floor-based safety (thin-context abstain, relevance-floor gate, the new
answerability gate) reads `rerank_score` off `packed_context`. On the current
branch that score was **never present**, for two stacked reasons:

1. **`no_candidate_ids`** — `LLMReranker.maybe_rerank` keyed candidates solely on
   `item["id"]` and bailed when absent. The `ContextAssembler` pipeline's candidates
   carry `content_hash` (the dedupe key), not `id` → global rerank silently skipped
   for **every query** (exceptions swallowed) → no `rerank_score`. **Fixed**
   (`reranker.py`: backfill `id = content_hash`; TDD `test_reranker_id_fallback.py`).
2. **Only `local_cross_encoder` emits scores** — the `llm_chat` rerank path (gemini,
   the `.env` default `RETRIEVAL_RERANK_BACKEND=llm_chat`) reorders candidates but
   attaches **no** `rerank_score`. So even with fix #1, the default config can't drive
   any floor logic. To get abstain, the backend must be `local_cross_encoder`.
3. **Model-name trap** — setting `RETRIEVAL_RERANK_BACKEND=local_cross_encoder` while
   `.env` still has `RETRIEVAL_RERANK_MODEL=gemini-2.5-flash-lite` makes the
   CrossEncoder try to load "gemini-2.5-flash-lite" from HuggingFace → `OSError` →
   swallowed → inert. Must also set `RETRIEVAL_RERANK_MODEL=dengcao/bge-reranker-v2-m3`.

Result in config A: `refusal_rate=0.000` — the system cited **8–22 distractor
passages** for fabricated drugs/diseases. Since the 19/06 report showed abstain
working (0→7/15), scores were present then → these are almost certainly regressions
from the `e4eb895` "checkpoint" commit (same commit that broke `bootstrap_search(intent=)`).

## 2. With rerank fixed, floor-based abstain revives

- **Config B** (prompt-abstain only): refusal `0.000 → 0.267`. The id-fix alone
  revived 4 clean refusals. But `hallucination=0.667` — out-of-corpus questions score
  `max≈0.50` (< floor 0.6), so thin-context fires, yet **gemini-2.5-pro ignores the
  refuse prompt 2/3 of the time** and answers confidently with 8–15 distractor cites.
- **Config C** (hard `RETRIEVAL_RELEVANCE_GATE_ENABLED=true`): drops sub-floor context
  **before** the answer node. Best refusal `0.400`, and crucially **distractor
  citations → 0 and hedged_cited → 0** — the model cannot cite what it cannot see.
  This directly serves the medical-safety priority (never cite a fabricated source).

## 3. The answerability gray-band gate stays OFF

`ANSWERABILITY_GATE_ENABLED` targets the band `[floor, floor+0.13)` = `[0.60, 0.73)`.
But every out-of-corpus question scores `≈0.50` — **below** the floor, handled by
thin-context, never in the gray band. So the gate engages on nothing for this failure
mode (gate-ON ≈ gate-OFF modulo LLM noise). Built correctly, but not the lever here.
Decision: keep `ANSWERABILITY_GATE_ENABLED=False`.

## 4. The residual: parametric hallucination

Config C still leaves `hallucination=0.533` — the model answers "React useState",
"TCP handshake", etc. **from its own training even when given empty context**. Prompt
text cannot reliably stop this. Per-case compliance is also noisy (React abstains in
C but hallucinates in B; GIL the reverse).

## Recommendations (priority order)

1. **Ship the rerank id-fix** (done, `af29043`) — without it ALL floor safety is dead
   AND retrieval precision is degraded system-wide (rerank never reorders).
2. **Default `RETRIEVAL_RERANK_BACKEND=local_cross_encoder`** (with
   `…_MODEL=dengcao/bge-reranker-v2-m3`) in `.env` — the only backend that powers
   abstain. Add a config-validation guard rejecting an API model-name under the local
   backend (kills the trap in §1.3).
3. **Enable the hard relevance-floor gate** and, when it empties the context,
   **short-circuit to a deterministic refusal without calling the answer LLM** — that
   is what finally kills the §4 parametric hallucination. Small, high-value change.
4. **Re-run the report's "relevance-gate counterproductive" A/B** — that verdict was
   measured on the broken-rerank system; config C contradicts it.
5. Keep the gray-band `ANSWERABILITY_GATE_ENABLED` OFF for now (§3).
6. Investigate `gemini-2.5-flash-lite` malformed-JSON on the agent decide/HyDE steps
   (`json_response parse failed` storms) — degrades the decision loop and inflates
   latency; a stricter JSON-extraction or a sturdier decide model would help.

Raw: `docs/eval/refusal_singlearm_gate0_floorgate1.json` (config C).
ARM A/B verdicts in the session run logs.
