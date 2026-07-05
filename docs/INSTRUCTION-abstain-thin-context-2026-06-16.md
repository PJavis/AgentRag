# Instruction — Abstain-on-thin-context A/B (2026-06-16)

Branch `feat/ragas-langfuse-reranker` (HEAD `d53491e`). Follow-up to the relevance-gate A/B (`b6c0c3e`) which **disproved** the retrieval-pruning gate.

## Context — what we learned

The relevance **gate** (`RETRIEVAL_RELEVANCE_GATE_ENABLED`) was tested and is **counterproductive — keep it OFF.** Dropping low-relevance chunks before the answer made the model fall back to parametric memory → `hallucination_rate` went UP (0.467→0.600) and in-corpus recall down (−0.030). Root cause: removing context ≠ silence; the model just fabricates from its weights.

The **corrected** fix is at the answer/prompt layer, shipped in `d53491e`:
**Abstain-on-thin-context** (`ANSWER_ABSTAIN_ON_THIN_CONTEXT`, default OFF):
- when the best packed chunk's `rerank_score < RETRIEVAL_RELEVANCE_FLOOR` → **KEEP** the context, but the answer prompt flips to: *"no relevant info → say so in one sentence; do NOT use background knowledge; do NOT cite."*
- on the resulting clean abstention → strip the distractor citations (`ground` → `citations=[]`).

Difference from the gate: context is **kept** (no recall loss) but the prompt **forbids** answering from memory → should produce a real abstention, not a hallucination.

## Both flags stay OFF in prod until this A/B proves the abstain version.

---

## Run the A/B

Requires reranking on (`RETRIEVAL_RERANK_ENABLED=true`, backend `local_cross_encoder`) — your `.env` already has it. Keep the relevance GATE off.

```bash
# Config A — OFF (baseline):
#   ANSWER_ABSTAIN_ON_THIN_CONTEXT=false
# Config B — ON:
#   ANSWER_ABSTAIN_ON_THIN_CONTEXT=true
#   RETRIEVAL_RELEVANCE_GATE_ENABLED=false   (keep the disproven gate OFF)

uv run python scripts/eval/run_benchmark.py --suite both --n 20 \
  --group-size 4 --refusal-set data/eval/refusal_set.json
```
Run both configs on the same ingested corpus (`--skip-ingest` for the 2nd so only the flag differs). Save each report.

## Win condition (vs the gate, which failed all of these)

| Metric | Target with abstain ON |
|---|---|
| `refusal_rate` (clean abstain) | **moves off 0.0** ↑ |
| `hallucination_rate` (DANGEROUS) | **goes DOWN** (gate pushed it to 0.600) |
| `hedged_cited_rate` | DOWN (converts to clean abstain, not to hallucinate) |
| in-corpus `answer_correctness` | **flat** (context kept) |
| in-corpus `contextual_recall` | **flat** (context kept) |

## If it doesn't work

- `refusal_rate` still 0 / `hallucination_rate` not down → the prompt instruction isn't strong enough; harden it (make the abstain prompt more forceful, or lower the answer model temperature on thin context).
- in-corpus `correctness`/`recall` drops → `RETRIEVAL_RELEVANCE_FLOOR` too high (it's shared with the gate); try 0.25.
- Tell me the numbers and I'll tune the floor or the prompt.

## Decision

- If B beats A on the win condition → enable `ANSWER_ABSTAIN_ON_THIN_CONTEXT=true` in prod, keep the gate OFF.
- If B doesn't help → leave both OFF; the abstain problem needs a different approach (e.g. a dedicated refusal classifier), and we discuss.

Prior reports: `docs/eval/benchmark_gate_ab_2026-06-16_vi.md` (gate disproved), `docs/eval/benchmark_grouped_2026-06-15_vi.md`. Broader next-steps: `docs/NEXT-STEPS-2026-06-16.md`.

---

## UPDATE 2026-06-26 — calibrated + the flaky-abstention fix (this is now the SHIPPED state)

`ANSWER_ABSTAIN_ON_THIN_CONTEXT` is **ON** (default). The prod-corpus eval-fidelity probe
(`docs/eval/eval_fidelity_probe_prod_2026-06-26.md`) surfaced the first concrete bug: 3 questions
the live system **refused despite the gold chunk being retrieved at rank 0**. Root-caused
(systematic-debugging) to *this* gate — not retrieval, not generation:

- `_is_thin_context` abstains when `max(rerank_score) < RETRIEVAL_RELEVANCE_FLOOR`. The bge
  cross-encoder scores **paraphrased-relevant** VN chunks ~**0.61** (and every other chunk a flat
  0.5 — it only meaningfully scores the top hit). The agent's decide-step query-rewrites make that
  max **wobble around the old 0.6 floor** → flip answer↔abstain run-to-run.

**Shipped fixes:**
1. **`RETRIEVAL_RELEVANCE_FLOOR` 0.6 → 0.55** (`f5dfe76`). Mid-band: OOC ~0.50 | floor | relevant
   ~0.61, ~0.05 margin both sides. **Supersedes the old "if recall drops, try 0.25" note above** —
   0.25 is below the bge output range (gate inert); 0.55 is the calibrated value. Re-validate OOC
   abstention if you re-tune it.
2. **`RETRIEVAL_INCLUDE_RAW_QUERY=true`** (`c706d6c`) — `ContextAssembler` always merges a
   plain-`hybrid` retrieval on the RAW question into the rerank candidate pool. The agent's
   rewrites (hybrid_kg + variants) can retrieve worse chunks than the raw question; this guarantees
   the rewrites only ADD, never drop the best chunk below the floor. `RETRIEVAL_RAW_QUERY_TOP_K=8`.
3. **Deterministic query-rewrite** (temp 0) so the same question doesn't flip.

**Validated** — the abstain fix eliminated the false-abstentions: **0 hard misses (was 3)**, row18 +
row21 now answer correctly, **OOC safety holds** (genuinely out-of-corpus questions still abstain at
0.55 — their raw hits also score ~0.50 < floor, so the raw-query injection does NOT loosen OOC). The
directional v2 lift (n=20) read system 0.842→0.950 / gap +0.019, but the **clean n=50 run**
(`docs/eval/eval_fidelity_probe_prod_v3_2026-06-26.md`, 0 skips) is the trustworthy number: system
**0.888**, oracle−system **+0.046 (<0.05)** — eval-is-the-ceiling holds, no false-abstention
regression. (v2's 0.950 was inflated by 10/30 gemini-503 skips = easy-Q selection bias.)

**Still tunable / open:** the per-call `LLM_REQUEST_TIMEOUT_S=60` + total `AGENT_TOTAL_TIMEOUT_S=90`
bound runaway `agent.chat` under gemini 503 storms (graceful "busy" response). The judge ran
all-DeepSeek (gemini free-tier `pro` at limit:0) → weak independence (pearson 0.730); an
**independent judge** (paid gemini, or wire the `anthropic` provider + `ANTHROPIC_API_KEY`) is needed
before quoting a correctness figure with confidence.
