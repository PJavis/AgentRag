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
