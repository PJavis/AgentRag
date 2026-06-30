# VITAL — Eval Home-Run Plan (next campaign, 2026-06-30)

Run these at home, in order. Goal: make the correctness number **trustworthy** (independent judge),
then learn **real-doc** behavior (`data/originals`), then test the **proven** fine-tune lever where it
can actually matter.

**Current state (what's already done):**
- Eval-fidelity ruler built; **eval-is-the-ceiling CONFIRMED** at clean n=50 (`oracle−system +0.046 < 0.05`).
- Trustworthy clean correctness ≈ **0.888** (`docs/eval/eval_fidelity_probe_prod_v3_2026-06-26.md`).
- Abstain false-abstention fixed; hang-budget added; embedding fine-tune **proven** (recall@10 +0.20).
- **Two open gaps this plan closes:** (1) the v3 judge was **all-DeepSeek** (same provider as the
  answer model → self-preference risk, pearson 0.730); (2) every run so far is on `vn_bkai`/`vn_legal`
  **residue**, not the real `data/originals` corpus.

**Prereqs:** stack up (`pg/es/valkey/ollama`), GPU for fine-tune/re-ingest, and **one** of:
`ANTHROPIC_API_KEY` (Phase 1 Option A), a **paid** gemini key (Option B), or accept the DeepSeek-judge
caveat and skip to Phase 2.

> ⚠️ **Scale note:** `data/originals` currently holds **7 PDFs**, not the 114 the old docs assumed.
> The real-corpus eval is small-n — directional. Add more real docs there for a firmer read.

---

## Phase 1 — Independent judge (DO FIRST — unblocks trust of every number)

The v3 0.888 isn't quotable until a model that does **not** share a provider with the answer model
(`deepseek-v4-flash`) scores it. Pick one option.

### Option A (recommended) — Claude judge (also closes the deferred "Claude in eval slots")

1. **Wire the `anthropic` provider** — `src/agentrag/agent/llm.py`:
   - In the auto-derive block (~line 92, beside the `deepseek`/`gpt-` branches) add:
     ```python
     elif mlow.startswith("claude"):
         provider_override = "anthropic"
     ```
   - In `_resolve_backend_for` add a branch:
     ```python
     if provider == "anthropic":
         if not settings.ANTHROPIC_API_KEY:
             raise ValueError("ANTHROPIC_API_KEY required for Claude-routed task")
         # Anthropic's OpenAI-compatible endpoint — works with the existing AsyncOpenAI
         # client for chat-completions JSON (judge only; no adaptive-thinking on this path).
         return (model or "claude-haiku-4-5", "https://api.anthropic.com/v1/", settings.ANTHROPIC_API_KEY)
     ```
   - `src/agentrag/config.py`: add `ANTHROPIC_API_KEY: str | None = None`.
   - (Optional, cleaner) add a config-validation guard rejecting a non-`claude-*` model under the
     anthropic provider, mirroring the local-reranker guard.
2. **`.env`** — add the key and point the eval slots so the PRIMARY judge is independent of the
   answer model, and the noise floor is cross-provider:
   ```bash
   ANTHROPIC_API_KEY=sk-ant-...
   # eval_judge (primary score) = Claude (independent of the deepseek answer model -> no self-preference)
   # eval_judge2 (noise floor)  = deepseek -> pearson now measures a REAL cross-provider agreement
   LLM_TASK_MODEL_MAP={..., "oracle_gen":"deepseek-v4-pro","gold_gen":"deepseek-v4-pro",
                       "eval_judge":"claude-haiku-4-5","eval_judge2":"deepseek-v4-pro"}
   ```
   (Use `claude-sonnet-4-6` for a stronger judge if budget allows.)
3. **Smoke** one judge call:
   `uv run python -c "import asyncio;from src.agentrag.services.llm_gateway import LLMGateway;
   g=LLMGateway();print(asyncio.run(g.json_response(system_prompt='return json',user_prompt='{\"ok\":1}',task='eval_judge')))"`

### Option B — paid gemini
Set `eval_judge=gemini-2.5-pro` (independent of deepseek answer), `eval_judge2=deepseek-v4-pro`; no
code change. Requires a paid gemini key (free tier serves `pro` at limit:0).

### Verify (the point of Phase 1)
Re-probe the existing v3 set with the new judge:
```bash
uv run python scripts/eval/oracle_probe.py \
  --eval-set data/eval/prod_corpus_evalset_v3.jsonl --n 50 --retries 3 \
  --out docs/eval/probe_v3_indep-judge.md
```
- **`judge-noise pearson`** is now a real cross-provider number (expect lower than the 0.730
  same-family figure — that's honest, not worse).
- If **system avg shifts notably from 0.888**, the DeepSeek self-preference was inflating it → quote
  the new number. If it holds ~0.888, the ceiling story is robust to judge choice.

---

## Phase 2 — Real prod-corpus A/B (the long-deferred test)

Everything so far confirms the *judge/ceiling* story on `vn` residue. This tells *real-doc* behavior.

1. **Ingest the real corpus** (CR+RAPTOR **OFF** first — set the CR/RAPTOR env flags off, see
   `scripts/eval/run_ablation.py` for the exact flag names):
   ```bash
   uv run python -c "import asyncio; from src.agentrag.ingestion.pipeline import ingest_folder; \
     asyncio.run(ingest_folder('data/originals', graph_ingest_mode='sync'))"
   ```
2. **Build an eval set over the REAL docs** (now `Segment.content` = `data/originals`):
   ```bash
   uv run python scripts/eval/build_prod_evalset.py --n 40 --out data/eval/prod_real.jsonl
   ```
   (n capped by the chunk pool — 7 PDFs is small; inspect a few rows for grounded gold.)
3. **Probe with the Phase-1 independent judge:**
   ```bash
   uv run python scripts/eval/oracle_probe.py \
     --eval-set data/eval/prod_real.jsonl --n 40 --retries 3 \
     --out docs/eval/probe_real_CRoff.md
   ```
4. **CR+RAPTOR A/B:** re-ingest the same folder with CR+RAPTOR **ON**, re-build the eval set (or
   reuse — the eval set's gold chunks are corpus-text, stable), re-probe →
   `docs/eval/probe_real_CRon.md`. Compare precision/recall/correctness → **settles the deferred
   CR+RAPTOR decision on real docs** (the n=80 synthetic result said OFF; this is the real-shape test).

**Decision gate:** `oracle − system` on real docs. `< 0.05` → system at the eval ceiling on real docs
too (metric-bound, like the residue). Larger → **real** retrieval/generation headroom → actionable.

---

## Phase 3 — Promote the proven fine-tuned embedding (fold into Phase 2's re-ingest)

The embedding fine-tune is proven (recall@10 **+0.20**, commit `1b28bac`) but the gain didn't propagate
to correctness on the *easy* residue. The real corpus (harder retrieval) is where it can.

1. **(Re)train if needed** — `make train-embed` (reads `data/finetune/embed_triplets.jsonl` from
   `mine_finetune_pairs.py` → `models/agentrag-embed-v1`).
2. **Promote** — `make serve-embed` (TEI on :8080), then point `EMBEDDING_PROVIDER`/`EMBEDDING_MODEL`/
   `EMBEDDING_BASE_URL` at the served fine-tuned model.
3. **Re-ingest** `data/originals` with the new embedding, **re-probe** → compare correctness vs the
   baseline-embedding real run (Phase 2).

**Decision:** correctness moves on real docs → fine-tune is worth promoting; flat → confirms retrieval
isn't the correctness bottleneck even on real docs (consistent with `oracle−system ≈ noise`).

---

## Phase 4 (optional) — harden + calibrate

- **Multi-chunk synth-Q (v1.1):** the single-chunk gen makes most Qs answerable from one chunk
  (scores stack at 1.0). Add a multi-hop subset in `build_prod_evalset.py` to spread the score mass
  and discriminate harder.
- **Human calibration:** hand-label 25–30 Q from the real set; require judge↔human agreement
  **κ ≥ 0.7** before quoting any absolute correctness figure.

---

## Decision gates (one screen)

| Phase | Run | Pass / read |
|---|---|---|
| 1 | re-probe v3 with independent judge | real cross-provider pearson; 0.888 holds or corrects |
| 2 | real-corpus probe (CR off) | `oracle−system <0.05` → ceiling on real docs too |
| 2 | CR off vs on | precision/recall/correctness Δ → keep/flip CR+RAPTOR on real shape |
| 3 | fine-tuned embed re-probe | correctness Δ on real docs → promote or shelve |
| 4 | human calibration | κ ≥ 0.7 → numbers are quotable |

## Quick path (low on keys/time)
Skip Phase 1; run **Phase 2 with the DeepSeek judge** (carry the same-provider caveat in the report) to
learn real-doc behavior now. Add the independent judge (Phase 1) when a key is available, then re-probe.

## Files / refs
Probe + build: `scripts/eval/oracle_probe.py`, `scripts/eval/build_prod_evalset.py`. Judge:
`src/agentrag/eval/correctness_judge.py`. Ingest: `src/agentrag/ingestion/pipeline.py::ingest_folder`.
CR+RAPTOR flags + A/B harness: `scripts/eval/run_ablation.py`. Background:
`docs/eval/eval_fidelity_probe_prod_v3_2026-06-26.md`, `docs/eval/finetune_gate_2026-06-27.md`,
`docs/INSTRUCTION-abstain-thin-context-2026-06-16.md`.
