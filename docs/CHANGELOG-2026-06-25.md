# Changelog & Results — 2026-06-25 (T6: WS1–5 ablation)

Closes **T6**, the last open task of the P0+P1 plan
(`docs/superpowers/plans/2026-06-24-vital-improvement-p0-p1.md`). T1–T5 and T7
shipped earlier (see `CHANGELOG-2026-06-24.md`); with T6 done, **P0+P1 is fully
complete**. All work ran on the real production stack — **DeepSeek + local Ollama**
(`decide/classify/domain_router/followup` → `llama3.2:3b`; `plan/extract/answer/…`
→ `deepseek-v4-*`), embeddings via TEI (`bge-m3`), rerank `local_cross_encoder`
(`bge-reranker-v2-m3`). The 06-24 home-run guide's `.env` block is Gemini-centric —
that doc drift is a known follow-up, not yet fixed.

---

## 1. What shipped

### 🔬 Ablation harness — two correctness/feasibility fixes
- **`62ef610` True-baseline isolation + index-shape grouping.**
  - `build_env` now force-sets all six WS flags **off** before applying a row's
    overrides. The live `.env` has `CONTEXTUAL_RETRIEVAL_ENABLED=true` +
    `RAPTOR_ENABLED=true`, and env vars override `.env` in pydantic — so the child
    process silently ran "baseline" as **CR+RAPTOR**, making `cr`/`cr_raptor`
    identical to it and destroying per-WS attribution. Baseline is now genuinely
    all-off.
  - Configs are **grouped by index-shape** `(CR, RAPTOR)`: the first member of each
    shape re-ingests; query-time-only siblings (CRAG / MULTIHOP / ADAPTIVE_ROUTING /
    SEMANTIC_CACHE) reuse that index via `run_benchmark --skip-ingest`. The dominant
    cost (re-ingest + StructMem graph-extraction, ~2 h / 100 docs) drops from
    one-per-row to **one-per-shape: 8 configs → 3 ingests** (~16 h → ~5 h).
- **`7cab933` `--group-size` passthrough.** `run_ablation` forwards `--group-size`
  to the ingesting child. The default `group_size=0` ingests one gold context per
  doc (1 chunk/doc), which leaves **RAPTOR (needs ≥8 chunks/doc) and CR near-inert**
  — so the matrix understated them; a `≥8` re-bench gives a fair test.

### 📊 Results (vn, n=10 → 20 questions/config, judge = DeepSeek)
- **`2df53b7` gs=0 matrix** (8 configs) → `docs/eval/benchmark_ablation_2026-06-24.md`.
- **`156f882` gs=8 re-bench** (baseline/cr/cr_raptor, RAPTOR actually builds) →
  `docs/eval/benchmark_ablation_2026-06-25-gs8.md`.
- **`0e304cc`** records the verdict inline in `.env.example`.
- Raw per-config JSON backed up to `data/eval/matrix_2026-06-24/` (`data/eval`
  is gitignored).

---

## 2. Verdict

Noise floor (judge + sampling, single-run n=20) ≈ **±0.05–0.07**; treat smaller
deltas as noise.

| WS feature | effect | decision |
|---|---|---|
| **CR + RAPTOR** | **contextual_precision ~+0.04 in BOTH corpus shapes** (consistent direction → trustworthy); faithfulness preserved (0.935–0.958); no latency cost | **Keep ON** in live `.env` (validated). `.env.example` stays OFF (cheap fresh deploy; RAPTOR ingest is expensive) with the ablation cited. |
| CR alone | precision flat; trades correctness for faithfulness | OFF |
| CRAG | no precision/correctness gain; **drops faithfulness 0.989 → 0.814** | OFF |
| Adaptive fast-path | no latency win (p50 stays ~20 s) | OFF |
| Semantic cache | can't hit on unique-query benchmark — **untested, not disproven** | OFF (needs a production-traffic A/B) |
| Multi-hop | flat (single-fact questions need no hops) | OFF |

Key reads:
- **CR+RAPTOR precision win is the one robust signal** — same direction across both
  the gs=0 and gs=8 runs. Correctness flipped sign between runs (+0.07 vs −0.09) →
  noise, no real effect. The win is modest and single-run; an **n≥40 confirmation**
  is the honest next step before treating it as settled.
- **Latency p50 ≈ 20–25 s is answer-LLM-bound** (DeepSeek answer-gen ~18–20 s/Q).
  The <10 s roadmap target is **unreachable by any WS flag** — it belongs to a
  separate model/serving track, not the retrieval features.
- The roadmap's "precision 0.699 → 0.80" goal is moot: the clean baseline already
  measures **0.84–0.86**. `full` (CR+RAPTOR + all query-time flags) reached
  **correctness 0.900 / precision 0.927** at gs=0, but with a faithfulness cost from
  CRAG — so it is not the recommended config.

---

## 3. State

- **Branch `feat/ragas-langfuse-reranker` is merge-ready** per the spec appendix
  (P0+P1 landed, re-benchmark green). Continuing to P2 was chosen over merging.
- **P2 not started** (deferred): Langfuse online (scaffolded, flag OFF), eval
  fidelity (cost/latency forced-cloud distortion — hit during this run), CI
  (`ci.yml` exists, needs first-PR validation), feedback capture (`FeedbackButtons.tsx`
  exists → needs persistence → DPO dataset), ops hardening. Each is its own
  spec→plan→implement cycle.

---

## 4. Known follow-ups
- **n≥40 confirmation** of the CR+RAPTOR precision win (current is single-run n=20).
- **Eval fidelity**: env-gate the benchmark so cost/latency use the internal models
  (the forced-cloud distortion surfaced again here).
- **Faster ingest** for ablation: StructMem graph-extraction dominates (~2 h/100 docs).
- **Push toward 0.9 correctness**: bump `answer` flash→pro, proper RAPTOR ingest
  (group_size), and the VN-medical reranker fine-tune (`docs/FINETUNE_STRATEGY.md`)
  — the structural ceiling-raiser.
- **Doc drift**: home-run guide `.env` block is Gemini-centric; real stack is DeepSeek.
