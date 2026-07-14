# VITAL — Home-Run: Miss Buckets + CRAG A/B + Flywheel Seed (2026-07-14)

Run these at home, in order. Goal: split the **+0.088 real-corpus headroom** (~5/40 misses,
`docs/eval/c2_probe_n40_gemini-judge.md`) into named failure classes, decide the
`CRAG_ENABLED` flag with an A/B, and seed the citation-reward reranker training file.

This executes **Task 4** of `docs/superpowers/plans/2026-07-14-miss-buckets-crag-flywheel.md`.
Tasks 1–3 (the tooling) are already merged on this branch and tested (84/84 eval tests):

- `oracle_probe.py --rows-out x.jsonl` — per-question dump (answers, judge scores, packed
  passages + rerank scores, inline `[n]` citations, refusal class, tool queries).
- `scripts/eval/report_miss_buckets.py` — classifies each miss (sys < 0.5) into
  `false_abstention` / `retrieval_miss` / `generation_miss` + judge-gap flags.
- `scripts/eval/mine_citation_pairs.py` — RMM flywheel: cited packed passage = positive,
  hardest uncited = hard negative, only from rows with sys ≥ 0.75. Output shape matches
  `scripts/finetune_reranker.py` / `finetune_embedding.py` input.

**Why each bucket matters (pre-registered decisions):**

| bucket | meaning | action it green-lights |
|---|---|---|
| `retrieval_miss` | gold chunk never reached the answer LLM | HippoRAG-2 StructMem/graph plan (majority → write that spec) |
| `false_abstention` | refused an answerable question | floor/gate tuning (NOT graph work) |
| `generation_miss` | gold was packed, answer still wrong | answer prompt/model work |

---

## Phase 0 — Prereqs (all four failed on the WSL box on 2026-07-14)

1. **Stack up:** Elasticsearch + Postgres + Redis + Ollama, TEI embedding on :8080
   (`make serve-embed` — e5-FT `agentrag-embed-v1`, 768-dim), local reranker GPU free.
   ```bash
   curl -s localhost:9200/_cluster/health | head -c 120   # green/yellow
   curl -s localhost:8080/health                           # ok
   ```
2. **Keys in `.env`:** `DEEPSEEK_API_KEY` (answer/oracle/judge2) + **paid** `GEMINI_API_KEY`
   (primary judge). Independent-judge map (same as the 2026-07-13 run):
   ```bash
   LLM_TASK_MODEL_MAP={..., "oracle_gen":"deepseek-v4-pro","gold_gen":"deepseek-v4-pro",
                       "eval_judge":"gemini-2.5-pro","eval_judge2":"deepseek-v4-pro"}
   RETRIEVAL_RELEVANCE_FLOOR=0.55
   RETRIEVAL_RERANK_BACKEND=local_cross_encoder
   ```
   (Claude judge also works now — `eval_judge=claude-*` + `ANTHROPIC_API_KEY`.)
3. **Eval set:** `data/eval/c2_evalset_n40.jsonl` (gitignored) must exist on the rig that ran
   the 2026-07-13 probes. **Prefer reusing it** — numbers stay comparable to
   `c2_probe_n40_gemini-judge.md`. Only if lost, rebuild (new sample → non-comparable, say so
   in the report):
   ```bash
   uv run python scripts/eval/build_prod_evalset.py --n 40 --multihop 12 \
     --out data/eval/c2_evalset_n40.jsonl
   ```
   ⚠️ Never point probes at `prod_corpus_evalset*.jsonl` — residue-corpus questions, score
   sys=0.00 against the real corpus (validity rule: an eval set is only valid against the
   corpus snapshot it was generated from).

## Phase 1 — Baseline arm (CRAG OFF) + bucket report

```bash
CRAG_ENABLED=false uv run python scripts/eval/oracle_probe.py \
  --eval-set data/eval/c2_evalset_n40.jsonl --n 40 --retries 3 \
  --rows-out docs/eval/rows_c2_n40_crag_off.jsonl \
  --out docs/eval/c2_probe_crag_off_2026-07-14.md

uv run python scripts/eval/report_miss_buckets.py \
  --rows docs/eval/rows_c2_n40_crag_off.jsonl \
  --out docs/eval/miss_buckets_2026-07-14.md --label c2_evalset_n40-crag-off
```

Expect ~30–60 s/question (run under `nohup`, tail the log). The bucket split printed by the
second command **is the campaign's main deliverable** — it decides where the next month of
retrieval work goes (table above).

## Phase 2 — CRAG ON arm + abstain-safety check

The CRAG loop already exists in `graph_service.py` (`critique` → `corrective_retrieve`:
step-back rewrite + re-retrieve + re-answer, max `AGENT_CRITIQUE_MAX_RETRIES=1`), default OFF.

```bash
CRAG_ENABLED=true uv run python scripts/eval/oracle_probe.py \
  --eval-set data/eval/c2_evalset_n40.jsonl --n 40 --retries 3 \
  --rows-out docs/eval/rows_c2_n40_crag_on.jsonl \
  --out docs/eval/c2_probe_crag_on_2026-07-14.md
```

**Safety gate (required):** CRAG treats an uncertain answer as ungrounded and retries — it
must NOT convert clean out-of-corpus abstentions into hallucinations:

```bash
CRAG_ENABLED=true uv run python scripts/eval/run_refusal_ab.py
```
Compare refusal classes to `docs/eval/benchmark_answerability_ab_2026-06-24_vi.md` — zero
`hallucinated` on the OOC set or CRAG stays off, full stop.

## Phase 3 — Decision doc

Write `docs/eval/crag_ab_2026-07-14.md`: table `arm | system avg | oracle−system | misses |
bucket split | refusal classes`. **Pre-registered rule:** flip `CRAG_ENABLED: bool = True` in
`src/agentrag/config.py` only if CRAG-on gains **system_avg ≥ +0.02** AND the refusal set
shows **zero new hallucinated**. Otherwise keep OFF and record why. Also record the HippoRAG
gate verdict from the Phase 1 buckets (multi-hop rows are tagged `source=prod_corpus_multihop`
if the set was rebuilt with `--multihop`).

## Phase 4 — Seed the flywheel

```bash
uv run python scripts/eval/mine_citation_pairs.py \
  --rows docs/eval/rows_c2_n40_crag_off.jsonl \
  --out data/finetune/citation_pairs.jsonl
wc -l data/finetune/citation_pairs.jsonl
```
Expect dozens of triplets — too few to train alone; accumulate across future probe runs with
`--append`, then blend with `data/finetune/embed_triplets.jsonl` for the next reranker FT
(`scripts/finetune_reranker.py`, ~5 GB VRAM).

## Wrap-up

```bash
printf 'docs/eval/rows_*.jsonl\n' >> .gitignore   # row dumps carry corpus text — keep out of git
git add .gitignore docs/eval/miss_buckets_2026-07-14.md docs/eval/crag_ab_2026-07-14.md \
  docs/eval/c2_probe_crag_off_2026-07-14.md docs/eval/c2_probe_crag_on_2026-07-14.md
git commit -m "docs(eval): miss buckets + CRAG A/B on real corpus"
```
If the decision flips the flag, commit that separately:
`git add src/agentrag/config.py && git commit -m "feat(agent): enable CRAG critique loop (A/B-validated)"`.

**Deliverables checklist:**
- [ ] `docs/eval/miss_buckets_2026-07-14.md` — bucket split of the ~5 misses
- [ ] `docs/eval/crag_ab_2026-07-14.md` — CRAG enable/keep-off decision + evidence
- [ ] refusal-safety confirmation with CRAG on (zero hallucinated)
- [ ] `data/finetune/citation_pairs.jsonl` — first flywheel seed
- [ ] HippoRAG-2 go/no-go verdict recorded (drives the next plan)
