# VITAL — Home-Run: Clean Re-Measure After the Eval-Quality Finding (2026-07-15)

**Why this run.** The 2026-07-14 CRAG A/B (`docs/eval/crag_ab_2026-07-14.md`) settled
one thing — **CRAG stays OFF** (Δ+0.015 < +0.02, decided) — but the per-row miss audit
found that **6 of 9 misses were broken synthetic questions**, not system failures:
dangling demonstratives ("bệnh nhân **này**", "môn học **này**") and meta-references to
the source artifact ("câu 8 đến 12 trong **đề thi**", "trong **đoạn văn**"). On those,
the oracle scores ~1.0 only because it is handed the gold context; the system scores 0
because a standalone question with no anchor has nothing to retrieve on. That inflates
the +0.171 oracle−system gap with fake headroom — the same class as the v3 landmine.

So before committing to any retrieval build (HippoRAG-2), we re-measure on a **clean**
eval set. A question-quality filter now ships in `build_prod_evalset.py`
(`src/agentrag/eval/question_quality.py`) that drops those questions at build time.

**Goal of this run:** get the TRUE oracle−system headroom and a clean bucket split, then
decide the HippoRAG-2 gate on real questions only.

---

## ⚡ Quick-start (copy-paste in order)

```bash
# 0. code + stack
git fetch origin && git checkout feat/miss-buckets-crag-flywheel && git pull
make serve-embed &                                   # TEI :8080 (e5-FT, 768-dim)
curl -s localhost:9200/_cluster/health | head -c 80  # ES green/yellow
curl -s localhost:8080/health && echo " TEI OK"
grep -c "^DEEPSEEK_API_KEY=." .env                   # must be 1; judges: gemini-2.5-pro / deepseek-v4-pro

# 1. rebuild eval set WITH the context-dependent-question filter
#    (also stamps corpus_fp so the probe's fingerprint guard is active)
uv run python scripts/eval/build_prod_evalset.py --n 40 --multihop 12 \
  --out data/eval/c2_evalset_n40_clean.jsonl
#    watch the log tail: "wrote N eval rows ... (dropped M context-dependent questions)"
#    expect M ≈ 4-8 dropped. If M = 0 the filter didn't engage — stop and ping.

# 2. baseline probe on the CLEAN set, CRAG OFF, ~30-60 min (detached)
CRAG_ENABLED=false nohup uv run python scripts/eval/oracle_probe.py \
  --eval-set data/eval/c2_evalset_n40_clean.jsonl --n 40 --retries 3 \
  --rows-out docs/eval/rows_c2_clean_off.jsonl \
  --out docs/eval/c2_probe_clean_off_2026-07-15.md > /tmp/probe_clean.log 2>&1 &
tail -f /tmp/probe_clean.log                         # wait for "[probe] wrote ..."

# 3. bucket the misses  ← THE DELIVERABLE
uv run python scripts/eval/report_miss_buckets.py \
  --rows docs/eval/rows_c2_clean_off.jsonl \
  --out docs/eval/miss_buckets_clean_2026-07-15.md --label c2_clean-crag-off

# 4. re-seed the flywheel from the clean run (more valid triplets now)
uv run python scripts/eval/mine_citation_pairs.py \
  --rows docs/eval/rows_c2_clean_off.jsonl \
  --out data/finetune/citation_pairs.jsonl --append

# 5. commit results + push
printf 'docs/eval/rows_*.jsonl\n' >> .gitignore   # row dumps carry corpus text
git add .gitignore docs/eval/c2_probe_clean_off_2026-07-15.md \
  docs/eval/miss_buckets_clean_2026-07-15.md
git commit -m "docs(eval): clean re-measure on filtered eval set"
git push
```

---

## How to read the result (pre-registered)

Compare the CLEAN run to the 2026-07-14 dirty run:

| signal | dirty (2026-07-14) | clean target | meaning |
|---|---|---|---|
| system avg | 0.740 | ↑ (broken Qs removed) | true system correctness |
| oracle − system | +0.171 | should SHRINK | fake headroom was eval-set noise |
| misses | 9/40 | fewer | |
| dominant bucket | retrieval_miss (contaminated) | — | now genuine |

**Decision branch — the whole point of the run:**

1. **Clean oracle−system shrinks toward ~0.05 AND the few remaining misses are NOT
   `retrieval_miss`** → the system is near the eval ceiling on real questions. **Shelve
   HippoRAG-2.** Move to the generation/abstention tail: the `generation_miss` rows
   (gold packed, answer wrong → answer-prompt work) and the one false-abstention
   (prod_corpus-27: gold packed at rerank 0.72, LLM refused anyway → abstain-prompt tune).

2. **`retrieval_miss` is STILL the dominant bucket among the now-genuine questions** →
   the HippoRAG-2 gate is truly green. Build the spec
   (`docs/superpowers/specs/2026-07-14-hipporag2-structmem-design.md`): inspect those
   real retrieval-miss rows first (are they multi-hop? synonym/terminology mismatch?) to
   confirm the graph design targets the actual failure shape before writing code.

Either way: **bring back the clean bucket split and the clean oracle−system number.**
That single line decides the next month of work.

---

## Notes / gotchas

- The filter is conservative (high-precision meta-artifact terms + a fixed dangling-noun
  list) so it won't drop real medical questions — verified against the 6 known-broken +
  5 known-good questions from the dirty run (`tests/eval/test_question_quality.py`). If
  the drop count looks too high (>12/40), skim the "drop context-dependent Q" log lines;
  a real question mis-flagged means loosening a pattern in `question_quality.py`.
- The clean set is a NEW question sample → its absolute numbers are not row-comparable to
  the dirty c2 set; compare the AGGREGATES (system avg, oracle−system, bucket split), not
  per-qid.
- Fingerprint guard is active: if you re-ingested anything since the build, the probe
  will hard-abort on mismatch — rebuild the set (step 1) after any ingest.
- CRAG-on arm is NOT repeated here — that decision (keep OFF) is already made on the dirty
  run and a cleaner correctness base won't flip a +0.015 gap. Skip it.
