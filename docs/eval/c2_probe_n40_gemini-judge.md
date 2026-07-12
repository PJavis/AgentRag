# Eval-fidelity probe — data/eval/c2_evalset_n40.jsonl n=40

- examples: 40
- system avg (ensemble): 0.759
- oracle avg (strong model + gold context): 0.847
- **oracle − system: +0.088**
- judge-noise floor (pearson, judge1 vs judge2): 0.921

## Read

If oracle − system is small (< ~0.05), perfect retrieval + a strong generator barely beats the live system — the cap is the gold/metric, not the system. A low judge-noise pearson means the judge itself is unreliable and must be fixed before any correctness number is trusted.

## Result read (2026-07-13) — independent-judge run

**Setup:** first run with a PAID gemini key. `eval_judge=gemini-2.5-pro` (primary, independent of
the deepseek answer model), `eval_judge2=deepseek-v4-pro` → the pearson is a **real cross-provider
agreement**, not same-family self-consistency. System under test = current prod: e5-FT embedding
(`agentrag-embed-v1`, 768-dim via TEI), lean corpus (CR/RAPTOR/StructMem/vision OFF,
115 docs / 3359 segs), answer=deepseek-v4-flash.

**Conclusions:**

1. **Judge-independence gap CLOSED.** Cross-provider pearson **0.921** (vs 0.730 same-family in
   v3). A gemini judge and a deepseek judge agree strongly on the same answers → the deepseek-judged
   history (C2 baseline 0.813, FT 0.764) was **not** self-preference-inflated. Correctness numbers
   from this rig are now quotable with the independent-judge caveat removed.
2. **Number robust to judge choice.** System 0.759 (gemini-judged) ≈ 0.764 (deepseek-judged FT run,
   same eval set) — within noise of each other.
3. **Real e2e headroom on the real corpus: +0.088** — above the ~0.05 noise line and consistent
   with the earlier deepseek-judged +0.080 baseline read. Unlike the residue corpus (+0.046,
   metric-bound), the real corpus is **system-bound**: better retrieval/answering can still gain
   up to ~0.09. Loss concentrated in ~5/40 total misses (sys=0.00).
4. This run doubles as the **CR-off arm** of the deferred real-corpus CR+RAPTOR A/B (identical
   judge map + eval set to be reused for the CR-on arm).

⚠️ `prod_corpus_evalset_v3.jsonl` is residue-corpus questions — invalid against the real corpus
(sys=0.00 on every q; verified 2026-07-13). Real-corpus probes must use `c2_evalset_n40.jsonl` or a
freshly built set.
