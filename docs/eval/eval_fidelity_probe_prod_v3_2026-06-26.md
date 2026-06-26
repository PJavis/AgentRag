# Eval-fidelity probe — data/eval/prod_corpus_evalset_v3.jsonl n=50

- examples: 50
- system avg (ensemble): 0.888
- oracle avg (strong model + gold context): 0.934
- **oracle − system: +0.046**
- judge-noise floor (pearson, judge1 vs judge2): 0.730

## Read

If oracle − system is small (< ~0.05), perfect retrieval + a strong generator barely beats the live system — the cap is the gold/metric, not the system. A low judge-noise pearson means the judge itself is unreliable and must be fixed before any correctness number is trusted.

## Interpretation (n=50 clean run vs v2)

| | v2 (n=20, gemini judge, 10×503 skips) | **v3 (n=50, deepseek judge, 0 skips)** |
|---|---|---|
| system avg | 0.950 | **0.888** |
| oracle avg | 0.969 | 0.934 |
| oracle − system | +0.019 | **+0.046** |
| judge-noise pearson | — | 0.730 |

**Two findings:**
1. **Reframe CONFIRMED at clean n=50.** oracle − system = +0.046 (< 0.05): perfect retrieval +
   a strong generator barely beats the live system → the correctness cap is the **metric/gold**,
   not retrieval or generation. The eval-is-the-ceiling conclusion survives a 2.5× larger,
   skip-free sample.
2. **The v2 0.950 was optimistic — real clean number is ~0.888.** v2 scored only 20 of 30 (10
   dropped on gemini-503) → selection bias toward the easy questions that didn't error, plus a
   different judge. The cleaner n=50 lands the system at **0.888**. Same lesson as the CR+RAPTOR
   ablation: small-n numbers run hot; trust higher-n.

**Caveats on this run:**
- **Judge independence is weak.** Free-tier gemini serves `gemini-2.5-pro` at limit:0, so the
  documented cross-provider eval path was unrunnable; all eval slots (oracle_gen/gold_gen/
  eval_judge/eval_judge2) used DeepSeek. judge1/judge2 are deepseek flash vs pro → the
  judge-noise pearson (0.730) is the *optimistic* (same-family) case, and the judge shares a
  provider with the system's answer model (deepseek-v4-flash) → possible mild self-preference
  inflating 0.888. A true cross-provider noise floor + a bias-free score need **paid gemini**
  (or another independent provider) in the judge slot.
- **Corpus is the indexed eval-residue gold contexts** (`vn_bkai`/`vn_legal`), not the real
  `data/originals` medical corpus — this confirms the judge/ceiling story, not real-doc behavior.
  The prod-corpus A/B (ingest `data/originals` first) remains the separate, deferred test.

**Net:** the system is at the eval ceiling (confirmed); the trustworthy clean correctness number
is ~0.888 (not 0.950); the ruler itself still needs an independent (paid-gemini) judge before any
correctness figure is quoted with confidence.
