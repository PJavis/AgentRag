# Eval-fidelity probe — data/eval/c2_evalset_n40_clean.jsonl n=40

- examples: 40
- system avg (ensemble): 0.787
- oracle avg (strong model + gold context): 0.950
- **oracle − system: +0.163**
- judge-noise floor (pearson, judge1 vs judge2): 0.934

## Read

If oracle − system is small (< ~0.05), perfect retrieval + a strong generator barely beats the live system — the cap is the gold/metric, not the system. A low judge-noise pearson means the judge itself is unreliable and must be fixed before any correctness number is trusted.
