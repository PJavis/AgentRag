# Eval-fidelity probe — data/eval/c2_evalset.jsonl n=10

- examples: 10
- system avg (ensemble): 0.920
- oracle avg (strong model + gold context): 0.950
- **oracle − system: +0.030**
- judge-noise floor (pearson, judge1 vs judge2): 0.812

## Read

If oracle − system is small (< ~0.05), perfect retrieval + a strong generator barely beats the live system — the cap is the gold/metric, not the system. A low judge-noise pearson means the judge itself is unreliable and must be fixed before any correctness number is trusted.
