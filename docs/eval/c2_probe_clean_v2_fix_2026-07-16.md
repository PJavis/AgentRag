# Eval-fidelity probe — data/eval/c2_evalset_n40_clean_v2.jsonl n=42

- examples: 42
- system avg (ensemble): 0.884
- oracle avg (strong model + gold context): 0.928
- **oracle − system: +0.044**
- judge-noise floor (pearson, judge1 vs judge2): 0.862

## Read

If oracle − system is small (< ~0.05), perfect retrieval + a strong generator barely beats the live system — the cap is the gold/metric, not the system. A low judge-noise pearson means the judge itself is unreliable and must be fixed before any correctness number is trusted.
