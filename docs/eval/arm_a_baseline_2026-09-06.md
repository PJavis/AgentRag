# Arm-B Rollout Monitor

Answers examined: **19**, of which **16** cite a page that
carries a detected table (the only ones the overrun metric applies to).

| arm | answers | table-citing | abstention (all / table-citing) | lookup-overrun (of table-citing) | median latency |
|---|---|---|---|---|---|
| unknown | 19 | 16 | 0.11 / 0.06 | 0.25 | 37098 ms |

**`mixed` is answers citing segments from both arms** — mid-rollout that is
the transition, not evidence about either arm, so it is reported and never
folded in. `unknown` is answers whose citations carry no ingest stamp
(segments written before the stamp existed).

**This is not a controlled A/B.** Documents ingested before and after the flip
differ in more than the flag. Read this as a guardrail and a directional
check; see `docs/eval/table_arm_b_rollout_2026-09-06.md` for what a one-armed
rollout can and cannot conclude.
