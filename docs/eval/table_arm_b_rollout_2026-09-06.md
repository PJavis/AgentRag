# Arm B — Ship Decision and Rollout Record (2026-09-06)

**Decision:** ship arm B behind `PDF_PRESERVE_TABLES`. Do not run Road 1.
**Decided by:** dungnq, 2026-09-06, on the Road 0 evidence below.
**State of the flag right now:** still `False`. Nothing in this document turns it
on; §3 is the pre-registration that must exist *before* it is turned on.

## 1. The basis, stated as it is

Not "the probe validated arm B". What actually happened:

| | |
|---|---|
| direction | consistent: **3 better / 0 worse / 16 same** across 19 paired lookups in 7 documents |
| mechanism | measured and specific: arm A's surplus answer tokens are **90% borrowed from other cells of the same table**; arm B's are **0%** |
| confound | ruled out: arm C (same content duplicated, no structure) scores **0.68**, *below* arm A's 0.72 — the gain is structure, not token volume |
| continuous metric | token-F1 **0.695 → 0.843**, doc-clustered 95% CI **[+0.000, +0.269]** |
| **the shipped rule** | **p = 0.250 — fails the threshold. INCONCLUSIVE.** |

The binary "cells read correctly" metric discards information: a case where arm A
locates the right value and glues debris from the adjacent row onto it scores as a
tie. That is where the 16 ties come from, and it is why the sign test on 3
discordant pairs is pessimistic here rather than merely underpowered.

The decision rests on **mechanism + consistent direction + a ruled-out confound**
outweighing a sign test on 3 discordant pairs — not on the sign test passing. It
did not pass. Anyone reading this later should see exactly that.

Road 1 was declined because the same analysis that says this evidence is
directionally sound also says Road 1 arrives at another INCONCLUSIVE: same unit
(the table), same 27-table pool, same 52%-in-one-document clustering, n_eff≈7.
Paying weeks for the same verdict is not diligence.

## 2. Recorded deviations from the pre-registered plan

Stopping rules exist to block post-hoc rationalisation, so every departure is
listed here with its justification — including the two nobody asked about.

### 2.1 The 0b kill condition was not honoured (declared, justified)

**Pre-registered:** *"NO-GO for the probe if arm B does not raise the gold row: the
mechanism fails before the answer model is even involved."*

**What happened:** arm B did not raise the gold row, and the NO-GO was not reported.

**Justification:** the rule's stated premise — *"therefore the answer model never
sees the row"* — was falsified by the same run's data. Arm A retrieves the gold row
at **recall@10 = 0.95** (bm25 and rrf alike). The model sees the row under both
arms, so "no retrieval gain" does not imply "no gain available". The rule was
written against a mechanism that turned out not to be the operative one.

**Why this survives scrutiny:** the premise is falsified by a *measurement inside
the pre-registered run*, not by an argument constructed afterwards to rescue a
result, and the falsifying number (recall@10 for **arm A**) is one that cannot be
argued to favour arm B — it is the number that removes arm B's advertised benefit.
The report states the non-firing in place of a NO-GO it did not earn.

### 2.2 A new step (0c) was added after seeing 0b's result

Road 0 pre-registered 0a and 0b only. 0c exists because 0b left exactly one live
hypothesis and it was cheap to test. It is a **post-hoc addition**: it was designed
after seeing 0b's outcome, and it is the step producing the effect this decision
rests on. Mitigations, all of which were in place before its numbers were read: the
metric (`extract()` ground truth), the scoring rule and the duplication control arm
were fixed in code and unit-tested first, and the shipped `decide_paired` verdict is
reported unchanged.

### 2.3 The 0b gold-matching rule was changed after seeing an unfavourable result

**This is the disclosure that matters most.** The first 0b run scored arm B as
*significantly worse* (ΔRR −0.200, p = 0.001). Investigating that result changed the
metric: gold chunks had been filtered by the chunker's page label, and a chunk
holding one page's tail plus the next page's marker is labelled with the *next*
page. Arm B's longer pages shift those boundaries more often, so the filter deleted
arm B's gold chunks. Removing the page filter moved the result to "no effect".

Changing a metric after seeing a result you dislike is exactly the pattern these
rules guard against. What makes it defensible here, and what a reader should check:

- The defect was demonstrated **mechanically on a named case** (`5164c4af` p16, where
  arm B's gold chunk existed and was labelled page 17), not inferred from the
  aggregate moving in a preferred direction.
- The replacement rule is **arm-neutral by construction** (bag-of-words within the
  corpus, no page constraint) and is unit-tested to be so.
- The corrected run still shows **no retrieval gain for arm B**. The change did not
  manufacture a favourable retrieval result; it removed a spurious unfavourable one.
  Arm B's case rests on 0c, which this fix does not touch.

### 2.4 Ranking scope and reproducibility were changed mid-flight

Ranking moved from per-document to corpus-wide (per-document hid the competition
arm B's own extra blocks create), and `AGENT_TEMPERATURE` was pinned to 0 with 3
samples per question after two runs of identical 0c code disagreed 15/19 vs 13/19.
Both changed before the final numbers were read; both are unfavourable-direction
changes for arm B (more competitors; less noise to win from).

## 3. What gets measured on production — pre-registered, before the flag goes on

Condition of the ship decision. In production there is **no `extract()` ground
truth** for an arbitrary user question, and every number in 0c rested on exactly
that. So what follows is what is decidable without it. Anything not on this list is
not being measured, and this document does not pretend otherwise.

### 3.0 Shipped with this decision so the rest is possible

`pipeline.build_chunk_metadata` now stamps `pdf_preserve_tables` into every PDF
segment's `extra_metadata`. Without it an answer cannot be attributed to an arm
after the fact and every metric below is assertable but not measurable. Segments
written before this stamp report as `unknown` and are never folded into A or B.

### 3.1 Tier 0 — deterministic, no traffic required (run before and after the flip)

| metric | how | pre-flip value |
|---|---|---|
| segments per corpus | re-chunk both arms offline | 1115 → 1161 (**+4.1%**) |
| historical traffic snapshot | `table_arm_b_monitor.py --from-db` | 12 answers, **0 table-citing** — the metric had never once fired |
| ingest failures | worker error rate | must not rise |

**Arm-A baseline, measured 2026-09-06** (`docs/eval/arm_a_baseline_2026-09-06.md`).
The historical snapshot was useless as a baseline — 12 answers, none citing a table —
so `scripts/eval/table_arm_baseline_run.py` drove the live `/chat` endpoint with the
same 19 row lookups 0c used, phrased as a user would. Re-run it after the flip with
`--tag armB-<date>` for the comparable other side.

| metric | arm A |
|---|---|
| answers / of which cite a table | 19 / **16** |
| abstention rate (all / table-citing) | 0.11 / **0.06** |
| **lookup-overrun rate** (of table-citing) | **0.25** |
| median latency | **37.1 s** |

Two things this baseline settles, neither of which was known before it ran:

- **The overrun metric fires on real answers** — 0.25, not 0. A metric that only
  ever reads zero would have made every rollback trigger vacuous.
- **The observed 0.25 sits next to the 0.30 the floor table assumed**, so the 62-per-arm
  figure for a 0.30 → 0.10 drop is the right order and does not need restating.

**Read the arm label honestly.** Every answer above reports arm `unknown`, because the
corpus was ingested before the provenance stamp existed. After the flip and re-ingest,
answers report `B`. The trial comparison is therefore *this snapshot* against a post-flip
`B` snapshot — the same questions and the same script, but not a randomised A/B, and
between the two snapshots the corpus is re-ingested, which changes more than the flag.

### 3.2 Tier 1 — from `chat_messages`, via `scripts/eval/table_arm_b_monitor.py`

Both metrics are the mechanism 0c isolated, not proxies invented to have something
to plot:

1. **Abstention rate** on answers citing a table-bearing page. 0c measured 5 → 2
   abstentions in 19 lookups; if that was real, the live rate falls. Uses the
   shipped `_UNCERTAINTY_MARKERS`, so it cannot drift from production.
2. **Lookup-overrun rate** — a *short* answer whose tokens are drawn from two or
   more distinct rows of the cited table. This is cell-boundary bleed, computable
   with no gold answer. Long answers are excluded: they may legitimately summarise
   several rows, and flagging them would manufacture a signal.

**Sample floor, fixed now so it cannot be chosen later.** Two-proportion test,
α = 0.05, power = 0.80:

| overrun rate change to detect | table-citing answers needed **per arm** |
|---|---|
| 0.30 → 0.05 | 36 |
| 0.30 → 0.10 | **62** |
| 0.30 → 0.15 | 121 |
| 0.20 → 0.10 | 199 |

**Measured reality: the baseline snapshot holds 12 answers, of which 0 cite a
table-bearing page.** At that traffic the floor is far away. Until it is met the
Tier-1 numbers are descriptive only and must not be quoted as a result.

### 3.3 Tier 2 — the only real A/B, if it is ever funded

Ingest the corpus into two indices (flag off / flag on), **replay stored production
questions** against both, and judge pairwise with an independent model. This uses
accumulated real user questions instead of hand-authored ones, which is what made
Road 1 expensive. It is not scheduled and not funded. Its absence is why §3.4 names
this what it is.

### 3.4 The honest name

With Tier 0 + Tier 1 only, and with the sample floor unmet, this is an
**instrumented trial rollout, not a measurement**. Documents ingested before and
after the flip differ in more than the flag — corpus growth, model drift, question
mix — so no Tier-1 comparison is causal. Calling it an A/B would be false.

## 4. Rollback triggers (any one, immediately)

- median answer latency on table-citing answers rises > 20% against the §3.1 baseline
  (**37.1 s → 44.5 s**)
- abstention rate on table-citing answers rises above **0.06** (the predicted
  direction is down; 0c measured 5 → 2 abstentions in the same 19 lookups)
- lookup-overrun rate on table-citing answers does not fall below **0.25**, or rises
- ingest failure rate rises, or segment count per document grows > 15% (expected +4.1%)
- any reported answer that attributes one table row's content to another

Rollback is `PDF_PRESERVE_TABLES=false` plus re-ingest of anything ingested while on.

## 5. What this rollout cannot conclude

- That arm B improves answers to real user questions. Every number behind the
  decision comes from mechanical cell lookup.
- Anything about the 25% of corpus pages with no text layer: `find_tables()` is
  blind there and arm B is byte-identical to arm A.
- Anything that repairs the clustering bound. 52% of the table pool is one document.
- That the sign test was wrong. It was not — it was pessimistic for a stated,
  inspectable reason (§1), and it still says INCONCLUSIVE.
