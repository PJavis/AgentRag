# VITAL — Home-Run: The 2 True Recall Gaps (2026-07-17)

**Why this run.** The rerank-before-trim fix (`docs/eval/rerank_before_trim_2026-07-16.md`)
closed the campaign to **system 0.884 / oracle−system +0.044 — at the eval ceiling**. Of the
3 remaining misses, one (multihop-6) is answer-vs-judge (needs an eyeball, not a build), and
**two are genuine recall gaps** where the gold passage never enters the candidate pool at all
— even at the widened breadth (raw-query top_k=50):

- **prod_corpus-15** — "Những yếu tố nào cần được khai thác trong tiền sử bệnh?" (history-taking factors)
- **prod_corpus-23** — "Các nguyên nhân thường gặp gây tiểu không tự chủ là gì?" (incontinence causes)

Both are broad **"list/enumerate" clinical questions** in general terms, while the gold chunk
is a specific medical list. That is a classic recall failure: neither BM25 (vocabulary
mismatch) nor the dense embedding surfaces it. This run is **diagnosis-first** — the lever
(chunking vs embedding vs query-side) isn't known yet, so measure before building.

---

## ⚡ Phase 0 — Diagnose: is the gold SPLIT, or present-but-unranked?

```bash
git fetch origin && git checkout feat/miss-buckets-crag-flywheel && git pull
make serve-embed &                                    # TEI :8080
curl -s localhost:9200/_cluster/health | head -c 80   # ES up

# pull the 2 gold passages out of the eval set
python3 - <<'PY'
import json
want = {"prod_corpus-15", "prod_corpus-23"}
for line in open("data/eval/c2_evalset_n40_clean_v2.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] in want:
        print("\n===", r["id"], "\nQ:", r["question"])
        for i, g in enumerate(r["gold_contexts"]):
            print(f"gold[{i}] ({len(g)} chars):", g[:400])
PY
```

For each gold passage, answer two questions:

1. **Is the gold text present as ONE segment in ES, or split across chunk boundaries?**
   Grep a distinctive ~8-word phrase from the gold against the corpus segments:
   ```bash
   curl -s 'localhost:9200/agentrag_segments/_search' -H 'Content-Type: application/json' -d '{
     "query": {"match_phrase": {"content": "<distinctive phrase from gold>"}},
     "_source": ["content","document_title"], "size": 3}' | python3 -m json.tool | head -40
   ```
   - **No full-phrase hit / phrase spans two segments** → **chunking split the answer** (the
     512-token/64-overlap boundary cut the list). → Lever A.
   - **Full phrase is in one segment** → gold is indexed fine but retrieval doesn't rank it. → Phase 0 step 2.

2. **Where does the gold segment rank in raw retrieval?** Run the question through hybrid,
   dense-only, sparse-only and read the gold's rank:
   ```bash
   uv run python scripts/eval/benchmark_retrieval.py --help   # confirm flags on this branch
   # or a one-off: embed the query, kNN top-100, print rank of the gold segment id
   ```
   - Gold ranks **51–100** (just past the top_k=50 backstop) → **breadth**, cheap fix (raise
     `RETRIEVAL_RAW_QUERY_TOP_K` again) — but confirm it doesn't reflood; unlikely to be the
     whole story since rerank-before-trim already widened.
   - Gold ranks **>100 or absent in dense** but the phrase IS indexed → **embedding recall
     failure on broad queries** → Lever B / C.

Write findings to `docs/eval/recall_gap_diagnosis_2026-07-17.md` (which of A/B/C each row is).
The diagnosis decides the lever — do NOT build before it.

---

## Lever A — chunking (if gold is split across segments)

The answer list straddles a chunk boundary, so no single segment scores well.

- Re-ingest with a bigger overlap so the full list co-occurs in one chunk:
  `SEARCH_CHUNK_OVERLAP_TOKENS=64 → 128` (config.py), or lift `SEARCH_CHUNK_MAX_TOKENS`
  512 → 768 for list-heavy docs. `HybridChunker` already splits on headings/paragraphs;
  check whether the gold list has a heading that a smaller `max_tokens` is cutting under.
- Re-ingest the affected doc(s) only, rebuild `c2_evalset_n40_clean_v2` **corpus_fp will
  change** → the probe's fingerprint guard forces a rebuild (that's correct).
- Gate: re-run the clean-v2 probe; **ship only if Δsystem ≥ +0.01, zero regressions, OOC
  15/15 abstain**. Bigger chunks cost precision — watch for new misses.

## Lever B — query-side (HyDE / expansion for enumerate questions)

If gold is indexed and single-segment but ranks deep in dense: the query is too general.
HyDE already exists (`retrieval/query_rewriter.py`) — a hypothetical answer carries the
specific medical terms the gold chunk uses, closing the vocabulary gap.

- Enable HyDE for the retrieval path (check the flag in `config.py` / `query_rewriter.py`)
  and re-probe. This is the **cheapest** real lever — no re-ingest, no train.
- Gate: same bar (Δ ≥ +0.01, zero regressions, OOC unchanged). HyDE can hurt OOC (a
  hallucinated hypothetical retrieves distractors) → the refusal A/B is mandatory here.

## Lever C — embedding recall (retrain e5-FT on hard enumerate cases)

If gold is single-segment, indexed, but dense recall@100 still misses it, the embedding
doesn't place broad "list" queries near their specific-list gold.

- Mine hard (query, gold) pairs of this shape, add to `data/finetune/embed_triplets.jsonl`,
  retrain `models/agentrag-embed-v1` via `scripts/finetune_embedding.py`.
- Gate with the existing promote rule: `scripts/eval/eval_retrieval.py` requires **recall@10
  +0.05 and mrr@10 +0.03 with ≤0.02 regression** before wiring the new model. Then re-probe
  end-to-end (recall lift did NOT propagate to correctness before — 2026-06-28 — so measure
  the e2e number, not just recall).
- Note: changing `EMBEDDING_MODEL` dims DROPs the ES index (`_recreate_index_if_dims_changed`)
  → full re-ingest. Highest-cost lever; only if A and B don't close it.

---

## Read

Two rows is a **small, cheap tail** — this is polishing at the ceiling, not a headroom
push (the +0.044 is already at the metric floor). Budget accordingly: try Lever B (HyDE,
no re-ingest) first, then A, and only reach for C if the diagnosis clearly says embedding.
If all three miss the +0.01 bar, the honest call is **accept the 2-row tail** — the system
is at the eval ceiling and further gain wants more/better gold, not more retrieval.

**Bring back:** `recall_gap_diagnosis_2026-07-17.md` (A/B/C per row) + whichever lever's
probe result. Same discipline as the whole campaign: measure, pre-register the bar, ship
only a real win.
