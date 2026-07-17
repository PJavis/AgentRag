# HippoRAG-2-style StructMem Evolution — Design (DRAFT, GATED)

- **Date:** 2026-07-14
- **Branch context:** `feat/miss-buckets-crag-flywheel`
- **Status:** ⛔ **SHELVED (2026-07-16) — gate resolved NO.** The clean re-measure
  (`docs/eval/clean_remeasure_v2_2026-07-16.md`, eval set filtered of context-dependent
  questions) shows **every multi-hop miss failed at GENERATION, not retrieval**
  (gold_overlap = 1.00 on both) — multi-hop retrieval already works, so this design targets
  a failure mode that is not occurring. The residual +0.118 headroom splits into single-hop
  retrieval-coverage misses (4/7 — chunking/embedding/rerank, not a graph) and answer-generation
  misses (3/7 — answer-prompt/model). Do NOT build this. Kept for the record + as a fallback if
  a future corpus grows a real multi-hop retrieval gap. Original gate rationale below.
- **Original gate:** DRAFT — gated on the miss-bucket verdict from
  `docs/HOME-RUN-miss-buckets-2026-07-14.md`. Build ONLY if `retrieval_miss` is the dominant
  bucket (especially on multi-hop rows). If `false_abstention` or `generation_miss` dominates,
  shelve this and do floor/gate or answer-prompt work instead.

## 0. Problem

Real-corpus probes (2026-07-13, `docs/eval/c2_probe_n40_gemini-judge.md`) show
**oracle − system = +0.088** — the live system leaves real headroom on the real medical
corpus, concentrated in ~5/40 total misses. The suspected shape of the retrieval-side tail:
**multi-hop questions** whose answer spans chunks that never co-occur in one rerank pool, and
**synonym/surface-form mismatches** (drug brand vs generic names, VN/EN terminology) that
hybrid search + cross-encoder rerank cannot bridge.

Current graph-ish assets:
- **Document StructMem** (`STRUCTMEM_INDEX=agentrag_memory_doc`, built at ingest): flat
  entries per chunk, retrieved as extra context. Measured +0.065 citation quality — kept.
- **Chat StructMem** (`ChatMemoryService`, conversation memory): out of scope here.
- **Multi-hop machinery** exists but is *sequential-query chaining*
  (`AGENT_MULTIHOP_ENABLED`, `_chain_query` in `graph_service.py`) — it depends on the
  planner emitting the right sub-queries; it cannot discover connections the planner didn't
  guess. Ablation (T6) showed no above-noise gain — flag OFF.

The prior evolution note (`memory: structmem-evolution-hipporag`) named **canonicalization**
as a prerequisite. HippoRAG 2's design removes that prerequisite: **synonym edges via
embedding-similarity threshold replace hard entity resolution.**

## 1. Goal & success criteria

Retrieval that finds passages connected to the question *through* intermediate entities —
without depending on the planner guessing the chain.

- **Primary gate (from the miss-bucket campaign):** on the `retrieval_miss` rows of the c2
  set (and its `--multihop` arm), the PPR-augmented retriever puts a gold passage into the
  packed context where the current retriever does not. Target: ≥ half of `retrieval_miss`
  rows recovered, ending within judge noise of oracle on those rows.
- **Recall floor:** recall@10 on the multi-hop eval arm improves ≥ +0.05 over hybrid+rerank
  baseline, without degrading single-hop recall (measure both arms).
- **No safety regression:** abstain behavior unchanged on the OOC refusal set (PPR adds
  candidates to the rerank pool; the floor gate still governs).
- **Latency budget:** +≤300 ms p50 per retrieval (PPR over a graph this size is
  milliseconds; the budget is for ES round-trips).

## 2. Design (HippoRAG 2 recipe, minimum viable adaptation)

### 2.1 Graph schema — bi-modal, stored alongside existing indices

Two node kinds, three edge kinds (nodes/edges as ES docs or a NetworkX graph persisted per
corpus snapshot; decide in the implementation plan — corpus is small: 115 docs / 3.3k segs):

- **Phrase nodes** — subjects/objects of OpenIE-style triples extracted per segment at
  ingest (LLM extraction, same pass that builds StructMem entries today; VN prompts).
- **Passage nodes** — one per existing `Segment` (`content_hash` as id; no re-chunking).
- **Relation edges** — phrase→phrase from triples `(subject, relation, object)`.
- **Synonym edges** — phrase↔phrase where embedding cosine > τ_syn (start 0.85 on the e5-FT
  embedding; sweep on the multi-hop arm). *This is the canonicalization replacement* — "tăng
  huyết áp" ↔ "cao huyết áp" ↔ "hypertension" connect without an ontology. The existing
  `ontology` module's synonym table can seed extra edges for free.
- **Contains edges** — passage→phrase for every phrase extracted from that passage.

### 2.2 Retrieval flow (new tool arm inside `ContextAssembler`/KnowledgeService)

1. **Seed selection** — embed the query; link to top-k triples/phrases by embedding
   similarity (query-to-triple, richer than entity-only). Passage nodes seeded by dense
   similarity as today.
2. **Recognition-memory filter** — one cheap LLM call filters the top-k linked triples for
   actual relevance to the question (HippoRAG 2's noise gate; task slot `classify`).
3. **PPR** — personalized PageRank with restart mass on surviving phrase seeds (rank-scaled)
   + passage seeds (similarity-scaled). NetworkX `pagerank` with `personalization=`; damping
   0.5 per HippoRAG.
4. **Merge** — top-m passages by PPR score are ADDED to the existing hybrid candidate pool
   (union, like `RETRIEVAL_INCLUDE_RAW_QUERY` — PPR can only add, never drop the current
   best), then the cross-encoder reranks the union as today. Floor/abstain logic untouched.

Flag: `RETRIEVAL_PPR_ENABLED` (default **False**), plus `PPR_TOP_M`, `PPR_SYNONYM_TAU`.

### 2.3 Ingest additions

Triple extraction per segment (LLM, cached like StructMem extraction), phrase embedding,
edge construction. Rebuild is per-corpus-snapshot; stamp with `corpus_fp`
(`src/agentrag/eval/corpus_fingerprint.py`) so a stale graph refuses to serve a new corpus —
same guard as eval sets.

## 3. Evaluation plan (reuses this branch's tooling)

1. Rebuild c2 eval set with `--multihop 12` (stamped with `corpus_fp`).
2. A/B: `RETRIEVAL_PPR_ENABLED` off/on → `oracle_probe.py --rows-out` both arms →
   `report_miss_buckets.py` — did `retrieval_miss` shrink? did any bucket grow?
3. Refusal safety: `run_refusal_ab.py` with PPR on (union + floor should hold; verify).
4. Decision rule (pre-registered): merge enabled only if primary gate met AND no safety
   regression AND latency within budget.

## 4. Risks / open questions

- **Extraction quality on VN medical text** — triples from a 7B/flash model may be noisy;
  the recognition filter + rerank union bound the damage (noise can't evict good hybrid
  hits). Sample-audit 50 extracted triples before building the full graph.
- **Graph staleness** — mitigated by `corpus_fp` stamp (refuse, don't drift).
- **τ_syn too low → topic drift** (PPR mass leaks through weak synonym edges); sweep on the
  multi-hop arm, start conservative (0.85).
- **Cost** — one extraction LLM call per segment at ingest (~3.3k calls once per corpus,
  cacheable); one filter call per query at retrieve time.
- **Is chat-memory PPR worth it too?** Out of scope; revisit after document-side verdict.

## 5. Out of scope

Community detection / hierarchical summaries (GraphRAG-style — the expensive machinery that
failed the T6/n40 test), LightRAG dual-level keyword routing (fallback candidate if PPR
underdelivers), MaGiX multi-aspect rerank (candidate follow-up if synonym edges prove
valuable), any change to abstain/floor semantics.
