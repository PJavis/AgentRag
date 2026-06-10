# RAG Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Contextual Retrieval, a RAPTOR summary layer, a CRAG critique loop with multi-hop chaining, adaptive fast-path routing, and a semantic retrieval cache to AgentRag — lifting answer correctness ≥0.85, precision ≥0.88, and cutting p50 latency below 10s.

**Architecture:** Five feature-flagged workstreams that slot into the existing LangGraph orchestrator, 3-layer domain-partitioned KB, and `ServiceContainer` DI. Ingest-time work (contextualize, RAPTOR) runs in the existing async worker; query-time work (adaptive route, semantic cache, critique) runs inside the LangGraph nodes. Every workstream has its own `*_ENABLED` flag so the benchmark can ablate each independently.

**Tech Stack:** Python 3 + asyncio, Pydantic `BaseSettings` config, Elasticsearch (`dense_vector` + BM25), `LLMGateway` (DeepSeek via `LLM_TASK_MODEL_MAP`), `umap-learn` + `scikit-learn` (RAPTOR clustering), `pytest`/`pytest-asyncio`.

**Conventions in this repo (read before starting):**
- Settings live in `src/agentrag/config.py` on `class Settings(BaseSettings)`; access via `from src.agentrag.config import settings`.
- LLM calls go through `LLMGateway.text_response(system_prompt, user_prompt, task=...)` / `.json_response(...)`; per-task model routing is keyed by the `task` string in `LLM_TASK_MODEL_MAP` (so a new task name like `"contextualize"` can be pointed at DeepSeek in `.env` without code changes).
- Tests use `pytest` + `pytest-asyncio`; run from repo root with `uv run pytest ...`.
- ES `content_hash` is `sha256(content)`; embeddings are `list[float]`; ES index name is `settings.ELASTICSEARCH_INDEX_NAME` (`agentrag_segments`).

---

## File Structure

**New files:**
- `src/agentrag/ingestion/contextualizer.py` — WS1: LLM context-gen per chunk, file-cached.
- `src/agentrag/ingestion/raptor.py` — WS2: recursive cluster + summarize → summary nodes.
- `src/agentrag/services/semantic_cache.py` — WS5: embedding-cosine retrieval cache.
- `tests/ingestion/test_contextualizer.py`, `tests/ingestion/test_raptor.py`
- `tests/services/test_semantic_cache.py`
- `tests/agent/test_critique.py`, `tests/agent/test_adaptive_routing.py`

**Modified files:**
- `src/agentrag/config.py` — all new flags.
- `src/agentrag/ingestion/stores/elasticsearch_store.py` — mapping fields `context_text`, `node_level`, `child_ids`; BM25 field; `_normalize_hits` passthrough.
- `src/agentrag/ingestion/pipeline.py` — contextualize step before embed; RAPTOR step after leaf index.
- `src/agentrag/retrieval/elasticsearch_retriever.py` — cache-key fix; semantic cache wiring; summary-level balance.
- `src/agentrag/structured/query_classifier.py` — `complexity` + `single_domain` on `ClassifierOutput`.
- `src/agentrag/agent/graph_service.py` — fast-path node + edge (WS4); critique node + edge (WS3); multi-hop sequential bootstrap branch.
- `src/agentrag/agent/service.py` — `_critique()` method.
- `pyproject.toml` — `umap-learn`, `scikit-learn`.
- `.env.example`, `README.md` — document new flags.

---

## PHASE 0 — Config + ES mapping foundation

### Task 0.1: Add all new config flags

**Files:**
- Modify: `src/agentrag/config.py` (append near the existing `RETRIEVAL_*` / `AGENT_PLAN_*` / `QUERY_REWRITE_*` block, ~line 280)
- Test: `tests/services/test_new_config_flags.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/services/test_new_config_flags.py
from src.agentrag.config import settings


def test_new_enhancement_flags_have_safe_defaults():
    # All new features default OFF so production behavior is unchanged until enabled.
    assert settings.CONTEXTUAL_RETRIEVAL_ENABLED is False
    assert settings.RAPTOR_ENABLED is False
    assert settings.CRAG_ENABLED is False
    assert settings.ADAPTIVE_ROUTING_ENABLED is False
    assert settings.SEMANTIC_CACHE_ENABLED is False
    # Sensible numeric defaults.
    assert settings.RAPTOR_MIN_LEAVES == 8
    assert settings.SEMANTIC_CACHE_THRESHOLD == 0.97
    assert settings.ADAPTIVE_FASTPATH_MIN_CONFIDENCE == 0.85
    assert settings.CONTEXTUAL_RETRIEVAL_TASK == "contextualize"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/services/test_new_config_flags.py -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'CONTEXTUAL_RETRIEVAL_ENABLED'`

- [ ] **Step 3: Add the flags**

In `src/agentrag/config.py`, after the `QUERY_REWRITE_DECOMPOSE` line (~284), insert:

```python
    # ── WS1: Contextual Retrieval (Anthropic) ────────────────────────────────
    #: When on, each chunk gets a 50–100 token LLM-generated context prepended
    #: to the text that is embedded + BM25-indexed (original `content` still cited).
    CONTEXTUAL_RETRIEVAL_ENABLED: bool = False
    #: LLMGateway task name → routable to DeepSeek via LLM_TASK_MODEL_MAP.
    CONTEXTUAL_RETRIEVAL_TASK: str = "contextualize"
    #: Cap the document text sent as the cached situating prefix (chars).
    CONTEXTUAL_MAX_DOC_CHARS: int = 48000
    CONTEXTUAL_CACHE_DIR: str = ".cache/agentrag/context"

    # ── WS2: RAPTOR summary layer ────────────────────────────────────────────
    RAPTOR_ENABLED: bool = False
    RAPTOR_MAX_LEVELS: int = 3
    #: Documents with fewer leaf chunks than this skip RAPTOR (no value).
    RAPTOR_MIN_LEAVES: int = 8
    #: Target average cluster size when picking the number of GMM components.
    RAPTOR_CLUSTER_SIZE: int = 5
    RAPTOR_SUMMARY_TASK: str = "raptor_summary"
    #: Max share of a result set that may be RAPTOR summary nodes.
    RAPTOR_SUMMARY_MAX_RATIO: float = 0.4

    # ── WS3: CRAG critique + multi-hop ───────────────────────────────────────
    CRAG_ENABLED: bool = False
    #: Min retrieved-hit count below which retrieval is judged "incorrect".
    CRAG_MIN_HITS: int = 1
    CRAG_GROUNDING_ENABLED: bool = True
    AGENT_CRITIQUE_MAX_RETRIES: int = 1
    AGENT_MULTIHOP_ENABLED: bool = False

    # ── WS4: Adaptive routing ────────────────────────────────────────────────
    ADAPTIVE_ROUTING_ENABLED: bool = False
    ADAPTIVE_FASTPATH_MIN_CONFIDENCE: float = 0.85

    # ── WS5: Semantic retrieval cache ────────────────────────────────────────
    SEMANTIC_CACHE_ENABLED: bool = False
    SEMANTIC_CACHE_THRESHOLD: float = 0.97
    SEMANTIC_CACHE_TTL_SECONDS: int = 120
    SEMANTIC_CACHE_MAX_ITEMS: int = 256
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/services/test_new_config_flags.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/config.py tests/services/test_new_config_flags.py
git commit -m "feat(config): add flags for CR, RAPTOR, CRAG, adaptive routing, semantic cache"
```

---

### Task 0.2: Add ES mapping fields `context_text`, `node_level`, `child_ids`

**Files:**
- Modify: `src/agentrag/ingestion/stores/elasticsearch_store.py` (`ensure_index` mapping ~136-171; `_ensure_segment_fields` ~188-205; `index_segments` doc_body ~327-341; `_normalize_hits` ~628-651)
- Modify: `src/agentrag/ingestion/stores/elasticsearch_store.py` (`sparse_search` fields ~461-465)
- Test: `tests/ingestion/test_es_mapping_fields.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_es_mapping_fields.py
from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore


def test_index_segments_doc_body_includes_new_fields(monkeypatch):
    store = ElasticsearchStore.__new__(ElasticsearchStore)  # no ES client
    store.index_name = "test_segments"
    captured = {}

    async def fake_ensure_index(dims):  # noqa: ANN001
        return None

    async def fake_bulk(body, refresh):  # noqa: ANN001
        captured["body"] = body
        return {"errors": False}

    class _FakeClient:
        async def bulk(self, body, refresh):  # noqa: ANN001
            return await fake_bulk(body, refresh)

    store.client = _FakeClient()
    store.ensure_index = fake_ensure_index  # type: ignore[assignment]

    chunks = [{
        "content": "leaf text",
        "context_text": "This passage is from the cardiology chapter on MI.",
        "embedding": [0.1, 0.2],
        "node_level": 0,
        "child_ids": [],
        "content_hash": "abc",
    }]
    import asyncio
    asyncio.run(store.index_segments(chunks, "Doc A"))

    doc_body = captured["body"][1]  # [action, doc, action, doc, ...]
    assert doc_body["context_text"] == "This passage is from the cardiology chapter on MI."
    assert doc_body["node_level"] == 0
    assert doc_body["child_ids"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_es_mapping_fields.py -v`
Expected: FAIL with `KeyError: 'context_text'`

- [ ] **Step 3: Add fields to mapping, doc_body, normalize, and BM25**

In `ensure_index` mapping `properties` (after `"canonical_terms"` ~153), add:

```python
                    "context_text": {"type": "text"},
                    "node_level": {"type": "integer"},
                    "child_ids": {"type": "keyword"},
```

In `_ensure_segment_fields` `properties` (after `"canonical_terms"` ~196), add the same three:

```python
                        "context_text": {"type": "text"},
                        "node_level": {"type": "integer"},
                        "child_ids": {"type": "keyword"},
```

In `index_segments` `doc_body` (after `"canonical_terms": ...` ~339), add:

```python
                "context_text": chunk.get("context_text"),
                "node_level": chunk.get("node_level", 0),
                "child_ids": chunk.get("child_ids") or [],
```

In `_normalize_hits` (after `"page_end": ...` ~647), add:

```python
                    "context_text": payload.get("context_text"),
                    "node_level": payload.get("node_level", 0),
```

In `sparse_search` `multi_match.fields` (~463), add `context_text` so the situating context is BM25-searchable:

```python
                "fields": ["content^2", "context_text^1.5", "document_title^1.5", "section_path"],
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_es_mapping_fields.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/ingestion/stores/elasticsearch_store.py tests/ingestion/test_es_mapping_fields.py
git commit -m "feat(es): add context_text, node_level, child_ids fields + BM25 context_text"
```

---

## PHASE 1 — WS5: Semantic cache + cache-key fix

### Task 1.1: Fix `_cache_key` to include `dense_query`

**Files:**
- Modify: `src/agentrag/retrieval/elasticsearch_retriever.py` (`_cache_key` ~25-28; call site ~101-104)
- Test: `tests/retrieval/test_cache_key.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_cache_key.py
from src.agentrag.retrieval.elasticsearch_retriever import _cache_key


def test_cache_key_distinguishes_dense_query():
    # Same base query, different HyDE-augmented dense_query → MUST differ,
    # else a HyDE result collides with a non-HyDE result.
    k_plain = _cache_key("đau ngực", "hybrid", 10, None, True, dense_query=None)
    k_hyde = _cache_key("đau ngực", "hybrid", 10, None, True, dense_query="đau ngực do nhồi máu cơ tim ...")
    assert k_plain != k_hyde
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_cache_key.py -v`
Expected: FAIL with `TypeError: _cache_key() got an unexpected keyword argument 'dense_query'`

- [ ] **Step 3: Add `dense_query` to the key**

Replace `_cache_key` (lines 25-28):

```python
def _cache_key(query: str, mode: str, top_k: int | None, document_title: str | None, rerank: bool | None, extra: dict | None = None, dense_query: str | None = None) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([query, mode, top_k, document_title, rerank, extra, dense_query], ensure_ascii=False, sort_keys=True).encode())
    return h.hexdigest()
```

Update the call site in `_search_impl` (lines 101-104):

```python
        ck = _cache_key(
            query, mode, top_k, document_title, rerank,
            extra=filters, dense_query=dense_query,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retrieval/test_cache_key.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/retrieval/elasticsearch_retriever.py tests/retrieval/test_cache_key.py
git commit -m "fix(retrieval): include dense_query in result cache key (HyDE collision)"
```

---

### Task 1.2: `SemanticCache` service (embedding cosine, TTL, LRU)

**Files:**
- Create: `src/agentrag/services/semantic_cache.py`
- Test: `tests/services/test_semantic_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/services/test_semantic_cache.py
from src.agentrag.services.semantic_cache import SemanticCache


def test_returns_hit_for_near_identical_embedding():
    cache = SemanticCache(threshold=0.97, ttl_seconds=100, max_items=8, clock=lambda: 0.0)
    cache.put([1.0, 0.0, 0.0], {"results": ["A"]})
    # cosine([1,0,0],[0.999,0.001,0]) ≈ 1.0 ≥ 0.97 → hit
    hit = cache.get([0.999, 0.001, 0.0])
    assert hit == {"results": ["A"]}


def test_miss_for_dissimilar_embedding():
    cache = SemanticCache(threshold=0.97, ttl_seconds=100, max_items=8, clock=lambda: 0.0)
    cache.put([1.0, 0.0, 0.0], {"results": ["A"]})
    assert cache.get([0.0, 1.0, 0.0]) is None


def test_entry_expires_after_ttl():
    now = {"t": 0.0}
    cache = SemanticCache(threshold=0.97, ttl_seconds=10, max_items=8, clock=lambda: now["t"])
    cache.put([1.0, 0.0, 0.0], {"results": ["A"]})
    now["t"] = 11.0
    assert cache.get([1.0, 0.0, 0.0]) is None


def test_lru_eviction_beyond_max_items():
    cache = SemanticCache(threshold=0.99, ttl_seconds=100, max_items=2, clock=lambda: 0.0)
    cache.put([1.0, 0.0], {"results": ["A"]})
    cache.put([0.0, 1.0], {"results": ["B"]})
    cache.put([1.0, 1.0], {"results": ["C"]})  # evicts oldest ([1,0])
    assert cache.get([1.0, 0.0]) is None
    assert cache.get([0.0, 1.0]) == {"results": ["B"]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/services/test_semantic_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agentrag.services.semantic_cache'`

- [ ] **Step 3: Implement `SemanticCache`**

```python
# src/agentrag/services/semantic_cache.py
from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Any, Callable


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SemanticCache:
    """In-process retrieval cache keyed by query embedding similarity.

    Tier-2 cache: complements the exact-key TTLCache in ElasticsearchRetriever.
    A lookup embeds the query, scans recent entries, and returns the cached
    payload of the most-similar entry whose cosine ≥ threshold and not expired.
    LRU-bounded by max_items; TTL-bounded by ttl_seconds. Per-worker (not
    distributed) — conservative threshold keeps false hits negligible.
    """

    def __init__(
        self,
        threshold: float,
        ttl_seconds: int,
        max_items: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._max = max_items
        self._clock = clock
        # key insertion-order → (embedding, payload, stored_at)
        self._store: "OrderedDict[int, tuple[list[float], Any, float]]" = OrderedDict()
        self._next_id = 0

    def get(self, embedding: list[float]) -> Any | None:
        now = self._clock()
        best_key: int | None = None
        best_sim = self._threshold
        for key, (emb, _payload, stored_at) in list(self._store.items()):
            if now - stored_at > self._ttl:
                del self._store[key]
                continue
            sim = _cosine(embedding, emb)
            if sim >= best_sim:
                best_sim = sim
                best_key = key
        if best_key is None:
            return None
        self._store.move_to_end(best_key)  # mark as recently used
        return self._store[best_key][1]

    def put(self, embedding: list[float], payload: Any) -> None:
        self._store[self._next_id] = (embedding, payload, self._clock())
        self._next_id += 1
        while len(self._store) > self._max:
            self._store.popitem(last=False)  # evict oldest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/services/test_semantic_cache.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/services/semantic_cache.py tests/services/test_semantic_cache.py
git commit -m "feat(cache): SemanticCache — embedding-cosine retrieval cache (TTL+LRU)"
```

---

### Task 1.3: Wire `SemanticCache` into the retriever

**Files:**
- Modify: `src/agentrag/retrieval/elasticsearch_retriever.py` (`__init__` ~32-36; `_search_impl` ~84-113)
- Test: `tests/retrieval/test_semantic_cache_wiring.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_semantic_cache_wiring.py
import asyncio
from src.agentrag.retrieval import elasticsearch_retriever as er


def test_second_similar_query_served_from_semantic_cache(monkeypatch):
    monkeypatch.setattr(er.settings, "SEMANTIC_CACHE_ENABLED", True)
    monkeypatch.setattr(er.settings, "RETRIEVAL_RERANK_ENABLED", False)

    r = er.ElasticsearchRetriever.__new__(er.ElasticsearchRetriever)
    r._last_rerank_reason = "not_attempted"
    r._semantic_cache = er.SemanticCache(threshold=0.97, ttl_seconds=100, max_items=8, clock=lambda: 0.0)

    calls = {"n": 0}

    class _FakeEmbedder:
        async def embed(self, texts):  # noqa: ANN001
            return [[1.0, 0.0, 0.0]]

    async def fake_impl(**kwargs):  # noqa: ANN001
        calls["n"] += 1
        return {"results": [{"rank": 1, "content": "X"}], "mode": kwargs["mode"], "top_k": 10}

    r.embedder = _FakeEmbedder()
    r._search_uncached = fake_impl  # type: ignore[assignment]

    out1 = asyncio.run(r.search_cached("q1", mode="hybrid", top_k=10))
    out2 = asyncio.run(r.search_cached("q1 rephrased", mode="hybrid", top_k=10))
    assert out1["results"] == out2["results"]
    assert calls["n"] == 1  # second served from semantic cache
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_semantic_cache_wiring.py -v`
Expected: FAIL with `AttributeError: 'ElasticsearchRetriever' object has no attribute 'search_cached'`

- [ ] **Step 3: Add semantic cache layer**

In `elasticsearch_retriever.py`, add the import near the top (after line 16):

```python
from src.agentrag.services.semantic_cache import SemanticCache
```

In `__init__` (after `self._last_rerank_reason = ...` ~36) add:

```python
        self._semantic_cache = (
            SemanticCache(
                threshold=settings.SEMANTIC_CACHE_THRESHOLD,
                ttl_seconds=settings.SEMANTIC_CACHE_TTL_SECONDS,
                max_items=settings.SEMANTIC_CACHE_MAX_ITEMS,
            )
            if settings.SEMANTIC_CACHE_ENABLED
            else None
        )
```

Add a thin cache wrapper method (place it just above `_search_impl`, ~line 83). It embeds the query once, checks the semantic cache, and only on a miss runs the real search. The existing `_search_impl` is renamed `_search_uncached`:

```python
    async def search_cached(self, **kwargs: Any) -> dict:
        """Semantic-cache wrapper around the per-query search. Only consulted
        for unfiltered, default-scope queries (domain/document filters bypass
        the semantic cache to avoid cross-scope leaks)."""
        if (
            self._semantic_cache is None
            or kwargs.get("filters")
            or kwargs.get("document_title")
        ):
            return await self._search_uncached(**kwargs)
        query = kwargs["query"]
        embed_text = kwargs.get("dense_query") or query
        try:
            q_emb = (await self.embedder.embed([embed_text]))[0]
        except Exception:
            return await self._search_uncached(**kwargs)
        hit = self._semantic_cache.get(q_emb)
        if hit is not None:
            return {**hit, "semantic_cache_hit": True}
        result = await self._search_uncached(**kwargs)
        self._semantic_cache.put(q_emb, result)
        return result
```

Rename the existing method `async def _search_impl(` → `async def _search_uncached(` (line 84), and update the call inside `search()` (line 55) from `self._search_impl(` to `self._search_uncached(` (both the primary call ~55 and the fallback ~70). Then change `search()`'s primary call to route through the cache: replace line 55 `payload = await self._search_impl(` with `payload = await self.search_cached(`.

> Note: the domain-filter fallback inside `search()` (lines 64-81) keeps calling `self._search_uncached(...)` directly — fallback results must never be cached because they relax filters.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retrieval/test_semantic_cache_wiring.py tests/retrieval/test_cache_key.py -v`
Expected: PASS

- [ ] **Step 5: Run the existing retrieval suite for regressions**

Run: `uv run pytest tests/retrieval/ -v`
Expected: PASS (no regressions in `test_federated`, `test_balance_segments`, `test_reranker_local`)

- [ ] **Step 6: Commit**

```bash
git add src/agentrag/retrieval/elasticsearch_retriever.py tests/retrieval/test_semantic_cache_wiring.py
git commit -m "feat(retrieval): semantic-cache wrapper for unfiltered queries"
```

---

## PHASE 2 — WS1: Contextual Retrieval

### Task 2.1: `Contextualizer` service (LLM situate + file cache)

**Files:**
- Create: `src/agentrag/ingestion/contextualizer.py`
- Test: `tests/ingestion/test_contextualizer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_contextualizer.py
import asyncio
from src.agentrag.ingestion.contextualizer import Contextualizer


class _FakeGateway:
    def __init__(self):
        self.calls = 0

    async def text_response(self, system_prompt, user_prompt, task="general"):  # noqa: ANN001
        self.calls += 1
        return f"CTX for: {user_prompt[:20]}"


def test_contextualize_sets_context_text_on_each_chunk(tmp_path):
    gw = _FakeGateway()
    ctx = Contextualizer(gw, cache_dir=str(tmp_path))
    chunks = [
        {"content": "Nhồi máu cơ tim là ...", "content_hash": "h1"},
        {"content": "Điều trị bằng aspirin ...", "content_hash": "h2"},
    ]
    out = asyncio.run(ctx.contextualize_chunks("Whole doc text", chunks, "Tim mạch"))
    assert all(c.get("context_text") for c in out)
    assert gw.calls == 2


def test_contextualize_uses_cache_on_second_run(tmp_path):
    gw = _FakeGateway()
    ctx = Contextualizer(gw, cache_dir=str(tmp_path))
    chunks = [{"content": "x", "content_hash": "h1"}]
    asyncio.run(ctx.contextualize_chunks("doc", chunks, "T"))
    first_calls = gw.calls
    # Fresh chunk dict, same hash → served from disk cache, no new LLM call.
    chunks2 = [{"content": "x", "content_hash": "h1"}]
    asyncio.run(ctx.contextualize_chunks("doc", chunks2, "T"))
    assert gw.calls == first_calls
    assert chunks2[0]["context_text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_contextualizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agentrag.ingestion.contextualizer'`

- [ ] **Step 3: Implement `Contextualizer`**

```python
# src/agentrag/ingestion/contextualizer.py
from __future__ import annotations

import asyncio
import logging
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agentrag.config import settings

if TYPE_CHECKING:
    from src.agentrag.services.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You situate a short passage within its source document for retrieval.
You are given the whole document, then one passage from it.
Write a single concise sentence (max ~80 tokens, same language as the passage)
that states what section/topic the passage belongs to and what it is about, so
the passage can be found by search even out of context. Output ONLY that
sentence, no preamble, no quotes."""


class Contextualizer:
    """WS1 — generate a situating context sentence per chunk.

    The whole document goes in the system prompt (a stable prefix that the
    provider's context cache, e.g. DeepSeek, reuses across the document's
    chunks); only the per-chunk passage varies. Results are file-cached keyed
    by (provider_signature, doc_hash, chunk_hash) so backfill is idempotent.
    """

    def __init__(self, gateway: "LLMGateway", cache_dir: str | None = None) -> None:
        self._gateway = gateway
        self._cache_dir = Path(cache_dir or settings.CONTEXTUAL_CACHE_DIR)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._sig = sha256(
            f"contextual_v1|{settings.CONTEXTUAL_RETRIEVAL_TASK}".encode("utf-8")
        ).hexdigest()[:12]

    async def contextualize_chunks(
        self, doc_text: str, chunks: list[dict[str, Any]], document_title: str
    ) -> list[dict[str, Any]]:
        if not chunks:
            return chunks
        doc_clip = doc_text[: settings.CONTEXTUAL_MAX_DOC_CHARS]
        doc_hash = sha256(doc_clip.encode("utf-8")).hexdigest()
        system = f"{_SYSTEM_PROMPT}\n\n<document title=\"{document_title}\">\n{doc_clip}\n</document>"

        sem = asyncio.Semaphore(max(settings.EMBEDDING_BATCH_SIZE // 4, 4))

        async def one(chunk: dict[str, Any]) -> None:
            chunk_hash = chunk.get("content_hash") or sha256(
                chunk["content"].encode("utf-8")
            ).hexdigest()
            cached = self._load(doc_hash, chunk_hash)
            if cached is not None:
                chunk["context_text"] = cached
                return
            async with sem:
                try:
                    text = await self._gateway.text_response(
                        system_prompt=system,
                        user_prompt=f"Passage:\n{chunk['content']}",
                        task=settings.CONTEXTUAL_RETRIEVAL_TASK,
                    )
                except Exception as exc:
                    logger.warning("contextualize failed (%s): %s", document_title, exc)
                    text = ""
            text = (text or "").strip()
            chunk["context_text"] = text or None
            if text:
                self._store(doc_hash, chunk_hash, text)

        await asyncio.gather(*(one(c) for c in chunks))
        return chunks

    def _path(self, doc_hash: str, chunk_hash: str) -> Path:
        key = sha256(f"{self._sig}|{doc_hash}|{chunk_hash}".encode("utf-8")).hexdigest()
        return self._cache_dir / f"{key}.txt"

    def _load(self, doc_hash: str, chunk_hash: str) -> str | None:
        p = self._path(doc_hash, chunk_hash)
        if not p.exists():
            return None
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None

    def _store(self, doc_hash: str, chunk_hash: str, text: str) -> None:
        try:
            self._path(doc_hash, chunk_hash).write_text(text, encoding="utf-8")
        except Exception:
            pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_contextualizer.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/ingestion/contextualizer.py tests/ingestion/test_contextualizer.py
git commit -m "feat(ingest): Contextualizer — LLM situating context per chunk (cached)"
```

---

### Task 2.2: Embed `context_text + content`; wire contextualize into pipeline

**Files:**
- Modify: `src/agentrag/ingestion/pipeline.py` (after tagging ~226, before embed ~227-232)
- Test: `tests/ingestion/test_pipeline_embed_input.py`

- [ ] **Step 1: Write the failing test (the embed-input helper)**

Add a small pure helper so the embed-input rule is unit-testable without running the whole pipeline.

```python
# tests/ingestion/test_pipeline_embed_input.py
from src.agentrag.ingestion.pipeline import _embed_input_for_chunk


def test_embed_input_prepends_context_when_present():
    c = {"content": "leaf body", "context_text": "From cardiology chapter."}
    assert _embed_input_for_chunk(c) == "From cardiology chapter.\n\nleaf body"


def test_embed_input_is_content_only_when_no_context():
    c = {"content": "leaf body", "context_text": None}
    assert _embed_input_for_chunk(c) == "leaf body"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_pipeline_embed_input.py -v`
Expected: FAIL with `ImportError: cannot import name '_embed_input_for_chunk'`

- [ ] **Step 3: Add helper + contextualize step**

In `pipeline.py`, add the helper near the top (after the `_AUDIO_SOURCE_TYPES` block ~38):

```python
def _embed_input_for_chunk(chunk: dict[str, Any]) -> str:
    """Text to embed/BM25: contextualized when WS1 produced a context_text,
    else the raw content. The original `content` is always what gets cited."""
    ctx = chunk.get("context_text")
    if ctx:
        return f"{ctx}\n\n{chunk['content']}"
    return chunk["content"]
```

Insert the contextualize step **after** the tagging block (after line 225, before the embed block at 227). It reuses the same `LLMGateway` already imported lazily for vision:

```python
            # WS1 — Contextual Retrieval: add a situating context sentence per
            # chunk BEFORE embedding so dense + BM25 see the contextualized text.
            if settings.CONTEXTUAL_RETRIEVAL_ENABLED:
                t0 = time.perf_counter()
                from src.agentrag.ingestion.contextualizer import Contextualizer
                from src.agentrag.services.llm_gateway import LLMGateway
                chunks_search = await Contextualizer(LLMGateway()).contextualize_chunks(
                    doc_text=content, chunks=chunks_search, document_title=doc["title"]
                )
                timings["contextualize_ms"] = (time.perf_counter() - t0) * 1000
```

Change the embed block (lines 228-229) from embedding raw content to embedding the contextualized input:

```python
            texts = [_embed_input_for_chunk(c) for c in chunks_search]
            embeddings = await embedder.embed(texts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_pipeline_embed_input.py -v`
Expected: PASS

- [ ] **Step 5: Run ingestion suite for regressions**

Run: `uv run pytest tests/ingestion/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/agentrag/ingestion/pipeline.py tests/ingestion/test_pipeline_embed_input.py
git commit -m "feat(ingest): embed contextualized text (WS1) behind CONTEXTUAL_RETRIEVAL_ENABLED"
```

---

## PHASE 3 — WS2: RAPTOR summary layer

### Task 3.1: Add clustering dependencies

**Files:**
- Modify: `pyproject.toml` (`[project] dependencies`)

- [ ] **Step 1: Add deps**

Add to the `dependencies` array in `pyproject.toml`:

```toml
    "umap-learn>=0.5.6",
    "scikit-learn>=1.5.0",
```

- [ ] **Step 2: Sync and verify import**

Run: `uv sync && uv run python -c "import umap, sklearn.mixture; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add umap-learn + scikit-learn for RAPTOR clustering"
```

---

### Task 3.2: `RaptorBuilder` — recursive cluster + summarize

**Files:**
- Create: `src/agentrag/ingestion/raptor.py`
- Test: `tests/ingestion/test_raptor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_raptor.py
import asyncio
from src.agentrag.ingestion.raptor import RaptorBuilder


class _FakeGateway:
    async def text_response(self, system_prompt, user_prompt, task="general"):  # noqa: ANN001
        return "SUMMARY"


class _FakeEmbedder:
    async def embed(self, texts):  # noqa: ANN001
        return [[float(len(t) % 7), 1.0, 0.0] for t in texts]


def _leaves(n):
    return [
        {"content": f"leaf {i} about topic {i % 3}", "content_hash": f"h{i}",
         "embedding": [float(i % 3), 1.0, 0.0], "system_tag": "tim_mach",
         "specialty_tag": ["noi"], "node_level": 0}
        for i in range(n)
    ]


def test_skips_when_too_few_leaves(monkeypatch):
    from src.agentrag.ingestion import raptor as R
    monkeypatch.setattr(R.settings, "RAPTOR_MIN_LEAVES", 8)
    builder = RaptorBuilder(_FakeGateway(), _FakeEmbedder())
    out = asyncio.run(builder.build(_leaves(5), "Doc"))
    assert out == []  # no summary nodes for tiny docs


def test_builds_summary_nodes_with_level_and_children(monkeypatch):
    from src.agentrag.ingestion import raptor as R
    monkeypatch.setattr(R.settings, "RAPTOR_MIN_LEAVES", 4)
    monkeypatch.setattr(R.settings, "RAPTOR_MAX_LEVELS", 2)
    monkeypatch.setattr(R.settings, "RAPTOR_CLUSTER_SIZE", 3)
    builder = RaptorBuilder(_FakeGateway(), _FakeEmbedder())
    out = asyncio.run(builder.build(_leaves(12), "Doc"))
    assert out, "expected at least one summary node"
    for node in out:
        assert node["node_level"] >= 1
        assert node["segment_type"] == "raptor_summary"
        assert node["child_ids"]                 # links to children
        assert node["embedding"]                  # embedded
        assert node["content"] == "SUMMARY"
        assert node["system_tag"] == "tim_mach"   # propagated from children
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_raptor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agentrag.ingestion.raptor'`

- [ ] **Step 3: Implement `RaptorBuilder`**

```python
# src/agentrag/ingestion/raptor.py
from __future__ import annotations

import logging
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from src.agentrag.config import settings

if TYPE_CHECKING:
    from src.agentrag.services.llm_gateway import LLMGateway
    from src.agentrag.ingestion.embedders.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM = """\
You write a faithful, self-contained summary of several related passages from a
medical document. Capture the shared topic and the key facts/relationships.
Same language as the passages. Output ONLY the summary text, no preamble."""


def _cluster_indices(vectors: list[list[float]], n_clusters: int) -> list[list[int]]:
    """Return groups of row indices via UMAP→GaussianMixture hard assignment.
    Falls back to contiguous chunking if reduction/fit fails or n is tiny."""
    n = len(vectors)
    if n_clusters <= 1 or n <= n_clusters:
        return [list(range(n))]
    try:
        import numpy as np
        import umap
        from sklearn.mixture import GaussianMixture

        arr = np.asarray(vectors, dtype="float32")
        n_components = min(10, max(2, n - 2))
        reduced = umap.UMAP(
            n_neighbors=min(15, n - 1), n_components=n_components, metric="cosine",
        ).fit_transform(arr)
        gm = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = gm.fit_predict(reduced)
        groups: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            groups.setdefault(int(lab), []).append(idx)
        return [g for g in groups.values() if g]
    except Exception as exc:  # numerical / convergence issues → contiguous split
        logger.warning("RAPTOR clustering fell back to contiguous: %s", exc)
        size = max(1, n // n_clusters)
        return [list(range(i, min(i + size, n))) for i in range(0, n, size)]


class RaptorBuilder:
    """WS2 — build a collapsed RAPTOR tree: recursively cluster node embeddings,
    summarize each cluster, embed the summary, and emit summary nodes carrying
    `node_level`, `child_ids`, and domain tags propagated (union) from children.
    Returned nodes are appended to the same `agentrag_segments` index."""

    def __init__(self, gateway: "LLMGateway", embedder: "BaseEmbeddingProvider") -> None:
        self._gateway = gateway
        self._embedder = embedder

    async def build(
        self, leaf_chunks: list[dict[str, Any]], document_title: str
    ) -> list[dict[str, Any]]:
        if len(leaf_chunks) < settings.RAPTOR_MIN_LEAVES:
            return []
        summary_nodes: list[dict[str, Any]] = []
        current = leaf_chunks
        for level in range(1, settings.RAPTOR_MAX_LEVELS + 1):
            vectors = [c.get("embedding") for c in current]
            if any(v is None for v in vectors):
                break
            n_clusters = max(2, len(current) // max(settings.RAPTOR_CLUSTER_SIZE, 2))
            groups = _cluster_indices(vectors, n_clusters)
            if len(groups) >= len(current):  # no compression → stop
                break
            level_nodes: list[dict[str, Any]] = []
            for group in groups:
                members = [current[i] for i in group]
                node = await self._summarize_group(members, document_title, level)
                if node is not None:
                    level_nodes.append(node)
            if not level_nodes:
                break
            # Embed this level's summaries so the next level can cluster them.
            embeddings = await self._embedder.embed([n["content"] for n in level_nodes])
            for node, emb in zip(level_nodes, embeddings):
                node["embedding"] = emb
            summary_nodes.extend(level_nodes)
            current = level_nodes
            if len(current) <= 1:  # reached root
                break
        return summary_nodes

    async def _summarize_group(
        self, members: list[dict[str, Any]], document_title: str, level: int
    ) -> dict[str, Any] | None:
        joined = "\n\n---\n\n".join(m["content"] for m in members)
        try:
            summary = await self._gateway.text_response(
                system_prompt=_SUMMARY_SYSTEM,
                user_prompt=f"Document: {document_title}\n\nPassages:\n{joined}",
                task=settings.RAPTOR_SUMMARY_TASK,
            )
        except Exception as exc:
            logger.warning("RAPTOR summary failed (%s): %s", document_title, exc)
            return None
        summary = (summary or "").strip()
        if not summary:
            return None
        systems = {m.get("system_tag") for m in members if m.get("system_tag")}
        specialties: set[str] = set()
        for m in members:
            specialties.update(m.get("specialty_tag") or [])
        child_ids = [m["content_hash"] for m in members if m.get("content_hash")]
        return {
            "content": summary,
            "content_hash": sha256(summary.encode("utf-8")).hexdigest(),
            "segment_type": "raptor_summary",
            "node_level": level,
            "child_ids": child_ids,
            "section_path": f"{document_title} / summary L{level}",
            "position": None,
            "system_tag": next(iter(systems), None),
            "specialty_tag": sorted(specialties),
            "canonical_terms": [],
            "metadata": {"document_title": document_title, "raptor_level": level},
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_raptor.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/ingestion/raptor.py tests/ingestion/test_raptor.py
git commit -m "feat(ingest): RaptorBuilder — recursive cluster+summarize summary layer"
```

---

### Task 3.3: Wire RAPTOR into pipeline (after leaf index)

**Files:**
- Modify: `src/agentrag/ingestion/pipeline.py` (after `es_store.index_segments(...)` ~247-250)
- Test: `tests/ingestion/test_pipeline_raptor_step.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_pipeline_raptor_step.py
import asyncio
from src.agentrag.ingestion.pipeline import _build_and_index_raptor


class _Store:
    def __init__(self):
        self.indexed = None

    async def index_segments(self, chunks, title):  # noqa: ANN001
        self.indexed = chunks


class _Builder:
    async def build(self, leaves, title):  # noqa: ANN001
        return [{"content": "S", "content_hash": "s1", "embedding": [1.0],
                 "node_level": 1, "segment_type": "raptor_summary"}]


def test_indexes_summary_nodes_when_builder_returns_them():
    store = _Store()
    leaves = [{"content": "x", "embedding": [1.0]}]
    asyncio.run(_build_and_index_raptor(_Builder(), store, leaves, "Doc"))
    assert store.indexed and store.indexed[0]["segment_type"] == "raptor_summary"


def test_no_index_when_builder_returns_empty():
    store = _Store()

    class _Empty:
        async def build(self, leaves, title):  # noqa: ANN001
            return []

    asyncio.run(_build_and_index_raptor(_Empty(), store, [{"content": "x"}], "Doc"))
    assert store.indexed is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_pipeline_raptor_step.py -v`
Expected: FAIL with `ImportError: cannot import name '_build_and_index_raptor'`

- [ ] **Step 3: Add helper + pipeline hook**

In `pipeline.py`, add the helper near `_embed_input_for_chunk`:

```python
async def _build_and_index_raptor(builder: Any, es_store: Any, leaf_chunks: list[dict[str, Any]], document_title: str) -> int:
    """Build RAPTOR summary nodes from leaves and index them. Returns count."""
    summary_nodes = await builder.build(leaf_chunks, document_title)
    if not summary_nodes:
        return 0
    await es_store.index_segments(summary_nodes, document_title)
    return len(summary_nodes)
```

Insert the RAPTOR step right after the leaf `index_segments` call (after line 250, inside the `if status != "retry":` block):

```python
                if settings.RAPTOR_ENABLED:
                    t0 = time.perf_counter()
                    from src.agentrag.ingestion.raptor import RaptorBuilder
                    from src.agentrag.services.llm_gateway import LLMGateway
                    raptor_count = await _build_and_index_raptor(
                        RaptorBuilder(LLMGateway(), embedder),
                        es_store, chunks_search, doc["title"],
                    )
                    timings["raptor_ms"] = (time.perf_counter() - t0) * 1000
                    report["raptor_summary_nodes"] = raptor_count
```

> Note: `chunks_search` already carry `embedding` (set at line 230-231) when RAPTOR runs, so `RaptorBuilder.build` can cluster them directly.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_pipeline_raptor_step.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/ingestion/pipeline.py tests/ingestion/test_pipeline_raptor_step.py
git commit -m "feat(ingest): index RAPTOR summary nodes after leaves (WS2)"
```

---

### Task 3.4: Cap RAPTOR summary share in results

**Files:**
- Modify: `src/agentrag/retrieval/elasticsearch_retriever.py` (add `_cap_summary_nodes`, call in each mode before `_finalize_ranks`)
- Test: `tests/retrieval/test_summary_cap.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_summary_cap.py
from src.agentrag.retrieval import elasticsearch_retriever as er


def test_summary_nodes_capped_by_ratio(monkeypatch):
    monkeypatch.setattr(er.settings, "RAPTOR_SUMMARY_MAX_RATIO", 0.4)
    r = er.ElasticsearchRetriever.__new__(er.ElasticsearchRetriever)
    hits = [
        {"node_level": 1, "content": "s1"}, {"node_level": 1, "content": "s2"},
        {"node_level": 1, "content": "s3"}, {"node_level": 0, "content": "l1"},
        {"node_level": 0, "content": "l2"},
    ]
    out = r._cap_summary_nodes(hits, size=5)
    n_summary = sum(1 for h in out if h.get("node_level", 0) >= 1)
    assert n_summary <= 2  # floor(0.4 * 5)
    assert len(out) == 5   # leaves backfill the dropped summaries
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_summary_cap.py -v`
Expected: FAIL with `AttributeError: ... has no attribute '_cap_summary_nodes'`

- [ ] **Step 3: Implement the cap**

Add this method to `ElasticsearchRetriever` (near `_balance_segment_types_for_query`):

```python
    def _cap_summary_nodes(self, hits: list[dict], size: int) -> list[dict]:
        """Keep RAPTOR summary nodes (node_level>=1) to at most
        RAPTOR_SUMMARY_MAX_RATIO of the result set so a query can't return only
        summaries; leaves backfill the freed slots. Preserves order."""
        max_summary = int(settings.RAPTOR_SUMMARY_MAX_RATIO * size)
        kept: list[dict] = []
        summary_seen = 0
        overflow: list[dict] = []
        for h in hits:
            if h.get("node_level", 0) >= 1:
                if summary_seen < max_summary:
                    kept.append(h)
                    summary_seen += 1
                else:
                    overflow.append(h)
            else:
                kept.append(h)
        kept.extend(overflow)  # demoted summaries go last (backfill)
        return kept[: size if size else len(kept)]
```

Call it in `_search_impl`/`_search_uncached` right before each `_finalize_ranks(hits)` call (sparse ~131, dense ~163, hybrid ~239):

```python
            hits = self._cap_summary_nodes(hits, size)
            hits = self._finalize_ranks(hits)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retrieval/test_summary_cap.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/retrieval/elasticsearch_retriever.py tests/retrieval/test_summary_cap.py
git commit -m "feat(retrieval): cap RAPTOR summary share in result set"
```

---

## PHASE 4 — WS4: Adaptive routing

### Task 4.1: Add `complexity` + `single_domain` to the classifier

**Files:**
- Modify: `src/agentrag/structured/query_classifier.py` (`ClassifierOutput` ~17-23; `_classify_l1` ~173-188; default returns; `_classify_l2` schema ~208-237)
- Test: `tests/orchestration/test_classifier_complexity.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestration/test_classifier_complexity.py
import asyncio
from src.agentrag.structured.query_classifier import QueryIntentClassifier, ClassifierOutput


def test_short_factual_question_is_simple():
    c = QueryIntentClassifier()
    out = asyncio.run(c.classify("Triệu chứng của nhồi máu cơ tim là gì?"))
    assert isinstance(out, ClassifierOutput)
    assert out.complexity == "simple"
    assert out.single_domain is True


def test_comparison_question_is_complex():
    c = QueryIntentClassifier()
    out = asyncio.run(c.classify("So sánh nhồi máu cơ tim và đột quỵ về cơ chế và điều trị"))
    # structured comparison → complex, and spans >1 domain
    assert out.complexity == "complex"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/orchestration/test_classifier_complexity.py -v`
Expected: FAIL with `AttributeError: 'ClassifierOutput' object has no attribute 'complexity'`

- [ ] **Step 3: Extend `ClassifierOutput` + rule heuristics**

Replace the `ClassifierOutput` dataclass (lines 17-23):

```python
@dataclass
class ClassifierOutput:
    intent: QueryIntent
    query_type: QueryType | None
    confidence: float
    reasoning: str
    method: Literal["rule", "llm", "default"]
    complexity: Literal["simple", "complex"] = "simple"
    single_domain: bool = True
```

Add a heuristic helper inside `QueryIntentClassifier` (after `_classify_l1`):

```python
    # Markers that a question needs multi-step / multi-domain reasoning.
    _COMPLEX_MARKERS: list[re.Pattern] = [
        re.compile(p, _FLAG) for p in [
            r"\bso sánh\b", r"\bcompare\b", r"\bvà\b.{0,40}\bkhác\b",
            r"\btại sao\b", r"\bvì sao\b", r"\bwhy\b", r"\bcơ chế\b",
            r"\bmối liên hệ\b", r"\brelationship\b", r"\bphân tích\b",
            r"\bđánh giá\b", r"\btổng hợp\b", r"\bnhiều\b.{0,20}\bkhía cạnh\b",
        ]
    ]

    def _estimate_complexity(self, question: str, intent: ClassifierOutput | None = None) -> tuple[str, bool]:
        q = question.strip()
        long_q = len(q) >= settings.AGENT_PLAN_TRIGGER_MIN_CHARS
        has_marker = any(p.search(q) for p in self._COMPLEX_MARKERS)
        multi_clause = (" và " in f" {q.lower()} ") or (";" in q) or (q.count("?") > 1)
        complex_ = bool(has_marker or (long_q and multi_clause))
        single_domain = not (has_marker or multi_clause)
        return ("complex" if complex_ else "simple"), single_domain
```

In `_classify_l1`, before returning the structured match, compute and attach complexity:

```python
        for query_type, patterns in self._PATTERN_MAP:
            for pattern in patterns:
                if pattern.search(question):
                    complexity, single_domain = self._estimate_complexity(question)
                    # Any structured query_type is treated as at least 'complex'
                    # except a bare single-fact aggregation.
                    return ClassifierOutput(
                        intent="structured",
                        query_type=query_type,
                        confidence=0.95,
                        reasoning=f"L1 rule matched pattern for query_type='{query_type}'",
                        method="rule",
                        complexity="complex" if query_type != "aggregation" else complexity,
                        single_domain=single_domain,
                    )
        return None
```

In the two semantic default `ClassifierOutput(...)` returns inside `classify` (the no-rule default ~165-171 and any other default), and in `_classify_l2`'s success return, set complexity from the heuristic. For the default-in-`classify` return:

```python
        complexity, single_domain = self._classify_l1_complexity(question)
        return ClassifierOutput(
            intent="semantic",
            query_type=None,
            confidence=0.5,
            reasoning="No rule matched; defaulting to semantic path.",
            method="default",
            complexity=complexity,
            single_domain=single_domain,
        )
```

Add the tiny wrapper used above (so semantic-path questions get complexity too):

```python
    def _classify_l1_complexity(self, question: str) -> tuple[str, bool]:
        return self._estimate_complexity(question)
```

In `_classify_l2`, extend the JSON schema instruction and the returned object. Change the schema line in `system_prompt` to:

```python
            "Return JSON: {\"intent\": str, \"query_type\": str|null, \"confidence\": float, \"complexity\": \"simple\"|\"complex\", \"single_domain\": bool, \"reasoning\": str}\n"
```

and the success return:

```python
            complexity = result.get("complexity")
            if complexity not in ("simple", "complex"):
                complexity, _sd = self._estimate_complexity(question)
            return ClassifierOutput(
                intent=intent,
                query_type=query_type,
                confidence=float(result.get("confidence", 0.7)),
                reasoning=result.get("reasoning", ""),
                method="llm",
                complexity=complexity,
                single_domain=bool(result.get("single_domain", True)),
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/orchestration/test_classifier_complexity.py tests/orchestration/test_domain_router.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/structured/query_classifier.py tests/orchestration/test_classifier_complexity.py
git commit -m "feat(classify): emit complexity + single_domain for adaptive routing"
```

---

### Task 4.2: Fast-path node + route in the graph

**Files:**
- Modify: `src/agentrag/agent/graph_service.py` (`classify` ~122-133; add `fast_answer` node + `_route_intent` change; `_build_graph` ~389-423)
- Test: `tests/agent/test_adaptive_routing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_adaptive_routing.py
from src.agentrag.agent import graph_service as gs
from src.agentrag.structured.query_classifier import ClassifierOutput


def _co(intent="semantic", complexity="simple", single_domain=True, conf=0.95):
    return ClassifierOutput(intent=intent, query_type=None, confidence=conf,
                            reasoning="", method="rule", complexity=complexity,
                            single_domain=single_domain)


def test_route_takes_fast_path_for_simple_single_domain(monkeypatch):
    monkeypatch.setattr(gs.settings, "ADAPTIVE_ROUTING_ENABLED", True)
    monkeypatch.setattr(gs.settings, "ADAPTIVE_FASTPATH_MIN_CONFIDENCE", 0.85)
    state = {"intent": "semantic", "classifier_output": _co()}
    assert gs._route_intent(state) == "fast_answer"


def test_route_takes_full_path_for_complex(monkeypatch):
    monkeypatch.setattr(gs.settings, "ADAPTIVE_ROUTING_ENABLED", True)
    state = {"intent": "semantic", "classifier_output": _co(complexity="complex")}
    assert gs._route_intent(state) == "semantic_plan"


def test_route_full_path_when_flag_off(monkeypatch):
    monkeypatch.setattr(gs.settings, "ADAPTIVE_ROUTING_ENABLED", False)
    state = {"intent": "semantic", "classifier_output": _co()}
    assert gs._route_intent(state) == "semantic_plan"


def test_structured_intent_still_routes_structured(monkeypatch):
    monkeypatch.setattr(gs.settings, "ADAPTIVE_ROUTING_ENABLED", True)
    state = {"intent": "structured", "classifier_output": _co(intent="structured", complexity="complex")}
    assert gs._route_intent(state) == "structured_run"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agent/test_adaptive_routing.py -v`
Expected: FAIL (`_route_intent` returns `"semantic_plan"`, never `"fast_answer"`)

- [ ] **Step 3: Add fast-path routing + node**

Replace `_route_intent` (lines 368-369):

```python
def _route_intent(state: ChatState) -> str:
    if state.get("intent") == "structured":
        return "structured_run"
    co = state.get("classifier_output")
    if (
        settings.ADAPTIVE_ROUTING_ENABLED
        and co is not None
        and getattr(co, "complexity", "complex") == "simple"
        and getattr(co, "single_domain", False)
        and getattr(co, "confidence", 0.0) >= settings.ADAPTIVE_FASTPATH_MIN_CONFIDENCE
    ):
        return "fast_answer"
    return "semantic_plan"
```

Add the `fast_answer` node (after `bootstrap`, ~line 232). It does one bootstrap retrieve + assemble + single-shot answer + ground, with no plan/decide/tool loop:

```python
async def fast_answer(state: ChatState) -> dict[str, Any]:
    """Adaptive fast path: single retrieve + single-shot answer, skipping the
    plan→decide→tool loop. Used only for high-confidence simple single-domain
    questions (WS4)."""
    doc_title = state.get("document_title")
    co = state.get("classifier_output")
    boot_in, boot_out = await _INNER.knowledge.bootstrap_search(
        query=state["question"], document_title=doc_title, intent=co,
    )
    boot_out = _INNER.security.filter_tool_results(tool_output=boot_out, document_title=doc_title)
    trace = [{"tool_name": "search_hybrid_kg", "tool_input": boot_in, "tool_output": boot_out}]

    assembled = await _INNER.context.assemble(state["question"], [boot_out])
    packed = assembled.get("packed_context", []) if isinstance(assembled, dict) else assembled

    started = time.perf_counter()
    out = await _INNER._answer(
        question=state["question"], packed_context=packed, tool_trace=trace,
        final_answer=None, chat_history=state.get("chat_history"),
        memory_context=state.get("memory_context"), verbosity=state.get("verbosity"),
    )
    return {
        "tool_trace": trace,
        "packed_context": packed,
        "answer": out.get("answer", ""),
        "citations": out.get("citations", []),
        "highlights": out.get("highlights", []),
        "answer_latency_ms": (time.perf_counter() - started) * 1000,
        "reasoning_path": "fast",
    }
```

In `_build_graph`, register the node and wire its edges. After `g.add_node("bootstrap", bootstrap)` add:

```python
    g.add_node("fast_answer", fast_answer)
```

Add `"fast_answer"` to the `classify` conditional-edge mapping (line 411-412):

```python
    g.add_conditional_edges("classify", _route_intent,
                            {"structured_run": "structured_run",
                             "semantic_plan": "semantic_plan",
                             "fast_answer": "fast_answer"})
```

And route `fast_answer` straight to `ground` (so citations are built and timings exported). After `g.add_edge("answer", "ground")` add:

```python
    g.add_edge("fast_answer", "ground")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agent/test_adaptive_routing.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the agent + integration suites for regressions**

Run: `uv run pytest tests/agent/ tests/integration/test_s4_planes.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/agentrag/agent/graph_service.py tests/agent/test_adaptive_routing.py
git commit -m "feat(agent): adaptive fast-path for simple single-domain queries (WS4)"
```

---

## PHASE 5 — WS3: CRAG critique + multi-hop

### Task 5.1: `_critique` on `AgentService`

**Files:**
- Modify: `src/agentrag/agent/service.py` (add `_critique` method + module helper `_has_uncertainty`)
- Test: `tests/agent/test_critique.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_critique.py
from src.agentrag.agent.service import _has_uncertainty, AgentService


def test_uncertainty_phrases_detected():
    assert _has_uncertainty("Xin lỗi, tôi không tìm thấy thông tin về điều này.")
    assert _has_uncertainty("I don't have enough information to answer.")
    assert not _has_uncertainty("Nhồi máu cơ tim là tình trạng tắc nghẽn mạch vành.")


def test_critique_flags_when_no_citations():
    svc = AgentService.__new__(AgentService)
    decision = svc._critique(
        answer="Nhồi máu cơ tim là ...",
        citations=[],                      # nothing grounded
        packed_context=[{"content": "..."}],
    )
    assert decision["grounded"] is False
    assert decision["reason"] == "no_citations"


def test_critique_flags_on_too_few_hits():
    svc = AgentService.__new__(AgentService)
    decision = svc._critique(
        answer="Câu trả lời.",
        citations=[{"source": 1}],
        packed_context=[],                 # retrieval returned nothing
    )
    assert decision["grounded"] is False
    assert decision["reason"] == "insufficient_context"


def test_critique_passes_for_grounded_answer():
    svc = AgentService.__new__(AgentService)
    decision = svc._critique(
        answer="Nhồi máu cơ tim là tắc động mạch vành [1].",
        citations=[{"source": 1}],
        packed_context=[{"content": "..."}, {"content": "..."}],
    )
    assert decision["grounded"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agent/test_critique.py -v`
Expected: FAIL with `ImportError: cannot import name '_has_uncertainty'`

- [ ] **Step 3: Implement `_has_uncertainty` + `_critique`**

Add the module-level helper near the other module helpers in `service.py` (e.g. beside `_is_chitchat`):

```python
_UNCERTAINTY_MARKERS = (
    "không tìm thấy", "không có thông tin", "không đủ thông tin",
    "không thể trả lời", "tôi không biết", "chưa có dữ liệu",
    "i don't have", "i do not have", "no information", "cannot answer",
    "not enough information", "unable to answer",
)


def _has_uncertainty(answer: str) -> bool:
    low = (answer or "").lower()
    return any(m in low for m in _UNCERTAINTY_MARKERS)
```

Add the `_critique` method on `AgentService`:

```python
    def _critique(
        self,
        answer: str,
        citations: list[dict[str, Any]],
        packed_context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """CRAG relevance + grounding check (no extra LLM call).

        Relevance: retrieval must have returned at least CRAG_MIN_HITS passages.
        Grounding: the answer must cite a source and not be an uncertainty
        ('không tìm thấy …') response. Returns {grounded: bool, reason: str}.
        """
        if len(packed_context) < settings.CRAG_MIN_HITS:
            return {"grounded": False, "reason": "insufficient_context"}
        if settings.CRAG_GROUNDING_ENABLED:
            if not citations:
                return {"grounded": False, "reason": "no_citations"}
            if _has_uncertainty(answer):
                return {"grounded": False, "reason": "uncertain_answer"}
        return {"grounded": True, "reason": "ok"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agent/test_critique.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/agent/service.py tests/agent/test_critique.py
git commit -m "feat(agent): CRAG critique — relevance + grounding check (WS3)"
```

---

### Task 5.2: Critique node + corrective re-retrieve edge

**Files:**
- Modify: `src/agentrag/agent/graph_service.py` (add `critique` + `corrective_retrieve` nodes; edges; state fields)
- Test: `tests/agent/test_critique_routing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_critique_routing.py
from src.agentrag.agent import graph_service as gs


def test_route_critique_to_corrective_when_ungrounded(monkeypatch):
    monkeypatch.setattr(gs.settings, "CRAG_ENABLED", True)
    monkeypatch.setattr(gs.settings, "AGENT_CRITIQUE_MAX_RETRIES", 1)
    state = {"critique_decision": {"grounded": False, "reason": "no_citations"},
             "critique_retries": 0}
    assert gs._route_critique(state) == "corrective_retrieve"


def test_route_critique_to_end_when_grounded(monkeypatch):
    monkeypatch.setattr(gs.settings, "CRAG_ENABLED", True)
    state = {"critique_decision": {"grounded": True}, "critique_retries": 0}
    assert gs._route_critique(state) == "ground"


def test_route_critique_stops_after_max_retries(monkeypatch):
    monkeypatch.setattr(gs.settings, "CRAG_ENABLED", True)
    monkeypatch.setattr(gs.settings, "AGENT_CRITIQUE_MAX_RETRIES", 1)
    state = {"critique_decision": {"grounded": False}, "critique_retries": 1}
    assert gs._route_critique(state) == "ground"  # give up, return best effort


def test_route_critique_disabled_goes_straight_to_ground(monkeypatch):
    monkeypatch.setattr(gs.settings, "CRAG_ENABLED", False)
    state = {"critique_decision": {"grounded": False}, "critique_retries": 0}
    assert gs._route_critique(state) == "ground"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agent/test_critique_routing.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_route_critique'`

- [ ] **Step 3: Add critique state, nodes, router, edges**

Add to `ChatState` (after `packed_context` ~64):

```python
    critique_decision: Optional[dict[str, Any]]
    critique_retries: int
```

Add the `critique` and `corrective_retrieve` nodes (after `answer_node`, ~line 324):

```python
async def critique(state: ChatState) -> dict[str, Any]:
    if not settings.CRAG_ENABLED:
        return {"critique_decision": {"grounded": True, "reason": "disabled"}}
    decision = _INNER._critique(
        answer=state.get("answer", ""),
        citations=state.get("citations", []),
        packed_context=state.get("packed_context", []),
    )
    return {"critique_decision": decision}


async def corrective_retrieve(state: ChatState) -> dict[str, Any]:
    """One bounded CRAG correction: step-back query rewrite + re-retrieve +
    re-answer. Appends new evidence to the trace, then loops to critique."""
    doc_title = state.get("document_title")
    co = state.get("classifier_output")
    # Step-back: broaden the query to recover when the first retrieval missed.
    stepback = f"{state['question']} (bối cảnh tổng quát, định nghĩa, nguyên nhân)"
    boot_in, boot_out = await _INNER.knowledge.bootstrap_search(
        query=stepback, document_title=doc_title, intent=co,
    )
    boot_out = _INNER.security.filter_tool_results(tool_output=boot_out, document_title=doc_title)
    trace = list(state.get("tool_trace") or [])
    trace.append({"tool_name": "search_hybrid_kg", "tool_input": boot_in,
                  "tool_output": boot_out, "corrective": True})

    assembled = await _INNER.context.assemble(
        state["question"], [s["tool_output"] for s in trace]
    )
    packed = assembled.get("packed_context", []) if isinstance(assembled, dict) else assembled
    out = await _INNER._answer(
        question=state["question"], packed_context=packed, tool_trace=trace,
        final_answer=None, chat_history=state.get("chat_history"),
        memory_context=state.get("memory_context"), verbosity=state.get("verbosity"),
    )
    return {
        "tool_trace": trace,
        "packed_context": packed,
        "answer": out.get("answer", ""),
        "citations": out.get("citations", []),
        "highlights": out.get("highlights", []),
        "critique_retries": state.get("critique_retries", 0) + 1,
    }
```

Add the router (after `_route_decide` ~384):

```python
def _route_critique(state: ChatState) -> str:
    if not settings.CRAG_ENABLED:
        return "ground"
    decision = state.get("critique_decision") or {}
    if decision.get("grounded", True):
        return "ground"
    if state.get("critique_retries", 0) >= settings.AGENT_CRITIQUE_MAX_RETRIES:
        return "ground"
    return "corrective_retrieve"
```

In `_build_graph`, register nodes and rewire the answer→critique→{corrective|ground} path. Add nodes after `g.add_node("answer", answer_node)`:

```python
    g.add_node("critique", critique)
    g.add_node("corrective_retrieve", corrective_retrieve)
```

Replace the edge `g.add_edge("answer", "ground")` (line 421) with:

```python
    g.add_edge("answer", "critique")
    g.add_conditional_edges("critique", _route_critique,
                            {"ground": "ground", "corrective_retrieve": "corrective_retrieve"})
    g.add_edge("corrective_retrieve", "critique")
```

Also route the fast path through critique instead of straight to ground — replace the `g.add_edge("fast_answer", "ground")` added in Task 4.2 with:

```python
    g.add_edge("fast_answer", "critique")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agent/test_critique_routing.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run agent + integration suites**

Run: `uv run pytest tests/agent/ tests/integration/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/agentrag/agent/graph_service.py tests/agent/test_critique_routing.py
git commit -m "feat(agent): CRAG critique node + bounded corrective re-retrieve (WS3)"
```

---

### Task 5.3: Multi-hop sequential bootstrap (dependent sub-queries)

**Files:**
- Modify: `src/agentrag/agent/graph_service.py` (`bootstrap` node ~170-232: add sequential branch)
- Test: `tests/agent/test_multihop_bootstrap.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_multihop_bootstrap.py
from src.agentrag.agent.graph_service import _chain_query


def test_chain_query_prepends_prior_snippet():
    prior = {"hits": [{"content": "Aspirin ức chế kết tập tiểu cầu, dùng trong NMCT."}]}
    chained = _chain_query("Liều dùng là bao nhiêu?", prior)
    assert "Aspirin" in chained
    assert "Liều dùng" in chained


def test_chain_query_no_prior_returns_original():
    assert _chain_query("Câu hỏi gốc", None) == "Câu hỏi gốc"
    assert _chain_query("Câu hỏi gốc", {"hits": []}) == "Câu hỏi gốc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agent/test_multihop_bootstrap.py -v`
Expected: FAIL with `ImportError: cannot import name '_chain_query'`

- [ ] **Step 3: Add `_chain_query` + sequential branch in `bootstrap`**

Add the helper near the top of `graph_service.py` (after `_INNER = AgentService()`):

```python
def _chain_query(subquery: str, prior_output: dict[str, Any] | None) -> str:
    """Multi-hop chaining (WS3): seed a dependent sub-query with the top snippet
    from the previous hop so later hops build on earlier answers instead of
    being retrieved blind."""
    if not prior_output:
        return subquery
    hits = prior_output.get("hits") or []
    if not hits:
        return subquery
    snippet = (hits[0].get("content") or "")[:240]
    if not snippet:
        return subquery
    return f"Bối cảnh: {snippet}\n\nCâu hỏi: {subquery}"
```

In the `bootstrap` node, replace the parallel `plan_subqueries` block (lines 179-207) with a branch: parallel by default, **sequential chaining** when `AGENT_MULTIHOP_ENABLED`:

```python
    # Plan subqueries: parallel (default) or sequential-chained (multi-hop).
    if state.get("plan_subqueries"):
        started = time.perf_counter()
        subqueries = state["plan_subqueries"]
        if settings.AGENT_MULTIHOP_ENABLED:
            prior_out: dict[str, Any] | None = None
            for sq in subqueries:
                chained = _chain_query(sq, prior_out)
                try:
                    sub_in, sub_out = await _INNER.knowledge.bootstrap_search(
                        query=chained, document_title=doc_title, intent=classifier_output,
                    )
                except BaseException:
                    continue
                sub_out = _INNER.security.filter_tool_results(
                    tool_output=sub_out, document_title=doc_title
                )
                prior_out = sub_out
                fp = _INNER.knowledge.fingerprint_call("search_hybrid_kg", sub_in)
                if fp in seen:
                    continue
                seen.add(fp)
                tool_trace.append({
                    "tool_name": "search_hybrid_kg", "tool_input": sub_in,
                    "tool_output": sub_out, "plan_subquery": sq, "multihop": True,
                })
        else:
            results = await _asyncio.gather(
                *[
                    _INNER.knowledge.bootstrap_search(
                        query=sq, document_title=doc_title, intent=classifier_output,
                    )
                    for sq in subqueries
                ],
                return_exceptions=True,
            )
            for sq, res in zip(subqueries, results):
                if isinstance(res, BaseException):
                    continue
                sub_in, sub_out = res
                sub_out = _INNER.security.filter_tool_results(
                    tool_output=sub_out, document_title=doc_title
                )
                fp = _INNER.knowledge.fingerprint_call("search_hybrid_kg", sub_in)
                if fp in seen:
                    continue
                seen.add(fp)
                tool_trace.append({
                    "tool_name": "search_hybrid_kg", "tool_input": sub_in,
                    "tool_output": sub_out, "plan_subquery": sq,
                })
        tool_latency_ms += (time.perf_counter() - started) * 1000
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agent/test_multihop_bootstrap.py -v`
Expected: PASS

- [ ] **Step 5: Run agent suite**

Run: `uv run pytest tests/agent/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/agentrag/agent/graph_service.py tests/agent/test_multihop_bootstrap.py
git commit -m "feat(agent): multi-hop sequential sub-query chaining (WS3)"
```

---

## PHASE 6 — Integration, benchmark ablation, docs

### Task 6.1: End-to-end smoke test (all flags on, mocked LLM/ES)

**Files:**
- Test: `tests/integration/test_enhancement_smoke.py`

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_enhancement_smoke.py
"""Smoke test: with every enhancement flag ON, the embed-input helper,
summary cap, classifier complexity, and critique compose without error.
Pure-unit level (no live ES/LLM) — guards against signature drift."""
from src.agentrag.ingestion.pipeline import _embed_input_for_chunk
from src.agentrag.retrieval import elasticsearch_retriever as er
from src.agentrag.structured.query_classifier import QueryIntentClassifier
from src.agentrag.agent.service import _has_uncertainty
import asyncio


def test_pipeline_helper_and_cap_and_classify(monkeypatch):
    monkeypatch.setattr(er.settings, "RAPTOR_SUMMARY_MAX_RATIO", 0.4)

    # WS1 embed-input
    assert _embed_input_for_chunk({"content": "x", "context_text": "ctx"}) == "ctx\n\nx"

    # WS2 cap
    r = er.ElasticsearchRetriever.__new__(er.ElasticsearchRetriever)
    capped = r._cap_summary_nodes(
        [{"node_level": 1}, {"node_level": 1}, {"node_level": 0}], size=3)
    assert sum(1 for h in capped if h.get("node_level", 0) >= 1) <= 1

    # WS4 complexity
    out = asyncio.run(QueryIntentClassifier().classify("Định nghĩa NMCT là gì?"))
    assert out.complexity in ("simple", "complex")

    # WS3 grounding marker
    assert _has_uncertainty("Tôi không tìm thấy thông tin.")
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/integration/test_enhancement_smoke.py -v`
Expected: PASS

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (no regressions across the whole suite)

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_enhancement_smoke.py
git commit -m "test(integration): enhancement smoke test (all WS composable)"
```

---

### Task 6.2: Benchmark ablation — wire flags into the eval harness

**Files:**
- Modify: `scripts/eval/` benchmark runner (locate the entrypoint with `grep -rl "STRUCTMEM_INGEST_MODE\|benchmark" scripts/eval`)
- Test: manual benchmark run (not a unit test)

- [ ] **Step 1: Find the benchmark entrypoint**

Run: `grep -rln "deepeval\|contextual_recall\|benchmark" scripts/eval src/agentrag/eval`
Expected: prints the runner file(s). Open the main runner and confirm it sets env flags before running.

- [ ] **Step 2: Add an ablation matrix runner**

Add a CLI flag `--ablate` to the benchmark runner that runs the 80-question set once per configuration, toggling env at process start (before importing settings) via `os.environ`:

```python
# in the benchmark runner, near argument parsing
ABLATIONS = {
    "baseline":   {},
    "cr":         {"CONTEXTUAL_RETRIEVAL_ENABLED": "true"},
    "cr_raptor":  {"CONTEXTUAL_RETRIEVAL_ENABLED": "true", "RAPTOR_ENABLED": "true"},
    "cr_raptor_crag": {"CONTEXTUAL_RETRIEVAL_ENABLED": "true", "RAPTOR_ENABLED": "true", "CRAG_ENABLED": "true"},
    "full":       {"CONTEXTUAL_RETRIEVAL_ENABLED": "true", "RAPTOR_ENABLED": "true",
                   "CRAG_ENABLED": "true", "ADAPTIVE_ROUTING_ENABLED": "true",
                   "SEMANTIC_CACHE_ENABLED": "true", "AGENT_MULTIHOP_ENABLED": "true"},
}
```

The runner re-ingests gold contexts **synchronously** (`STRUCTMEM_INGEST_MODE=sync`) once per ablation that changes ingest (`cr`, `cr_raptor`, `full`) so the index reflects the flags, then scores. Emit a markdown table comparing all ablations on the 5 quality metrics + p50/p95 latency + cost.

- [ ] **Step 3: Run the ablation benchmark**

Run (long; one-time): `uv run python -m scripts.eval.<runner> --dataset both --ablate full`
Expected: produces `docs/eval/benchmark_ablation_<date>.md` with a per-config table.

**Acceptance gates (the `full` row must satisfy):**
- answer correctness > 0.792 (target ≥ 0.85)
- contextual precision > 0.819 (target ≥ 0.88)
- faithfulness ≥ 0.80 (no regression)
- contextual recall ≥ 0.70, citation accuracy ≥ 0.70, failure rate < 0.05
- p50 latency < 10s (production orchestration config)

- [ ] **Step 4: Commit the report**

```bash
git add docs/eval/benchmark_ablation_*.md scripts/eval/
git commit -m "eval: ablation benchmark for CR/RAPTOR/CRAG/adaptive/cache"
```

---

### Task 6.3: Document new flags

**Files:**
- Modify: `.env.example`, `README.md`

- [ ] **Step 1: Add a documented block to `.env.example`**

Append, with the same comment style as the existing `RETRIEVAL_*` block:

```bash
# ── RAG enhancement (2026-06) — all default OFF, enable to A/B ──────────────
# WS1 Contextual Retrieval (re-ingest needed). Route CONTEXTUAL_RETRIEVAL_TASK
# to DeepSeek in LLM_TASK_MODEL_MAP for cheap doc-prefix-cached context-gen.
CONTEXTUAL_RETRIEVAL_ENABLED=false
# WS2 RAPTOR summary layer (re-ingest needed)
RAPTOR_ENABLED=false
RAPTOR_MAX_LEVELS=3
RAPTOR_MIN_LEAVES=8
# WS3 CRAG critique + multi-hop
CRAG_ENABLED=false
AGENT_CRITIQUE_MAX_RETRIES=1
AGENT_MULTIHOP_ENABLED=false
# WS4 Adaptive fast-path routing
ADAPTIVE_ROUTING_ENABLED=false
ADAPTIVE_FASTPATH_MIN_CONFIDENCE=0.85
# WS5 Semantic retrieval cache
SEMANTIC_CACHE_ENABLED=false
SEMANTIC_CACHE_THRESHOLD=0.97
```

- [ ] **Step 2: Add a short README subsection**

Under the retrieval section of `README.md`, add a "RAG enhancement (2026-06)" subsection summarizing the five flags, what each does, that CR + RAPTOR require re-ingest (route their tasks to DeepSeek via `LLM_TASK_MODEL_MAP`), and a pointer to `docs/superpowers/specs/2026-06-10-rag-enhancement-design.md`.

- [ ] **Step 3: Commit**

```bash
git add .env.example README.md
git commit -m "docs: document RAG enhancement flags (CR, RAPTOR, CRAG, adaptive, cache)"
```

---

## Backfill / rollout note (operational, not a code task)

After merging with flags ON in a staging `.env`:
1. Point `CONTEXTUAL_RETRIEVAL_TASK` + `RAPTOR_SUMMARY_TASK` at DeepSeek in `LLM_TASK_MODEL_MAP`.
2. Re-ingest existing documents (the async worker re-runs contextualize → embed → index → RAPTOR; the file caches make repeats idempotent). Watch `graph_status: enriching → done`.
3. Run the ablation benchmark; keep only the flags whose row beats baseline on the gates.
4. The semantic cache + adaptive routing are query-time only (no re-ingest) — safe to toggle independently.

---

## Self-Review

**Spec coverage:** WS1 → Tasks 0.2, 2.1, 2.2. WS2 → Tasks 0.2, 3.1–3.4. WS3 → Tasks 5.1–5.3. WS4 → Tasks 4.1–4.2. WS5 → Tasks 1.1–1.3. Migration/flags → Task 0.1, 0.2. Benchmark gates → Task 6.2. Docs → Task 6.3. All spec sections mapped.

**Type consistency:** `ClassifierOutput.complexity`/`single_domain` defined in 4.1, consumed in 4.2. `_critique` defined in 5.1, called by `critique` node in 5.2. `_cap_summary_nodes`, `_embed_input_for_chunk`, `_build_and_index_raptor`, `_chain_query`, `search_cached`/`_search_uncached`, `SemanticCache(threshold, ttl_seconds, max_items, clock)`, `Contextualizer.contextualize_chunks`, `RaptorBuilder.build` — names used consistently across tasks. ES fields `context_text`/`node_level`/`child_ids` defined in 0.2, written by RAPTOR (3.2) and read by cap (3.4).

**Placeholder scan:** every code step shows full code; commands have expected output; no "TBD"/"add error handling" left. Task 6.2 references a runner file discovered via grep (the one repo-specific path not hardcoded) — Step 1 locates it explicitly rather than guessing.
