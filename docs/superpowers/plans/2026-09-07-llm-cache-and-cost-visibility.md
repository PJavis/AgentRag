# LLM Cache + Cost Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LLM token cost visible and then actually reduce it — by measuring provider cache hits, by not throwing away the contextualizer cache on every parse change, and by adding a safe exact-match answer cache.

**Architecture:** Three independent layers, smallest-leverage-last. (1) `observability/cost.py` learns DeepSeek pricing and reads the `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` the provider already returns. (2) The contextualizer's file-cache key moves off the *parsed* document text and onto the *source bytes*, so a parser-config change reuses instead of regenerating. (3) A new valkey-backed exact-match answer cache, keyed on a corpus version that ingest bumps, defaulting OFF.

**Tech Stack:** Python 3.11, pydantic-settings, valkey/redis (`REDIS_URL`, already used by `ConversationStore` and the cost ledger), pytest + pytest-asyncio.

**Spec:** No separate spec doc. The design and its evidence are in this conversation and in `docs/eval/` — the measurements this plan argues from are restated under Motivating Measurements below.

## Global Constraints

- **Every new feature defaults OFF.** `ANSWER_CACHE_ENABLED: bool = False`. Matches the repo's standing convention (`test_new_config_flags.py` asserts safe defaults).
- **No new dependencies.** valkey/redis, cachetools and pytest-asyncio are already in `pyproject.toml`.
- **Never serve a cached answer across a corpus change.** The answer cache fails CLOSED: no known corpus version → no caching, read or write.
- **Never cache a conversational turn.** `chat_history` non-empty → bypass entirely. Follow-ups like *"viết dài hơn được không?"* depend on history, and `graph_service.chat` rewrites the question from it.
- **Exact match only.** No embedding similarity in the answer cache. See "Rejected" below — this is a clinical-safety decision, not a performance one.
- **Token counts are facts; USD is an estimate.** Cache-hit/miss token counts come from the provider and are recorded as-is. Prices are a local table that may drift; label them, do not present them as authoritative.
- Test command: `PYTHONPATH=. uv run pytest <path> -v`.

## Motivating Measurements

| finding | evidence |
|---|---|
| Provider prefix cache works and is the only thing saving tokens today | same 6k-token prefix twice: `cache_hit=0/miss=6043`, then `cache_hit=6016/miss=27` (99.6%) |
| Cost accounting cannot see it | `record()` reads only `usage.prompt_tokens`/`completion_tokens` |
| DeepSeek is priced as Gemini | `_PRICE_PER_1M` has no deepseek entry; `_price_for()` falls through to `gemini-2.5-flash` |
| The contextualizer cache is invalidated wholesale by a parse change | key is `(sig, doc_hash, chunk_hash)` where `doc_hash` = sha256 of the *parsed* text; the 2026-09-06 arm-B flip missed all 308 entries and paid ~4.6k calls |
| Nothing caches a completion | `_RESULT_CACHE` (60s) and `SemanticCache` (off) both store *retrieval payloads*, and both are per-gunicorn-worker across `-w 4` |

## File Structure

| file | responsibility |
|---|---|
| `src/agentrag/observability/cost.py` (modify) | pricing table + cache-token accounting + summary fields |
| `src/agentrag/ingestion/connectors/folder.py` (modify) | expose `source_bytes_sha` alongside the parse-aware `content_hash` |
| `src/agentrag/ingestion/contextualizer.py` (modify) | accept an explicit `doc_key`; key the file cache on it |
| `src/agentrag/ingestion/pipeline.py` (modify) | pass `doc_key`; bump the corpus version when an ingest finishes |
| `src/agentrag/common/corpus_version.py` (create) | read/bump the shared corpus version in valkey |
| `src/agentrag/services/answer_cache.py` (create) | exact-match answer cache: key construction, get/put |
| `src/agentrag/agent/graph_service.py` (modify) | wire the cache around `chat()` |
| `src/agentrag/config.py` (modify) | `ANSWER_CACHE_ENABLED`, `ANSWER_CACHE_TTL_SECONDS` |

---

### Task 1: Cost visibility — DeepSeek pricing + cache-token accounting

**Files:**
- Modify: `src/agentrag/observability/cost.py`
- Test: `tests/observability/test_cost_cache_tokens.py`

**Interfaces:**
- Produces: `record_llm_call(..., usage=...)` records `cache_hit_tokens` / `cache_miss_tokens`; `cost_summary()` gains `cache_hit_tokens`, `cache_miss_tokens`, `cache_hit_rate`.

- [ ] **Step 1: Write the failing test**

```python
# tests/observability/test_cost_cache_tokens.py
"""Cache-hit tokens are billed differently by the provider; the ledger must see them.

Measured on DeepSeek: a repeated 6k-token prefix returned
prompt_cache_hit_tokens=6016 / prompt_cache_miss_tokens=27. Billing all prompt
tokens at the miss rate overstates spend and hides whether caching works at all.
"""
from types import SimpleNamespace

import pytest

from src.agentrag.config import settings
from src.agentrag.observability import cost


@pytest.fixture(autouse=True)
def _tracking_on(monkeypatch):
    monkeypatch.setattr(settings, "LLM_COST_TRACKING_ENABLED", True)
    cost.reset_ledger()


def test_deepseek_has_its_own_price_and_is_not_billed_as_gemini():
    assert cost._price_for("deepseek-chat") != cost._price_for("gemini-2.5-flash")


def test_cache_hit_tokens_are_recorded_from_provider_usage():
    usage = SimpleNamespace(
        prompt_tokens=6043, completion_tokens=1,
        prompt_cache_hit_tokens=6016, prompt_cache_miss_tokens=27,
    )
    cost.record_llm_call(task="answer", model="deepseek-chat", latency_ms=10.0, usage=usage)
    entry = cost.recent(1)[0]
    assert entry["cache_hit_tokens"] == 6016
    assert entry["cache_miss_tokens"] == 27


def test_a_cached_prompt_costs_less_than_an_uncached_one():
    hit = SimpleNamespace(prompt_tokens=6043, completion_tokens=1,
                          prompt_cache_hit_tokens=6016, prompt_cache_miss_tokens=27)
    miss = SimpleNamespace(prompt_tokens=6043, completion_tokens=1,
                           prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=6043)
    cost.record_llm_call(task="answer", model="deepseek-chat", latency_ms=1.0, usage=miss)
    cost.record_llm_call(task="answer", model="deepseek-chat", latency_ms=1.0, usage=hit)
    miss_entry, hit_entry = cost.recent(2)[0], cost.recent(2)[1]
    assert hit_entry["usd"] < miss_entry["usd"]


def test_a_provider_without_cache_fields_still_records_zero_not_none():
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10)
    cost.record_llm_call(task="answer", model="gemini-2.5-flash", latency_ms=1.0, usage=usage)
    entry = cost.recent(1)[0]
    assert entry["cache_hit_tokens"] == 0
    assert entry["cache_miss_tokens"] == 100  # all prompt tokens were misses


def test_summary_reports_a_cache_hit_rate():
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=1,
                            prompt_cache_hit_tokens=900, prompt_cache_miss_tokens=100)
    cost.record_llm_call(task="answer", model="deepseek-chat", latency_ms=1.0, usage=usage)
    summary = cost.cost_summary()
    assert summary["cache_hit_tokens"] == 900
    assert summary["cache_miss_tokens"] == 100
    assert abs(summary["cache_hit_rate"] - 0.9) < 1e-9


def test_hit_rate_is_none_when_no_prompt_tokens_were_seen():
    assert cost.cost_summary()["cache_hit_rate"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/observability/test_cost_cache_tokens.py -v`
Expected: FAIL — `cost.reset_ledger` / `cost.recent` may not exist, and `cache_hit_tokens` is absent.

- [ ] **Step 3: Implement**

In `cost.py`, add DeepSeek to the price table and a cache-hit input price table:

```python
_PRICE_PER_1M = {
    # ... existing entries unchanged ...
    # DeepSeek: (uncached input, output). Cached input is priced separately in
    # _CACHE_HIT_PRICE_PER_1M. VERIFY against current published pricing — these
    # are a local estimate, and only the token COUNTS below are provider facts.
    "deepseek-chat":     (0.27, 1.10),
    "deepseek-v4-flash": (0.27, 1.10),
    "deepseek-v4-pro":   (0.55, 2.19),
}

#: Input price for tokens the provider served from its own prefix cache.
_CACHE_HIT_PRICE_PER_1M = {
    "deepseek-chat":     0.027,
    "deepseek-v4-flash": 0.027,
    "deepseek-v4-pro":   0.055,
}


def _cache_hit_price_for(model: str) -> float:
    """Price for cached input tokens; falls back to the uncached input price."""
    for key, price in _CACHE_HIT_PRICE_PER_1M.items():
        if model == key or key in model:
            return price
    return _price_for(model)[0]
```

Replace the token/price block inside `record_llm_call()`:

```python
    in_tokens: int | None = None
    out_tokens: int | None = None
    hit_tokens = 0
    miss_tokens: int | None = None
    if usage is not None:
        in_tokens = getattr(usage, "prompt_tokens", None)
        out_tokens = getattr(usage, "completion_tokens", None)
        # DeepSeek reports these; most providers do not. Absent → all misses.
        hit_tokens = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
        raw_miss = getattr(usage, "prompt_cache_miss_tokens", None)
        miss_tokens = int(raw_miss) if raw_miss is not None else None
    if in_tokens is None:
        in_tokens = _estimate_tokens(in_text)
    if out_tokens is None:
        out_tokens = _estimate_tokens(out_text)
    if miss_tokens is None:
        miss_tokens = max(int(in_tokens) - hit_tokens, 0)

    in_price, out_price = _price_for(model)
    hit_price = _cache_hit_price_for(model)
    usd = (
        miss_tokens * in_price + hit_tokens * hit_price + out_tokens * out_price
    ) / 1_000_000.0
```

Add the two fields to the `entry` dict built in `record_llm_call()`:

```python
        "cache_hit_tokens": int(hit_tokens),
        "cache_miss_tokens": int(miss_tokens),
```

Add them to `_coerce_entry` so stream round-trips keep them:

```python
        "cache_hit_tokens": int(fields.get("cache_hit_tokens", 0) or 0),
        "cache_miss_tokens": int(fields.get("cache_miss_tokens", 0) or 0),
```

Add introspection helpers near `cost_summary`:

```python
def recent(n: int = 20) -> list[dict[str, Any]]:
    """Last n ledger entries, oldest first. Test/debug introspection."""
    return _read_entries()[-n:]


def reset_ledger() -> None:
    """Clear the in-process ledger. Tests only — never called by the app."""
    with _LOCK:
        _LEDGER.clear()
```

In `cost_summary()`, accumulate and report the new fields — add before the return:

```python
    cache_hit = sum(e.get("cache_hit_tokens", 0) for e in entries)
    cache_miss = sum(e.get("cache_miss_tokens", 0) for e in entries)
    prompt_total = cache_hit + cache_miss
```

and include in the returned dict:

```python
        "cache_hit_tokens": cache_hit,
        "cache_miss_tokens": cache_miss,
        "cache_hit_rate": (cache_hit / prompt_total) if prompt_total else None,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/observability/test_cost_cache_tokens.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/observability/cost.py tests/observability/test_cost_cache_tokens.py
git commit -m "feat(cost): record provider cache-hit tokens and price DeepSeek as DeepSeek"
```

---

### Task 2: Contextualizer cache keyed on source bytes, not parsed text

**Files:**
- Modify: `src/agentrag/ingestion/connectors/folder.py`
- Modify: `src/agentrag/ingestion/contextualizer.py`
- Modify: `src/agentrag/ingestion/pipeline.py`
- Test: `tests/ingestion/test_contextualizer_cache_key.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: document dicts gain `source_bytes_sha: str`; `Contextualizer.contextualize_chunks(doc_text, chunks, document_title, doc_key: str | None = None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_contextualizer_cache_key.py
"""A parser-config change must not throw away every context sentence.

The context blurb answers "what is this passage about", which depends on the
chunk and the document's identity — not on whether the parser rendered tables.
Keying on the PARSED text made the 2026-09-06 arm-B flip regenerate ~4.6k
sentences that would all have been reusable.

Note the deliberate asymmetry with the document cache key in
`connectors/folder.py`, which DOES include the parser arm: a re-ingest must
redo the parse, while a derived per-chunk blurb need not. Same word "cache",
different dependency, different key.
"""
import hashlib

from src.agentrag.config import settings
from src.agentrag.ingestion.connectors.folder import FolderConnector
from src.agentrag.ingestion.contextualizer import Contextualizer


class _StubGateway:
    def __init__(self):
        self.calls = 0

    async def text_response(self, system_prompt, user_prompt, task):
        self.calls += 1
        return f"context {self.calls}"


def test_documents_expose_the_raw_source_hash_separately(tmp_path, monkeypatch):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 bytes")
    monkeypatch.setattr(settings, "PDF_PRESERVE_TABLES", True)
    doc = FolderConnector(str(tmp_path)).list_documents()[0]
    assert doc["source_bytes_sha"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # content_hash stays parse-aware so a flag flip still forces a re-ingest
    assert doc["content_hash"] != doc["source_bytes_sha"]


def test_a_parse_change_reuses_cached_context(tmp_path):
    gateway = _StubGateway()
    ctx = Contextualizer(gateway, cache_dir=str(tmp_path))
    chunks = [{"content": "Paracetamol 500 mg", "content_hash": "chunk-1"}]

    import asyncio
    asyncio.run(ctx.contextualize_chunks(
        doc_text="ORIGINAL PARSE", chunks=chunks, document_title="d", doc_key="src-sha"))
    assert gateway.calls == 1

    # Same document, different parse output (arm B appended table markdown).
    chunks2 = [{"content": "Paracetamol 500 mg", "content_hash": "chunk-1"}]
    asyncio.run(ctx.contextualize_chunks(
        doc_text="PARSE WITH | TABLES |", chunks=chunks2,
        document_title="d", doc_key="src-sha"))
    assert gateway.calls == 1, "a parse change must not invalidate the blurb"
    assert chunks2[0]["context_text"] == "context 1"


def test_a_genuine_source_change_does_invalidate(tmp_path):
    gateway = _StubGateway()
    ctx = Contextualizer(gateway, cache_dir=str(tmp_path))
    chunks = [{"content": "Paracetamol 500 mg", "content_hash": "chunk-1"}]

    import asyncio
    asyncio.run(ctx.contextualize_chunks(
        doc_text="A", chunks=chunks, document_title="d", doc_key="sha-v1"))
    chunks2 = [{"content": "Paracetamol 500 mg", "content_hash": "chunk-1"}]
    asyncio.run(ctx.contextualize_chunks(
        doc_text="A", chunks=chunks2, document_title="d", doc_key="sha-v2"))
    assert gateway.calls == 2


def test_without_a_doc_key_it_falls_back_to_the_old_behaviour(tmp_path):
    """Callers that pass no key keep the previous parsed-text keying."""
    gateway = _StubGateway()
    ctx = Contextualizer(gateway, cache_dir=str(tmp_path))
    import asyncio
    for text in ("PARSE A", "PARSE B"):
        chunks = [{"content": "x", "content_hash": "c1"}]
        asyncio.run(ctx.contextualize_chunks(
            doc_text=text, chunks=chunks, document_title="d"))
    assert gateway.calls == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/ingestion/test_contextualizer_cache_key.py -v`
Expected: FAIL — `KeyError: 'source_bytes_sha'` and `contextualize_chunks() got an unexpected keyword argument 'doc_key'`.

- [ ] **Step 3: Implement**

`folder.py` — return the raw hash alongside the parse-aware one:

```python
            raw_sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
            content_hash = _document_cache_key(file_path, path.suffix.lower())
            documents.append(
                {
                    "source_id": str(path.relative_to(self.folder_path)),
                    "title": path.stem,
                    "file_path": str(file_path),
                    "content_hash": content_hash,
                    # Identity of the FILE, independent of parser settings.
                    # Derived artefacts that do not depend on the parse (the
                    # contextualizer's per-chunk blurb) key on this instead, so a
                    # flag flip does not discard them.
                    "source_bytes_sha": raw_sha,
                    "source_type": _EXT_TO_SOURCE_TYPE.get(path.suffix.lower(), "unknown"),
                }
            )
```

`contextualizer.py` — accept and prefer an explicit key:

```python
    async def contextualize_chunks(
        self,
        doc_text: str,
        chunks: list[dict[str, Any]],
        document_title: str,
        doc_key: str | None = None,
    ) -> list[dict[str, Any]]:
        if not chunks:
            return chunks
        doc_clip = doc_text[: settings.CONTEXTUAL_MAX_DOC_CHARS]
        # Cache identity: the SOURCE when the caller supplies one, else the
        # parsed text. The blurb describes what a passage is about, which a
        # parser-config change does not alter — keying on the parse made every
        # such change regenerate the whole corpus.
        doc_hash = doc_key or sha256(doc_clip.encode("utf-8")).hexdigest()
```

`pipeline.py` — pass it at the call site:

```python
                chunks_search = await Contextualizer(LLMGateway()).contextualize_chunks(
                    doc_text=content,
                    chunks=chunks_search,
                    document_title=doc["title"],
                    doc_key=doc.get("source_bytes_sha"),
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/ingestion/test_contextualizer_cache_key.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/ingestion/connectors/folder.py src/agentrag/ingestion/contextualizer.py src/agentrag/ingestion/pipeline.py tests/ingestion/test_contextualizer_cache_key.py
git commit -m "perf(ingest): key the contextual cache on source bytes, not parsed text"
```

---

### Task 3: Corpus version in valkey

**Files:**
- Create: `src/agentrag/common/corpus_version.py`
- Modify: `src/agentrag/ingestion/pipeline.py`
- Test: `tests/common/test_corpus_version.py`

**Interfaces:**
- Produces: `get_corpus_version() -> str | None`, `bump_corpus_version() -> str | None`.
  `None` means "unknown" and every caller must fail closed on it.

- [ ] **Step 1: Write the failing test**

```python
# tests/common/test_corpus_version.py
"""The answer cache must never outlive the corpus it answered from.

2026-09-06 is the cautionary case: a re-ingest replaced every segment. A cached
answer keyed without a corpus version would have gone on citing segment ids that
no longer existed.
"""
from src.agentrag.common import corpus_version


class _FakeRedis:
    def __init__(self, value=None, broken=False):
        self.value = value
        self.broken = broken

    def get(self, key):
        if self.broken:
            raise RuntimeError("valkey down")
        return self.value

    def set(self, key, value):
        if self.broken:
            raise RuntimeError("valkey down")
        self.value = value


def test_version_is_none_when_never_set(monkeypatch):
    monkeypatch.setattr(corpus_version, "_client", lambda: _FakeRedis(None))
    assert corpus_version.get_corpus_version() is None


def test_bump_sets_and_returns_a_new_version(monkeypatch):
    fake = _FakeRedis(None)
    monkeypatch.setattr(corpus_version, "_client", lambda: fake)
    first = corpus_version.bump_corpus_version()
    assert first and fake.value == first
    second = corpus_version.bump_corpus_version()
    assert second != first, "each ingest must produce a distinct version"


def test_unreachable_valkey_reports_unknown_rather_than_raising(monkeypatch):
    monkeypatch.setattr(corpus_version, "_client", lambda: _FakeRedis(broken=True))
    assert corpus_version.get_corpus_version() is None
    assert corpus_version.bump_corpus_version() is None


def test_no_client_at_all_is_unknown(monkeypatch):
    monkeypatch.setattr(corpus_version, "_client", lambda: None)
    assert corpus_version.get_corpus_version() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/common/test_corpus_version.py -v`
Expected: FAIL — `ModuleNotFoundError: src.agentrag.common.corpus_version`.

- [ ] **Step 3: Implement**

```python
# src/agentrag/common/corpus_version.py
"""Shared marker for "which corpus generation is currently indexed".

Anything that caches an answer derived from the corpus must key on this, so a
re-ingest cannot leave a cached answer pointing at segments that no longer
exist. Returns None when the value is unknown — callers MUST fail closed on
None rather than caching against a guess.
"""
from __future__ import annotations

import logging
import time
import uuid

from src.agentrag.config import settings

logger = logging.getLogger(__name__)

_KEY = "agentrag:corpus_version"
_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        if not settings.REDIS_URL:
            return None
        try:
            from redis import Redis

            _CLIENT = Redis.from_url(settings.REDIS_URL, decode_responses=True)
        except Exception as exc:  # noqa: BLE001 — valkey optional
            logger.debug("corpus_version: redis init failed (%s)", exc)
            return None
    return _CLIENT


def get_corpus_version() -> str | None:
    client = _client()
    if client is None:
        return None
    try:
        return client.get(_KEY) or None
    except Exception as exc:  # noqa: BLE001 — valkey unreachable
        logger.debug("corpus_version: get failed (%s)", exc)
        return None


def bump_corpus_version() -> str | None:
    """Mark a new corpus generation. Called when an ingest run completes."""
    client = _client()
    if client is None:
        return None
    version = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    try:
        client.set(_KEY, version)
        return version
    except Exception as exc:  # noqa: BLE001 — valkey unreachable
        logger.debug("corpus_version: set failed (%s)", exc)
        return None
```

In `pipeline.py`, at the end of `ingest_folder` just before it returns its report, bump the version:

```python
    # A new corpus generation exists — invalidate anything keyed on the old one.
    try:
        from src.agentrag.common.corpus_version import bump_corpus_version

        bump_corpus_version()
    except Exception:
        logger.exception("corpus version bump failed")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/common/test_corpus_version.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/common/corpus_version.py src/agentrag/ingestion/pipeline.py tests/common/test_corpus_version.py
git commit -m "feat(ingest): publish a corpus version so caches can be invalidated"
```

---

### Task 4: Exact-match answer cache (default OFF)

**Files:**
- Create: `src/agentrag/services/answer_cache.py`
- Modify: `src/agentrag/config.py`
- Modify: `src/agentrag/agent/graph_service.py`
- Test: `tests/services/test_answer_cache.py`

**Interfaces:**
- Consumes: `get_corpus_version()` from Task 3.
- Produces: `AnswerCache.key(...) -> str | None`, `AnswerCache.get(key) -> dict | None`, `AnswerCache.put(key, payload) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/test_answer_cache.py
"""Exact-match answer cache. Deliberately NOT semantic — see the module docstring.

The dangerous case this design refuses: at 0.97 cosine, "liều dùng cho người lớn"
and "liều dùng cho trẻ em" are neighbours, and serving one for the other is a
clinical failure, not a cache miss.
"""
import json

import pytest

from src.agentrag.config import settings
from src.agentrag.services.answer_cache import AnswerCache


class _FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value


@pytest.fixture
def cache(monkeypatch):
    monkeypatch.setattr(settings, "ANSWER_CACHE_ENABLED", True)
    c = AnswerCache()
    monkeypatch.setattr(c, "_client", lambda: _FakeRedis())
    return c


def test_flag_defaults_off():
    from src.agentrag.config import Settings

    assert Settings.model_fields["ANSWER_CACHE_ENABLED"].default is False


def test_the_same_question_produces_the_same_key(cache):
    a = cache.key(question="Liều paracetamol?", corpus_version="v1",
                  document_title=None, domain_filter=None, verbosity=None, model="m")
    b = cache.key(question="  liều   PARACETAMOL? ", corpus_version="v1",
                  document_title=None, domain_filter=None, verbosity=None, model="m")
    assert a == b, "whitespace and case must normalise"


def test_a_different_question_produces_a_different_key(cache):
    a = cache.key(question="liều dùng cho người lớn", corpus_version="v1",
                  document_title=None, domain_filter=None, verbosity=None, model="m")
    b = cache.key(question="liều dùng cho trẻ em", corpus_version="v1",
                  document_title=None, domain_filter=None, verbosity=None, model="m")
    assert a != b


@pytest.mark.parametrize("field,value", [
    ("corpus_version", "v2"),
    ("document_title", "doc.pdf"),
    ("verbosity", "detailed"),
    ("model", "other-model"),
])
def test_every_scoping_input_changes_the_key(cache, field, value):
    base = dict(question="q", corpus_version="v1", document_title=None,
                domain_filter=None, verbosity=None, model="m")
    changed = {**base, field: value}
    assert cache.key(**base) != cache.key(**changed)


def test_domain_filter_changes_the_key_regardless_of_dict_ordering(cache):
    base = dict(question="q", corpus_version="v1", document_title=None, verbosity=None, model="m")
    one = cache.key(**base, domain_filter={"system": "tim_mach", "specialties": ["a"]})
    two = cache.key(**base, domain_filter={"specialties": ["a"], "system": "tim_mach"})
    three = cache.key(**base, domain_filter={"system": "noi"})
    assert one == two, "key order must not matter"
    assert one != three


def test_no_corpus_version_means_no_key_and_therefore_no_caching(cache):
    assert cache.key(question="q", corpus_version=None, document_title=None,
                     domain_filter=None, verbosity=None, model="m") is None


def test_put_then_get_round_trips(cache):
    fake = _FakeRedis()
    cache._client = lambda: fake
    key = cache.key(question="q", corpus_version="v1", document_title=None,
                    domain_filter=None, verbosity=None, model="m")
    cache.put(key, {"answer": "42", "citations": []})
    assert cache.get(key)["answer"] == "42"


def test_get_of_a_missing_key_is_none(cache):
    assert cache.get("nope") is None


def test_a_none_key_never_touches_the_store(cache):
    assert cache.get(None) is None
    cache.put(None, {"answer": "x"})  # must not raise


def test_disabled_cache_never_returns_anything(monkeypatch):
    monkeypatch.setattr(settings, "ANSWER_CACHE_ENABLED", False)
    c = AnswerCache()
    fake = _FakeRedis()
    fake.store["k"] = json.dumps({"answer": "stale"})
    monkeypatch.setattr(c, "_client", lambda: fake)
    assert c.get("k") is None


def test_corrupt_cached_json_is_ignored_not_raised(cache):
    fake = _FakeRedis()
    fake.store["k"] = "{not json"
    cache._client = lambda: fake
    assert cache.get("k") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/services/test_answer_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: src.agentrag.services.answer_cache`.

- [ ] **Step 3: Implement the config flags**

In `config.py`, after `SEMANTIC_CACHE_MAX_ITEMS`:

```python
    #: Exact-match answer cache (valkey). EXACT match only — never semantic:
    #: at a 0.97 similarity threshold "liều cho người lớn" and "liều cho trẻ em"
    #: are neighbours, and serving one for the other is a clinical failure.
    #: Keyed on the corpus version, so a re-ingest invalidates every entry.
    ANSWER_CACHE_ENABLED: bool = False
    ANSWER_CACHE_TTL_SECONDS: int = 86400
```

- [ ] **Step 4: Implement the cache**

```python
# src/agentrag/services/answer_cache.py
"""Exact-match answer cache, keyed on the corpus version.

Deliberately NOT semantic. A similarity-keyed answer cache in a medical corpus
serves the wrong dose to the wrong patient class: at a 0.97 cosine threshold
"liều dùng cho người lớn" and "liều dùng cho trẻ em" are neighbours. Exact match
has no such failure mode, and on repeated demo/eval traffic it still hits.

Fails closed everywhere: unknown corpus version, disabled flag, unreachable
valkey, or unparseable payload all mean "no cache", never "cache anyway".
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from src.agentrag.common.corpus_version import get_corpus_version
from src.agentrag.config import settings

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_PREFIX = "agentrag:answer_cache:"
_CLIENT = None


class AnswerCache:
    def _client(self):
        global _CLIENT
        if _CLIENT is None:
            if not settings.REDIS_URL:
                return None
            try:
                from redis import Redis

                _CLIENT = Redis.from_url(settings.REDIS_URL, decode_responses=True)
            except Exception as exc:  # noqa: BLE001 — valkey optional
                logger.debug("answer_cache: redis init failed (%s)", exc)
                return None
        return _CLIENT

    @staticmethod
    def _normalize(question: str) -> str:
        return _WS.sub(" ", (question or "").strip().lower())

    def key(
        self,
        question: str,
        corpus_version: str | None,
        document_title: str | None,
        domain_filter: dict[str, Any] | None,
        verbosity: str | None,
        model: str,
    ) -> str | None:
        """Cache key, or None when caching must not happen at all.

        No corpus version → None. A cached answer that outlives its corpus cites
        segments that no longer exist.
        """
        if not corpus_version:
            return None
        payload = json.dumps(
            {
                "q": self._normalize(question),
                "corpus": corpus_version,
                "doc": document_title or "",
                # sort_keys so an equivalent filter written in another order
                # produces the same key.
                "filter": domain_filter or {},
                "verbosity": verbosity or "",
                "model": model or "",
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return _PREFIX + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str | None) -> dict | None:
        if not key or not settings.ANSWER_CACHE_ENABLED:
            return None
        client = self._client()
        if client is None:
            return None
        try:
            raw = client.get(key)
        except Exception as exc:  # noqa: BLE001 — valkey unreachable
            logger.debug("answer_cache: get failed (%s)", exc)
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            # A corrupt entry is a miss, never an exception on the answer path.
            return None

    def put(self, key: str | None, payload: dict) -> None:
        if not key or not settings.ANSWER_CACHE_ENABLED:
            return
        client = self._client()
        if client is None:
            return
        try:
            client.setex(
                key,
                settings.ANSWER_CACHE_TTL_SECONDS,
                json.dumps(payload, ensure_ascii=False),
            )
        except Exception as exc:  # noqa: BLE001 — caching must never break answering
            logger.debug("answer_cache: put failed (%s)", exc)


def current_corpus_version() -> str | None:
    """Indirection so callers do not import corpus_version directly."""
    return get_corpus_version()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/services/test_answer_cache.py -v`
Expected: 12 passed.

- [ ] **Step 6: Wire it into `graph_service.chat`**

Add a test first:

```python
# append to tests/services/test_answer_cache.py
def test_a_conversational_turn_is_never_cached():
    """Follow-ups depend on history — graph_service rewrites the question from
    it — so a history-blind key would serve the wrong answer."""
    from src.agentrag.services.answer_cache import cacheable_turn

    assert cacheable_turn(chat_history=None)
    assert cacheable_turn(chat_history=[])
    assert not cacheable_turn(chat_history=[{"role": "user", "content": "trước đó"}])
```

Add to `answer_cache.py`:

```python
def cacheable_turn(chat_history: list[dict] | None) -> bool:
    """Only stateless first turns may be cached.

    `graph_service.chat` rewrites a verbose follow-up using the previous user
    message, so two identical question strings can legitimately require
    different answers. Caching those would serve the wrong one.
    """
    return not chat_history
```

In `graph_service.chat`, immediately after `update_turn_trace(...)`:

```python
        from src.agentrag.services.answer_cache import (
            AnswerCache, cacheable_turn, current_corpus_version,
        )

        _cache = AnswerCache()
        _cache_key = None
        if settings.ANSWER_CACHE_ENABLED and cacheable_turn(chat_history):
            _cache_key = _cache.key(
                question=question,
                corpus_version=current_corpus_version(),
                document_title=document_title,
                domain_filter=domain_filter,
                verbosity=verbosity,
                model=settings.AGENT_MODEL or "",
            )
            cached = _cache.get(_cache_key)
            if cached is not None:
                cached["cache_hit"] = True
                return cached
```

and immediately before `chat()` returns its result dict, store it:

```python
        if _cache_key:
            _cache.put(_cache_key, result)
```

- [ ] **Step 7: Run the full gate**

Run: `PYTHONPATH=. CONTEXTUAL_RETRIEVAL_ENABLED=false RAPTOR_ENABLED=false uv run pytest -q --ignore=tests/ontology --ignore=tests/ingestion`
Expected: all pass, including `tests/services/test_new_config_flags.py` (the new flag defaults False).

- [ ] **Step 8: Commit**

```bash
git add src/agentrag/services/answer_cache.py src/agentrag/config.py src/agentrag/agent/graph_service.py tests/services/test_answer_cache.py
git commit -m "feat(cache): exact-match answer cache keyed on corpus version (default off)"
```

---

## Rejected (recorded so they are not re-proposed)

| idea | why rejected |
|---|---|
| Semantic (embedding-similarity) answer cache | At 0.97 cosine, "liều cho người lớn" ≈ "liều cho trẻ em". Serving one for the other is a clinical failure, not a cache miss. Would need exact agreement on extracted entities/numbers before it could be considered |
| Caching embeddings | TEI is local and free — there is no token cost to recover |
| Raising `_RESULT_CACHE` TTL above 60s | It carries no corpus version, so a longer TTL can serve retrieval payloads referencing segments a re-ingest deleted. Add the version first |
| Making `SemanticCache` distributed | It caches retrieval payloads, not completions — it cannot reduce token cost, which is the goal here |

## Self-Review

**Spec coverage:** the three agreed items map to Tasks 1, 2 and 4; Task 3 exists because Task 4's correctness requires it. **Type consistency:** `doc_key` is `str | None` at both the pipeline call site and the contextualizer signature; `get_corpus_version()`/`current_corpus_version()` both return `str | None` and every consumer treats `None` as "do not cache"; `AnswerCache.key()` returns `str | None` and both `get`/`put` accept `None`. **Placeholders:** none — every step carries runnable code.
