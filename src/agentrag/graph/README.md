# Module: `graph` — StructMem Knowledge Extraction

**Vị trí:** `src/agentrag/graph/`

Triển khai StructMem (arXiv:2604.21748) thay thế Graphiti/Neo4j. Mỗi chunk tài liệu được xử lý bằng **2 LLM calls song song** (factual + relational), kết quả index vào Elasticsearch. Jobs chạy nền qua **ARQ** (Redis-backed queue).

---

## Files

| File | Class / Function | Mô tả |
|---|---|---|
| `structmem_service.py` | `StructMemService` | Core extraction — 2 parallel LLM calls/chunk, cache, retry |
| `structmem_sync.py` | `index_structmem_views()` | Build ES documents từ entries và bulk index |
| `graph_jobs.py` | `process_graph_job()` | Xử lý ingest job (được gọi bởi ARQ worker) |
| `consolidation_jobs.py` | `process_consolidation_job()` | Cross-chunk synthesis (được gọi bởi ARQ worker) |

---

## StructMem vs Graphiti

| | Graphiti (cũ) | StructMem (hiện tại) |
|---|---|---|
| LLM calls/chunk | 4 (tuần tự) | 2 (song song) |
| Storage | Neo4j | Elasticsearch |
| Token/chunk | ~5,300 | ~2,900 |
| Cost/100 chunks | ~$1.28 | ~$0.97 |
| Background jobs | asyncio.Queue | ARQ (Redis-backed) |

---

## Luồng xử lý

### Ingest (per chunk) — `process_graph_job()`

```
chunk content
  │
  ├──▶ asyncio.gather(
  │       _call_factual_extraction()    → factual_entries
  │       _call_relational_extraction() → relational_entries
  │    )
  │
  ├──▶ cache kết quả (SHA256 key)
  │
  └──▶ index_structmem_views()
          embed entries
          bulk upsert vào agentrag_entries

  [if total_chunks ≥ STRUCTMEM_CONSOLIDATION_THRESHOLD]
  └──▶ arq_pool.enqueue_job("consolidate", ...)
```

### Consolidation (background) — `process_consolidation_job()`

```
agentrag_entries {consolidated: false, group_id: X}
  │
  ├──▶ embed concat → cosine search → top-K historical seeds
  ├──▶ LLM synthesis → synthesis_entries (pattern, contradiction, causal_chain, ...)
  ├──▶ embed synthesis → bulk index vào agentrag_synthesis
  └──▶ bulk update: consolidated=true trên buffer entries
```

---

## ARQ Integration

Jobs không còn dùng `asyncio.Queue` — được gọi bởi ARQ worker:

```python
# graph_jobs.py
async def process_graph_job(job: GraphIngestJob, arq_pool: ArqRedis | None = None) -> None:
    ...
    # Chain sang consolidation qua arq_pool
    await arq_pool.enqueue_job("consolidate", group_id=..., document_id=..., ...)
```

ARQ function wrappers nằm ở `worker/functions.py`:
- `graph_ingest(ctx, ...)` → gọi `process_graph_job(job, arq_pool=ctx["redis"])`
- `consolidate(ctx, ...)` → gọi `process_consolidation_job(job)`

---

## Entry Schema

### `agentrag_entries`

```
content, entry_type (factual|relational), fact_type, relation_type,
source_entity, target_entity, document_title, group_id, chunk_position,
content_hash, consolidated (bool), embedding (dense_vector), created_at
```

### `agentrag_synthesis`

```
content, hypothesis_type (pattern|generalization|contradiction|causal_chain|gap),
supporting_entry_ids, entities_involved, confidence, reasoning,
document_title, group_id, consolidation_run_id, embedding, created_at
```

---

## Tương tác

| Module | Vai trò |
|---|---|
| `ingestion.pipeline` | Enqueue `graph_ingest` job qua `worker.pool.get_pool()` |
| `worker.functions` | ARQ wrappers gọi các hàm process trong module này |
| `ingestion.stores.ElasticsearchStore` | Index entries vào agentrag_entries + agentrag_synthesis |
| `retrieval.ElasticsearchRetriever` | Search entries + synthesis trong `_entries_search()` |

---

## Config liên quan

| Key | Default | Mô tả |
|---|---|---|
| `STRUCTMEM_ENABLED` | `true` | Bật/tắt StructMem extraction |
| `STRUCTMEM_ENTRIES_INDEX_NAME` | `agentrag_entries` | ES index cho entries |
| `STRUCTMEM_SYNTHESIS_INDEX_NAME` | `agentrag_synthesis` | ES index cho synthesis |
| `STRUCTMEM_CONSOLIDATION_THRESHOLD` | `20` | Số chunks trước khi trigger consolidation |
| `STRUCTMEM_CONSOLIDATION_HISTORY_TOP_K` | `15` | Số historical entries làm seed |
| `STRUCTMEM_CHUNK_MAX_TOKENS` | `1536` | Token/chunk cho extraction |
| `STRUCTMEM_MAX_CONCURRENCY` | `1` | Concurrent chunks trong worker |
| `STRUCTMEM_CHUNK_TIMEOUT_SECONDS` | `300` | Timeout mỗi job |
| `STRUCTMEM_CHUNK_RETRIES` | `3` | Retry khi fail |
| `STRUCTMEM_ENABLE_CACHE` | `true` | Cache extraction results |
| `STRUCTMEM_INGEST_MODE` | `async` | `sync` hoặc `async` |
