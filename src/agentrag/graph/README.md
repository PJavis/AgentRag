# Module: `graph` — StructMem Knowledge Extraction

**Vị trí:** `src/pam/graph/`

Triển khai StructMem (arXiv:2604.21748) thay thế Graphiti/Neo4j. Mỗi chunk tài liệu được xử lý bằng **2 LLM calls song song** (factual + relational), kết quả index vào Elasticsearch. Background worker định kỳ chạy cross-chunk consolidation để tạo synthesis entries.

---

## Files

| File | Class / Function | Mô tả |
|---|---|---|
| `structmem_service.py` | `StructMemService` | Core extraction — 2 parallel LLM calls/chunk, cache, retry |
| `structmem_sync.py` | `index_structmem_views()` | Build ES documents từ entries và bulk index |
| `graph_jobs.py` | `run_graph_worker()`, `process_graph_job()` | Async ingest worker — nhận jobs từ queue |
| `consolidation_jobs.py` | `run_consolidation_worker()`, `process_consolidation_job()` | Cross-chunk synthesis worker |

---

## StructMem vs Graphiti

| | Graphiti (cũ) | StructMem (hiện tại) |
|---|---|---|
| LLM calls/chunk | 4 (tuần tự) | 2 (song song) |
| Storage | Neo4j | Elasticsearch |
| Token/chunk | ~5,300 | ~2,900 |
| Chi phí/100 chunks | ~$1.28 | ~$0.97 |
| Latency/chunk | ~4x | ~2x (parallel) |

---

## Luồng xử lý

### Ingest (per chunk)

```
chunk content
  │
  ├──▶ asyncio.gather(
  │       _call_factual_extraction()    → factual_entries (definition, property, event, ...)
  │       _call_relational_extraction() → relational_entries (causes, enables, requires, ...)
  │    )
  │
  ├──▶ cache kết quả (SHA256 key)
  │
  └──▶ index_structmem_views()
          embed entries
          bulk upsert vào pam_entries (stable _id = SHA256(group|pos|type|content[:80]))
```

### Consolidation (background, mỗi CONSOLIDATION_THRESHOLD chunks)

```
pam_entries {consolidated: false, group_id: X}
  │
  ├──▶ embed concat → cosine search → top-K historical entries làm seed
  ├──▶ LLM synthesis call → synthesis_entries (pattern, contradiction, causal_chain, ...)
  ├──▶ embed synthesis → bulk index vào pam_synthesis
  └──▶ bulk update: consolidated=true trên buffer entries
```

---

## Entry Schema

### `pam_entries`

```
content, entry_type (factual|relational), fact_type, relation_type,
source_entity, target_entity, document_title, group_id, chunk_position,
content_hash, consolidated (bool), embedding (dense_vector), created_at
```

### `pam_synthesis`

```
content, hypothesis_type (pattern|generalization|contradiction|causal_chain|gap),
supporting_entry_ids, entities_involved, confidence, reasoning,
document_title, group_id, consolidation_run_id, embedding, created_at
```

---

## Tương tác

| Module | Vai trò |
|---|---|
| `ingestion.pipeline` | Gửi chunks vào graph_jobs queue (async) hoặc gọi trực tiếp (sync) |
| `ingestion.stores.ElasticsearchStore` | Index entries vào pam_entries + pam_synthesis |
| `services.LLMGateway` | Gọi LLM cho extraction + consolidation |
| `retrieval.ElasticsearchRetriever` | Search entries + synthesis trong `_entries_search()` |

---

## Config liên quan

| Key | Default | Mô tả |
|---|---|---|
| `STRUCTMEM_ENABLED` | `true` | Bật/tắt StructMem extraction |
| `STRUCTMEM_ENTRIES_INDEX_NAME` | `pam_entries` | ES index cho entries |
| `STRUCTMEM_SYNTHESIS_INDEX_NAME` | `pam_synthesis` | ES index cho synthesis |
| `STRUCTMEM_CONSOLIDATION_THRESHOLD` | `20` | Số chunks tích lũy trước khi trigger consolidation |
| `STRUCTMEM_CONSOLIDATION_HISTORY_TOP_K` | `15` | Số historical entries dùng làm seed |
| `GRAPH_CHUNK_MAX_TOKENS` | `1536` | Token/chunk cho StructMem extraction |
| `GRAPH_MAX_CONCURRENCY` | `1` | Số chunks xử lý đồng thời |
| `GRAPH_CHUNK_TIMEOUT_SECONDS` | `300` | Timeout mỗi chunk |
| `GRAPH_CHUNK_RETRIES` | `3` | Số lần retry khi fail |
| `GRAPH_ENABLE_CACHE` | `true` | Cache extraction results |
| `GRAPH_CACHE_DIR` | `.cache/pam/graph` | Thư mục cache |
| `GRAPH_INGEST_MODE` | `async` | `sync` = chạy ngay, `async` = enqueue |
