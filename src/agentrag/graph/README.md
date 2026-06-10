# graph — StructMem knowledge extraction + consolidation + vision ARQ jobs

## Mục đích / Purpose
Module này chứa các **background-job worker bodies** chạy sau khi document đã được
ingest text segments. Ba luồng việc: (1) **StructMem** trích xuất tri thức từ mỗi
chunk (2 LLM call song song: factual + relational) và index vào Elasticsearch;
(2) **Consolidation** — cross-chunk synthesis sinh ra các giả thuyết quan hệ liên đoạn;
(3) **Vision** — mô tả ảnh trong tài liệu bằng Vision LLM rồi index thành image
segments. StructMem thay thế Graphiti/Neo4j (arXiv:2604.21748): ít LLM call hơn, lưu
vào ES thay vì graph DB.

## Plane
**Execution Plane.** Đây là các stateless IO workers — chúng gọi LLM, embedder, ES,
PG theo input primitive và không sở hữu routing/prompt-branching ở tầng reasoning. Lưu
ý: các worker này *tự build prompt extraction/synthesis* (system prompts hard-coded
trong module), nên về mặt boundary chúng nằm ở ranh giới — nhưng vì là pure async job
bodies được gọi bởi ARQ wrappers với kwargs primitive (theo "Worker contract" trong
ARCHITECTURE.md), chúng thuộc Execution Plane.

## Key files
| File | Responsibility |
|---|---|
| `structmem_service.py` | `StructMemService` — core extraction. 2 parallel LLM call/chunk (`extract_chunk`), file cache (SHA256 key), retry + exponential backoff, batch `sync_chunks` với semaphore + progress callback. Provider routing (`openai`/`ollama`/`gemini`/`hf_inference`). |
| `structmem_sync.py` | `build_entry_docs()` + `index_structmem_views()` — convert raw extraction results thành ES docs, embed, bulk-index entries. `_coerce_entity()` chuẩn hoá entity (string \| list → string). |
| `graph_jobs.py` | `process_graph_job(GraphIngestJob, arq_pool)` — job body cho `graph_ingest`. Parse (cache hoặc file) → chunk → `sync_chunks` → `index_structmem_views` → cập nhật `Document.graph_*` status → chain sang `consolidate` nếu đủ ngưỡng. |
| `consolidation_jobs.py` | `process_consolidation_job(ConsolidationJob)` — job body cho `consolidate`. Lấy unconsolidated entries → mean-embedding cosine search → reconstruct context → LLM synthesis → index synthesis → mark consolidated. |
| `vision_jobs.py` | `process_vision_job(VisionExtractJob)` — job body cho `vision_extract`. Describe ảnh (batch hoặc per-image) với RPM token bucket + retry → embed (text + optional CLIP visual) → upsert image `Segment` vào PG + ES. |

## Public interface
Tất cả entry points được gọi **gián tiếp qua ARQ wrappers** ở `worker/functions.py` —
không import trực tiếp từ reasoning code. Wiring:

| ARQ function (`worker/functions.py`) | Gọi vào |
|---|---|
| `graph_ingest(ctx, ...)` | `process_graph_job(GraphIngestJob(...), arq_pool=ctx["redis"])` |
| `consolidate(ctx, ...)` | `process_consolidation_job(ConsolidationJob(...))` |
| `vision_extract(ctx, ...)` | `process_vision_job(VisionExtractJob(...))` |

Job dataclasses (input contracts, primitive kwargs):
```python
@dataclass(frozen=True)
class GraphIngestJob:
    document_id: uuid.UUID; folder_path: str; source_id: str
    title: str; parsed_cache_path: str | None = None

@dataclass(frozen=True)
class ConsolidationJob:
    group_id: str; document_id: uuid.UUID; trigger_chunk_count: int

@dataclass
class VisionExtractJob:
    document_id: uuid.UUID; title: str
    image_records: list[dict[str, Any]]   # [{path, page, mime, url}]
```

Hai thứ trong module này được **import trực tiếp** bởi retrieval (không qua ARQ):
- `StructMemService` và `StructMemService.normalize_group_id(value)` — dùng trong
  `retrieval/elasticsearch_retriever.py::_entries_search` để normalize source_ids
  thành group_ids khi search entries/synthesis.

## Data flow
**Upstream callers** (enqueue): `ingestion/pipeline.py` enqueue `graph_ingest` (khi
`STRUCTMEM_INGEST_MODE=async`) và `vision_extract` (khi `VISION_INGEST_MODE=async`)
qua `arq_pool.enqueue_job`. `process_graph_job` tự chain sang `consolidate` khi
`total_chunks ≥ STRUCTMEM_CONSOLIDATION_THRESHOLD`.

**StructMem ingest** (`process_graph_job`):
```
parsed content (cache_path hoặc re-parse Excel/MarkItDown/text)
  → HybridChunker.chunk
  → StructMemService.sync_chunks  (per chunk: gather(factual, relational), cache, retry)
  → index_structmem_views  (embed → es_store.index_entries → unified memory_doc index)
  → Document.graph_status = done | done_partial (on error)
  [if total_chunks ≥ threshold] → arq_pool.enqueue_job("consolidate", ...)
```

**Consolidation** (`process_consolidation_job`):
```
es_store.get_unconsolidated_entries(group_id)
  → embed buffer → mean vector → es_store.search_entries (top-K historical seeds)
  → es_store.get_entries_by_chunk_position (reconstruct context, dedupe buffer)
  → _run_synthesis  (LLM, reuses StructMemService._client/_model)
  → es_store.index_synthesis
  → es_store.mark_entries_consolidated(buffer ids)
```

**Vision** (`process_vision_job`):
```
image_records → _describe_one / image_parser.describe_batch  (RPM bucket + retry)
  → embed text (+ CLIP visual if VISUAL_EMBEDDING_ENABLED)
  → PG Segment rows (position = max(existing)+1, segment_type="image")
  → es_store.index_segments
```

**Downstream stores**: `ingestion/stores/elasticsearch_store.py`
(`index_entries`, `index_synthesis`, `search_entries`,
`get_unconsolidated_entries`, `get_entries_by_chunk_position`,
`mark_entries_consolidated`, `index_segments`), `ingestion/embedders/factory.py`,
`ingestion/parsers/{excel_parser,markitdown_parser,image_parser}.py`,
`ingestion/chunkers/hybrid_chunker.py`. PG: `database/models.py::{Document, Segment}`.

## Config
StructMem (`config.py`):
| Key | Default | Mô tả |
|---|---|---|
| `STRUCTMEM_ENABLED` | `true` | Bật StructMem extraction |
| `STRUCTMEM_INGEST_MODE` | `async` | `async` = enqueue `graph_ingest`; `sync` = inline |
| `STRUCTMEM_INDEX` | `agentrag_memory_doc` | **Unified** ES index cho cả entries + synthesis (phân biệt qua field `kind` ∈ {`entry`,`synthesis`}). R4 collapse — không còn index riêng. |
| `STRUCTMEM_CONSOLIDATION_THRESHOLD` | `20` | Số chunks trước khi chain consolidation |
| `STRUCTMEM_CONSOLIDATION_HISTORY_TOP_K` | `15` | Số historical seeds (buffer fetch dùng `*5`) |
| `STRUCTMEM_CHUNK_MAX_TOKENS` / `STRUCTMEM_CHUNK_OVERLAP_TOKENS` | `1536` / `128` | Chunking cho extraction |
| `STRUCTMEM_MAX_CONCURRENCY` | `1` | Concurrent chunks trong `sync_chunks` |
| `STRUCTMEM_CHUNK_TIMEOUT_SECONDS` / `STRUCTMEM_CHUNK_RETRIES` | `300` / `3` | Per-chunk timeout + retry |
| `STRUCTMEM_ENABLE_CACHE` | `true` | File cache extraction results |
| `STRUCTMEM_CACHE_DIR` | `.cache/agentrag/extract` | Cache dir |
| `EXTRACTION_PROVIDER` / `EXTRACTION_MODEL` / `EXTRACTION_BASE_URL` / `EXTRACTION_TEMPERATURE` | `ollama` / `llama3.1:8b…` / — / `0.0` | LLM backend cho extraction + synthesis |
| `CHUNK_TOKENIZER_MODEL` | `text-embedding-3-large` | Tokenizer cho HybridChunker |

Vision (`config.py`):
| Key | Default | Mô tả |
|---|---|---|
| `VISION_PROVIDER` | `None` | Nếu `None` → skip vision (text-only). Gate cho `process_vision_job`. |
| `VISION_INGEST_MODE` | `async` | `async` = enqueue `vision_extract` |
| `VISION_MAX_CONCURRENCY` / `VISION_MAX_RPM` | `4` / `10` | Semaphore + RPM token bucket (`_RpmBucket`); RPM=0 tắt cap |
| `VISION_PER_IMAGE_RETRIES` | `3` | Retry transient (429/503/timeout) |
| `VISION_FLUSH_BATCH_SIZE` | `10` | Embed+index batch size |
| `VISION_DESCRIBE_BATCH` | `4` | Ảnh/LLM call; `=1` → per-image |
| `VISUAL_EMBEDDING_ENABLED` | `true` | Thêm CLIP visual embedding cho cross-modal kNN |
| `ORIGINALS_DIR` | `data/originals` | Fallback path khi tmp upload dir đã bị cleanup |
| `EXCEL_INGEST_MODE` | `markdown` | Excel parse mode trong `process_graph_job` |

## Gotchas
- **Unified index**: `index_entries` set `kind="entry"`, `index_synthesis` set
  `kind="synthesis"`, cả hai ghi vào cùng `STRUCTMEM_INDEX` (`agentrag_memory_doc`).
  README cũ nói tới `agentrag_entries` / `agentrag_synthesis` và
  `STRUCTMEM_ENTRIES_INDEX_NAME` / `STRUCTMEM_SYNTHESIS_INDEX_NAME` — **đã bỏ ở R4**.
  Comment "pam_entries"/"pam_synthesis" trong docstring `structmem_sync.py` /
  `consolidation_jobs.py` cũng stale.
- **graph_ingest failure → `done_partial`, không phải `failed`**: text segments đã
  được index ở pipeline (status `searchable`) trước khi graph job chạy, nên doc vẫn
  chat được; chỉ thiếu graph enrichment. Worker bắt exception, set
  `graph_status="done_partial"` rồi return (không raise).
- **`_get_structmem_service()` cache global**: `StructMemService` được khởi tạo lần
  đầu rồi reuse trong process worker — provider/model đổi runtime sẽ không có hiệu lực
  cho tới khi restart worker.
- **`process_consolidation_job` reuse client**: `_run_synthesis` mượn `svc._client` và
  `svc._model` từ một `StructMemService()` mới — synthesis dùng `temperature=0.0`
  hard-coded, độc lập với `EXTRACTION_TEMPERATURE`.
- **`group_id` phải normalize**: `process_graph_job` index entries dưới
  `StructMemService.normalize_group_id(source_id)`; retriever cũng normalize source_ids
  khi search. Mọi nơi đọc/ghi entries phải dùng cùng hàm normalize (regex
  `[^a-zA-Z0-9_-]+` → `_`).
- **Vision position monotonicity**: `_flush` đọc `max(Segment.position)` của document
  **một lần** lúc đầu (`base_pos`) rồi gán tăng dần — không chịu được khi có job vision
  khác ghi song song cùng document.
- **CLIP visual embed best-effort**: lỗi visual embedding chỉ log + skip; image vẫn
  index với text embedding (không có `image_embedding`).
- LLM extraction trả `[]` thay vì raise khi parse JSON fail (per-call try/except trong
  `_call_factual_extraction` / `_call_relational_extraction`) — chunk lỗi sẽ ra 0 entry
  chứ không fail cả job.
