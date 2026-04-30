# Module: `worker` — ARQ Background Worker

**Vị trí:** `src/agentrag/worker/`

Quản lý background jobs qua **ARQ** (Async Redis Queue). Jobs được persist trong Redis — survive process restart và có thể chạy trên nhiều worker processes song song.

---

## Files

| File | Mô tả |
|---|---|
| `pool.py` | ARQ pool singleton — `init_pool`, `get_pool`, `close_pool` |
| `functions.py` | 3 ARQ task functions: `graph_ingest`, `consolidate`, `chat_memory` |
| `settings.py` | `WorkerSettings` — config cho `arq` CLI |

---

## Chạy worker

```bash
# Single worker process
arq src.agentrag.worker.settings.WorkerSettings

# Multiple workers (scale manual)
arq src.agentrag.worker.settings.WorkerSettings &
arq src.agentrag.worker.settings.WorkerSettings &

# Auto-scale theo queue depth (xem scaler.py ở root)
python scaler.py
```

---

## Job types

### `graph_ingest`

Parse, chunk, extract StructMem entries, và index một document vào Elasticsearch.

**Enqueue từ:** `ingestion/pipeline.py` sau khi lưu document (async mode)

```python
await get_pool().enqueue_job(
    "graph_ingest",
    document_id=str(doc_id),
    folder_path="/path/to/folder",
    source_id="filename.pdf",
    title="Document Title",
)
```

**Processing:** `graph.graph_jobs.process_graph_job()`
- Parse file → chunk (1536 tok) → StructMem extraction (2 parallel LLM calls) → index ES
- Cập nhật trạng thái document trong PostgreSQL (`processing` → `done` / `failed`)
- Nếu `total_chunks >= STRUCTMEM_CONSOLIDATION_THRESHOLD` → tự enqueue job `consolidate`

---

### `consolidate`

Cross-chunk synthesis — tổng hợp entries từ nhiều chunks thành higher-level hypotheses.

**Enqueue từ:** `graph_ingest` job (chained tự động)

```python
await arq_pool.enqueue_job(
    "consolidate",
    group_id="normalized_source_id",
    document_id=str(doc_id),
    trigger_chunk_count=42,
)
```

**Processing:** `graph.consolidation_jobs.process_consolidation_job()`
1. Lấy unconsolidated entries từ `agentrag_entries`
2. Embed buffer → cosine search → top-K historical seeds
3. LLM synthesis → cross-chunk hypotheses
4. Index vào `agentrag_synthesis`
5. Mark entries `consolidated=true`

---

### `chat_memory`

Extract dual-perspective memory entries từ một chat turn.

**Enqueue từ:** `main.py` sau mỗi `/chat` hoặc `/chat/stream` response (khi `CHAT_STRUCTMEM_ENABLED=true`)

```python
await get_pool().enqueue_job(
    "chat_memory",
    conversation_id="<uuid>",
    user_message="Câu hỏi người dùng",
    assistant_message="Câu trả lời assistant",
    turn_id="<id>",
    turn_timestamp="2024-01-01T00:00:00+00:00",
)
```

**Processing:** `chat.structmem.ChatMemoryService`
1. `process_turn()` — 2 parallel LLM calls (factual + relational) → embed → index
2. `count_unconsolidated()` — nếu ≥ `CHAT_MEMORY_CONSOLIDATION_THRESHOLD` → `consolidate()`

---

## `pool.py` — ARQ Pool Singleton

Được init một lần trong FastAPI lifespan, dùng lại trong toàn bộ app.

```python
# Khởi tạo (main.py lifespan)
await init_pool("redis://127.0.0.1:6379/0")

# Enqueue job (bất kỳ đâu trong app)
await get_pool().enqueue_job("graph_ingest", ...)

# Dọn dẹp (main.py lifespan shutdown)
await close_pool()
```

---

## `settings.py` — WorkerSettings

```python
class WorkerSettings:
    functions = [graph_ingest, consolidate, chat_memory]
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    max_jobs = settings.STRUCTMEM_MAX_CONCURRENCY  # concurrent jobs per worker
    job_timeout = settings.STRUCTMEM_CHUNK_TIMEOUT_SECONDS
    keep_result = 3600   # giữ job result 1 giờ
    max_tries = 2        # retry 1 lần khi fail
```

---

## Tương tác

| Module | Vai trò |
|---|---|
| `ingestion.pipeline` | Enqueue `graph_ingest` sau khi lưu document |
| `graph.graph_jobs` | Logic xử lý `graph_ingest` |
| `graph.consolidation_jobs` | Logic xử lý `consolidate` |
| `chat.structmem` | Logic xử lý `chat_memory` |
| `main.py` | Init pool trong lifespan; enqueue `chat_memory` sau chat turns |
| `scaler.py` | Quản lý số lượng worker processes theo queue depth |

---

## Config liên quan

| Key | Default | Mô tả |
|---|---|---|
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis connection (dùng chung cho cache + ARQ queue) |
| `STRUCTMEM_MAX_CONCURRENCY` | `1` | Max concurrent jobs per worker process |
| `STRUCTMEM_CHUNK_TIMEOUT_SECONDS` | `300` | Job timeout (giây) |
| `CHAT_STRUCTMEM_ENABLED` | `false` | Bật enqueue `chat_memory` jobs |
| `CHAT_MEMORY_CONSOLIDATION_THRESHOLD` | `10` | Số turns trước khi consolidate |
