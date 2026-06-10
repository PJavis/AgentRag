# worker — ARQ background-job runtime (enqueue, run, scale)

## Mục đích / Purpose

Quản lý **background jobs** qua [ARQ](https://arq-docs.helpmanual.io/) (Async Redis
Queue). Module này không chứa business logic — nó chỉ là **runtime glue**: định nghĩa
4 task functions (mỗi loại job một hàm), giữ một ARQ Redis pool singleton để API process
enqueue jobs, và khai báo `WorkerSettings` cho `arq` CLI khởi động worker process. Jobs
được persist trong Redis nên survive process restart và có thể chạy song song trên nhiều
worker processes. Logic thật nằm ở các module downstream (`graph.*`, `chat.structmem`).

## Plane

**Infrastructure.** Đây là job-queue plumbing: nó move work off the request path và
delegate sang các module khác. Bản thân `functions.py` chỉ unpack kwargs thành dataclass
job rồi gọi `process_*` — không tự ra quyết định (Reasoning) và không tự làm IO nghiệp vụ
(Execution); các plane đó nằm ở module được gọi.

## Key files

| File | Responsibility |
|---|---|
| `functions.py` | 4 ARQ task functions: `graph_ingest`, `vision_extract`, `consolidate`, `chat_memory`. Mỗi hàm `async def name(ctx, *, ...kwargs)`; chỉ build job dataclass + gọi downstream `process_*`. Lazy-imports để worker boot nhanh. |
| `settings.py` | `WorkerSettings` — class mà `arq` CLI đọc: danh sách `functions`, `redis_settings`, `max_jobs`, `job_timeout`, `keep_result`, `max_tries`. |
| `pool.py` | ARQ Redis pool singleton: `init_pool()`, `get_pool()`, `close_pool()`. Dùng bởi API process để enqueue. |
| `__init__.py` | Rỗng (package marker). |

## Public interface

**Pool (gọi từ API process — `main.py`, `ingestion/pipeline.py`):** import trực tiếp, KHÔNG
qua `ServiceContainer`.

```python
from src.agentrag.worker.pool import init_pool, get_pool, close_pool

await init_pool(redis_url)                     # 1 lần trong FastAPI lifespan startup
await get_pool().enqueue_job("graph_ingest", document_id=..., ...)   # bất kỳ đâu
await close_pool()                             # lifespan shutdown
```

`get_pool()` raises `RuntimeError` nếu chưa `init_pool()`.

**Task functions** không được gọi trực tiếp; chúng được **đặt tên (string) trong
`enqueue_job(...)`** và do worker process invoke. Chữ ký:

```python
async def graph_ingest(ctx, *, document_id, folder_path, source_id, title,
                       parsed_cache_path=None) -> None
async def vision_extract(ctx, *, document_id, title, image_records) -> None
async def consolidate(ctx, *, group_id, document_id, trigger_chunk_count) -> None
async def chat_memory(ctx, *, conversation_id, user_message, assistant_message,
                      turn_id, turn_timestamp) -> None
```

`ctx["redis"]` là ArqRedis pool bên trong worker — dùng để chain jobs (xem `graph_ingest`).

**Start a worker process:**

```bash
arq src.agentrag.worker.settings.WorkerSettings
```

`scaler.py` (repo root) quản lý số worker processes theo queue depth.

## Data flow

```
API process (enqueue, không block request)        Worker process (arq CLI)
────────────────────────────────────────         ──────────────────────────────
ingestion/pipeline.py ─ "graph_ingest"  ┐
main.py /chat,/chat/stream ─ "chat_memory"│  Redis   functions.py → process_*(job)
ingestion/pipeline.py ─ "vision_extract" ┘ ───────▶  graph_jobs / vision_jobs /
graph_jobs.py ─ "consolidate" (chained)               consolidation_jobs / chat.structmem
```

- **`graph_ingest`** — enqueue từ `ingestion/pipeline.py` (async mode) sau khi lưu document.
  → `graph.graph_jobs.process_graph_job(job, arq_pool=ctx["redis"])`: parse → chunk →
  StructMem extraction → index ES, cập nhật `Document` status trong PostgreSQL. Nếu
  `total_chunks >= STRUCTMEM_CONSOLIDATION_THRESHOLD` thì **chained-enqueue** `consolidate`.
- **`vision_extract`** — enqueue từ `ingestion/pipeline.py` khi `VISION_PROVIDER` được set và
  `VISION_INGEST_MODE=async`. → `graph.vision_jobs.process_vision_job(job)`: describe page-image
  bitmaps qua vision LLM (RPM/concurrency-bounded, batch-flush), upsert image segments vào PG + ES.
- **`consolidate`** — chained từ `graph_ingest` (không enqueue trực tiếp từ API).
  → `graph.consolidation_jobs.process_consolidation_job(job)`: lấy unconsolidated entries →
  embed + cosine search seeds → LLM synthesis cross-chunk hypotheses → index `agentrag_synthesis`
  → mark `consolidated=true`.
- **`chat_memory`** — enqueue từ `main.py` sau mỗi `/chat` và `/chat/stream` response khi
  `CHAT_STRUCTMEM_ENABLED=true`. → `chat.structmem.ChatMemoryService.process_turn()` (2 parallel
  LLM calls: factual + relational → embed → index). Hàm này tự gọi `count_unconsolidated()` và,
  nếu `>= CHAT_MEMORY_CONSOLIDATION_THRESHOLD`, gọi `svc.consolidate(conversation_id)` **in-line**
  (không qua một ARQ job riêng).

## Config

Đọc từ `src/agentrag/config.py` (qua `app_settings` / `settings`):

| Key | Default | Đọc ở đâu | Mô tả |
|---|---|---|---|
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | `settings.py` (+ `pool.py` caller) | Redis dùng chung cho cache + ARQ queue |
| `STRUCTMEM_WORKER_MAX_JOBS` | `1` | `settings.py` → `max_jobs` | Số document jobs đồng thời / worker process |
| `STRUCTMEM_JOB_TIMEOUT_SECONDS` | `3600` | `settings.py` → `job_timeout` | Timeout / job (cả document, không phải 1 chunk) |
| `CHAT_MEMORY_CONSOLIDATION_THRESHOLD` | `10` | `functions.py` (`chat_memory`) | Ngưỡng unconsolidated entries trước khi consolidate |

Các flag liên quan nhưng đọc ở **module downstream**, không phải trong `worker/` (để tham khảo):
`STRUCTMEM_MAX_CONCURRENCY` (1), `STRUCTMEM_CHUNK_TIMEOUT_SECONDS` (300),
`STRUCTMEM_CONSOLIDATION_THRESHOLD` (20, dùng trong `graph_jobs.py` để quyết định chain `consolidate`),
`VISION_PROVIDER` / `VISION_INGEST_MODE` / `VISION_MAX_CONCURRENCY` (4) / `VISION_MAX_RPM` (10) /
`VISION_PER_IMAGE_RETRIES` (3) / `VISION_FLUSH_BATCH_SIZE` (10), `CHAT_STRUCTMEM_ENABLED` (`True`).

## Recent additions (2026-06)

Không có. Module `worker/` **không bị chạm** bởi đợt RAG-enhancement / UI-signal gần đây
(Contextual Retrieval, RAPTOR, CRAG critique/corrective, adaptive routing, semantic cache).
Những tính năng đó nằm trong agent/ingestion/retrieval path, không qua background queue này.

## Gotchas

- **Hai process khác nhau.** API server (`uvicorn main:app`) chỉ **enqueue**; phải chạy
  `arq ...WorkerSettings` (hoặc `scaler.py`) ở process riêng để jobs thực sự được run. Quên
  worker → jobs nằm yên trong Redis, document kẹt ở status `processing`.
- **Lazy imports trong `functions.py` là cố ý** — giữ import nặng (graph/chat) ra khỏi
  module top-level để worker boot nhanh và tránh import cycles; đừng "dọn" lên đầu file.
- **`max_jobs=1` mặc định** để an toàn với local Ollama (1 GPU/CPU-bound LLM tại một thời điểm).
  Tăng `STRUCTMEM_WORKER_MAX_JOBS` chỉ khi backend LLM chịu được concurrency.
- **`max_tries=2`** → mỗi job retry đúng **1 lần** khi fail. Downstream `process_*` phải
  idempotent-an-toàn (re-parse / re-index) vì có thể chạy lại.
- **`consolidate` không phải lúc nào cũng chạy** — chỉ chained khi document đủ lớn
  (`total_chunks >= STRUCTMEM_CONSOLIDATION_THRESHOLD`). Document nhỏ sẽ không có synthesis layer.
- **`job_timeout` bao trùm cả document**, không phải 1 chunk; PDF nhiều hình + vision LLM chậm
  có thể chạm `STRUCTMEM_JOB_TIMEOUT_SECONDS` (3600s) — chỉnh lên nếu cần.
