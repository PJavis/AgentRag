# common — cross-cutting utilities with no business dependencies

## Mục đích / Purpose
`common/` chứa các tiện ích dùng chung, không phụ thuộc vào bất kỳ module nghiệp vụ nào
(agent / retrieval / ingestion / structured đều import vào, không import ngược ra).
Phạm vi: observability LLM (Langfuse + Phoenix tracing), per-stage latency tracing,
strip chain-of-thought của reasoning models, ingest progress pub/sub, và document-level
access policy. Mọi thứ ở đây hoặc là pure stdlib hoặc gọi IO nhẹ (Valkey/Redis) một cách
best-effort — không có prompt, không có decision logic.

## Plane
**Infrastructure.** Đây là lớp tiện ích cross-cutting: nó không quyết định *what to do*
(Reasoning) cũng không phải một service IO facade trong `ServiceContainer` (Execution).
Các module ở cả hai plane import trực tiếp từ đây (`from src.agentrag.common.* import ...`).

## Key files
| File | Responsibility |
|---|---|
| `langfuse_client.py` | Single chokepoint dựng `AsyncOpenAI`. `make_async_openai()` trả về client `langfuse.openai` đã auto-trace khi `LANGFUSE_ENABLED`, ngược lại trả `openai.AsyncOpenAI` thuần. Cộng `init_langfuse()` / `langfuse_flush()` cho app lifespan. |
| `phoenix_client.py` | `init_phoenix()` — đăng ký OTEL tracer trỏ tới Phoenix collector + auto-instrument OpenAI qua OpenInference. Chạy song song Langfuse, mặc định OFF. |
| `tracing.py` | `StageTracer` / `StageEvent` — in-memory per-stage latency + metadata tracer, zero external dep. Feed vào `timings_ms` của response. |
| `thinking.py` | `parse_thinking_content()` / `clean_thinking_content()` — strip `<think>…</think>` (và `<thought>…</thought>`) khỏi output của reasoning models. |
| `progress.py` | `publish_progress()` / `channel_for()` — ingest progress pub/sub qua Valkey/Redis cho SSE stream. Best-effort, không bao giờ raise. |
| `security_policy.py` | `PolicyRegistry` + `DocumentPolicy` — document/section-level access policy (deny prefix/regex/segment-type, cap `max_results`). Không cần user model. |
| `__init__.py` | Rỗng — `common` là namespace, không re-export gì. |

## Public interface
Truy cập bằng **direct import** (không qua `ServiceContainer`, không qua `services/protocols.py`).

**`langfuse_client.py`**
- `make_async_openai(**kwargs) -> AsyncOpenAI` — gọi bởi mọi nơi dựng OpenAI client:
  `agent/llm.py`, `services/llm_gateway.py`, `retrieval/reranker.py`.
- `init_langfuse() -> None` — export credentials từ `settings` vào `os.environ` (SDK chỉ đọc env). Gọi ở `main.py` lifespan startup.
- `langfuse_flush() -> None` — flush buffered traces ở shutdown. Gọi ở `main.py`.

**`phoenix_client.py`**
- `init_phoenix() -> None` — idempotent (guard `_REGISTERED`). Gọi ở `main.py` startup.

**`tracing.py`** — `StageTracer(request_id: str | None = None)` với
`start(stage, service, **metadata)`, `end(stage, **metadata) -> StageEvent`,
`fail(stage, error: Exception, **metadata) -> StageEvent`, `as_dict()`,
`as_timings_dict() -> dict[str, float]`, `total_elapsed_ms()`. Dùng bởi `structured/pipeline.py`.

**`thinking.py`** — `clean_thinking_content(content) -> str` (dùng trong `agent/llm.py` ×3, `services/llm_gateway.py`), `parse_thinking_content(content) -> tuple[thinking, cleaned]`.

**`progress.py`** — `await publish_progress(user_id, source_id, stage)` và `channel_for(user_id) -> str`. Publishers: `ingestion/pipeline.py`, `graph/graph_jobs.py`. Consumer: `adapter/routers/sources.py` (SSE `/sources/progress/stream`).

**`security_policy.py`** — `PolicyRegistry()` với `load_from_list(list[dict])`, `get(title) -> DocumentPolicy | None`, `has_policy`, `all_titles`. `DocumentPolicy` expose `matches_denied_section(section_path) -> bool`, `matches_denied_segment_type(segment_type) -> bool`, field `max_results`. Wrapped bởi `services/security_service.py::SecurityService`.

## Data flow
- **Tracing observability**: `main.py` lifespan → `init_langfuse()` + `init_phoenix()` → mọi LLM call qua `make_async_openai()` được trace. Shutdown → `langfuse_flush()`.
- **StageTracer**: caller (vd `structured/pipeline.py`) `start`/`end`/`fail` quanh từng stage → `as_timings_dict()` → gắn vào field `timings_ms` của API response.
- **thinking**: LLM raw output (chứa `<think>…`) → `clean_thinking_content()` → user-visible answer trong `agent/llm.py` / `llm_gateway.py`.
- **progress**: worker (`ingestion/pipeline.py`, `graph/graph_jobs.py`) → `publish_progress()` → Valkey channel `ingest:progress:{user_id}` → SSE relay trong `sources.py` → UI refresh.
- **security_policy**: startup load policies → `SecurityService` gọi `matches_denied_*` + `max_results` để filter tool/retrieval results.

## Config
Các flag từ `src/agentrag/config.py` mà module này đọc (tất cả default-OFF trừ `REDIS_URL`):

| Setting | Default | Đọc ở |
|---|---|---|
| `LANGFUSE_ENABLED` | `False` | `langfuse_client.py` (init / make / flush đều no-op khi OFF) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | `None` | `init_langfuse()` |
| `LANGFUSE_HOST` | `http://localhost:3000` | `init_langfuse()` |
| `PHOENIX_ENABLED` | `False` | `phoenix_client.py` |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006/v1/traces` | `init_phoenix()` |
| `PHOENIX_PROJECT` | `agentrag` | `init_phoenix()` |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | `progress.py` (publish skip nếu falsy) |

## Recent additions (2026-06)
- **Langfuse + Phoenix tracing** (`langfuse_client.py`, `phoenix_client.py`) là phần của nhánh `feat/ragas-langfuse-reranker`. Cả hai default-OFF; `make_async_openai()` là chokepoint duy nhất để swap traced client mà phần còn lại của codebase không đổi. Cost ledger (`observability/cost.py`) giữ độc lập — Langfuse cho trace/latency UI, ledger cho USD aggregates.
- `StageTracer.as_timings_dict()` feed vào `timings_ms` của response — cùng cơ chế chứa key `critique` mà các node CRAG mới ghi vào.

## Gotchas
- **Existing README cũ trỏ `src/pam/common/` và mô tả API không tồn tại** (`tracer.stage()` context manager, `to_dict()`, `PolicyRegistry.is_allowed()`, `filter_results()`). README này đã sửa: API thật là `start/end/fail` + `as_dict`/`as_timings_dict`, và filtering nằm ở `SecurityService`, không phải ở `PolicyRegistry`.
- `make_async_openai()` import `langfuse.openai` **lazy** trong hàm; nếu `LANGFUSE_ENABLED=True` nhưng package chưa cài, nó log warning và fallback client thuần (không crash).
- `init_langfuse()` phải chạy **trước** LLM call đầu tiên: Pydantic load key vào `settings` nhưng KHÔNG populate `os.environ`, mà Langfuse SDK chỉ đọc `os.environ`. Dùng `os.environ.setdefault` nên env có sẵn sẽ thắng settings.
- `init_phoenix()` idempotent qua module-global `_REGISTERED`; gọi nhiều lần an toàn.
- `publish_progress()` nuốt mọi exception (best-effort) — publish lỗi không bao giờ làm hỏng ingest; mất event là chấp nhận được. Mỗi call tự mở rồi `aclose()` một `Redis` client.
- `clean_thinking_content()` bỏ qua content > 100KB (`_MAX_BYTES`) để tránh regex backtracking bệnh lý — payload lớn trả nguyên văn, không strip. Cũng xử lý output méo `content</think>` thiếu thẻ mở (Nemotron).
- `DocumentPolicy.__post_init__` compile regex một lần (`re.IGNORECASE`); `PolicyRegistry` read-only sau init nên thread-safe — đừng mutate sau khi load.
- `langfuse_flush()` dùng v2 SDK path (`langfuse.openai.openai.flush_langfuse()`), fallback `Langfuse().flush()`; cả hai bọc try/except nên không vỡ shutdown.
