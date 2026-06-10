# observability — LLM cost ledger + user activity event log

## Mục đích / Purpose
Hai cơ chế ghi-nhận (telemetry) độc lập, best-effort, phục vụ dashboard và audit:
1. **`cost.py`** — sổ cái (ledger) chi phí/độ trễ cho **mọi LLM call thành công**. Mỗi call đẩy một entry vào Valkey/Redis stream `agentrag:llm:calls:v1` (cap `MAXLEN ~ 5000`), sống sót qua restart và gộp được trên nhiều worker. Tổng hợp ra per-task / per-model stats (calls, tokens, USD, latency p50/p95).
2. **`activity.py`** — ghi một dòng `event_log` (S6) cho mỗi hành động người dùng (chat turn, upload, search…) để dựng feed hoạt động.

Cả hai đều **không bao giờ làm vỡ business path**: lỗi được log rồi nuốt; cost ghi vào deque in-process khi Valkey không tới được.

## Plane
**Infrastructure / cross-cutting telemetry.** Không thuộc Reasoning hay Execution plane — đây là side-channel ghi nhận. Không đi qua `ServiceContainer`; các module gọi trực tiếp bằng import hàm module-level. Cost tracking là fire-and-forget từ Execution plane (LLM workers); activity là inline-await từ API routers (adapter layer).

## Key files
| File | Responsibility |
|------|----------------|
| `cost.py` | LLM cost/latency ledger. Valkey stream backend + in-process deque fallback; token estimate hoặc provider usage; USD theo bảng giá Gemini/OpenAI; tổng hợp summary + recent feed. |
| `activity.py` | `record_event()` — INSERT một dòng `event_log` (S6) per user action. Best-effort, never raises. |
| `__init__.py` | Empty — module có namespace, không re-export. |

## Public interface

### `cost.py` (sync, gọi trực tiếp)
```python
record_llm_call(*, task: str, model: str, latency_ms: float,
                in_text: str = "", out_text: str = "", usage: Any = None) -> None
cost_summary() -> dict[str, Any]
recent_calls(limit: int = 50, since: float | None = None) -> list[dict[str, Any]]
reset_ledger() -> None
```
- `record_llm_call` là **no-op khi `settings.LLM_COST_TRACKING_ENABLED` = False** (default). Gọi từ `agent/llm.py` (`AgentLLM`, 4 call sites) và `services/llm_gateway.py` (`LLMGateway`, 2 call sites). Nếu `usage` (provider object có `prompt_tokens`/`completion_tokens`) được truyền thì ưu tiên; ngược lại fallback char-density estimate. `usage_source` field đánh dấu `"provider"` hay `"estimate"`.
- `cost_summary` / `recent_calls` / `reset_ledger` được expose qua router `adapter/routers/config.py`: `GET /metrics/cost`, `GET /metrics/cost/recent`, `POST /metrics/cost/reset`. `LLMGateway.cost_summary()` cũng forward thẳng tới `cost_summary()`.

### `activity.py` (async, gọi trực tiếp)
```python
async record_event(user_id, event_type: str, *,
                   target_kind: str | None = None,
                   target_id = None,
                   payload: dict | None = None) -> None
```
- Caller **await** trực tiếp (sync-inline, ~1ms INSERT). `user_id`/`target_id` chấp nhận `uuid.UUID | str | None`; `"anonymous"` và string không hợp lệ → `NULL` (xem `_coerce_uuid`). Callers: `adapter/routers/chat.py`, `search.py`, `sources.py`. Feed đọc lại qua `adapter/routers/activity.py`.

## Data flow

**Cost ledger:**
LLM worker (`AgentLLM` / `LLMGateway`) đo `latency_ms` quanh call → `record_llm_call(...)` → tính tokens (provider usage hoặc `_estimate_tokens`) + USD (`_price_for`) → append vào `_LEDGER` (deque) → best-effort persist vào Valkey stream: nếu đang trong event loop dùng `loop.create_task(_stream_xadd_async)`, ngược lại `_stream_xadd_sync`. Đọc ra (`_read_entries`) ưu tiên `XRANGE` trên stream, fallback deque. `cost_summary` gộp per-task/per-model với p50/p95 latency.

**Activity:** Router handler → `await record_event(...)` → `EventLog` row (`database/models.py`) → INSERT qua `AsyncSessionLocal`. Đọc lại qua bảng `event_log`.

Downstream deps: `src.agentrag.config.settings`, `redis`/`redis.asyncio` (cost); `src.agentrag.database.AsyncSessionLocal` + `database.models.EventLog` (activity).

## Config (`src/agentrag/config.py`)
| Flag | Default | Effect |
|------|---------|--------|
| `LLM_COST_TRACKING_ENABLED` | `False` | Gate cho `record_llm_call`. Off → no-op, ledger trống. |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Backend stream của cost ledger. `None`/unreachable → latch `_VALKEY_DISABLED`, dùng deque in-process. |

`activity.py` không đọc settings trực tiếp (chỉ phụ thuộc DB session).

## Gotchas
- **Cost tracking default OFF.** `cost_summary` trả rỗng đến khi `LLM_COST_TRACKING_ENABLED=true`. Đây là nguyên nhân thường gặp khi dashboard không có số.
- **In-process fallback không gộp giữa các worker.** Khi Valkey unreachable, `_VALKEY_DISABLED` latch `True` (không tự retry trong process đó), mỗi worker chỉ thấy deque riêng → dashboard có thể partial. `summary["backend"]` cho biết `"valkey"` hay `"in-process"`.
- **`reset_ledger` xoá cả hai** (deque + `DELETE` stream key) — destructive, dùng cho dev/test.
- **USD chỉ là ước lượng.** Pricing hard-code trong `_PRICE_PER_1M`; model lạ fallback về `gemini-2.5-flash`. Token có thể là char-density heuristic (`~4 chars/tok` ASCII, `~1.8` cho VN/CJK) nếu provider không trả `usage`.
- **Async persist là fire-and-forget** (`loop.create_task`), không await — entry chắc chắn vào deque nhưng việc ghi stream có thể fail im lặng (chỉ `logger.warning`).
- **`record_event` không phân loại `event_type`** — không có enum/validation; caller tự quy ước string. `event_type` cap `String(32)` trong model.
- Module **không touch** các tính năng RAG mới 2026-06 (Contextual Retrieval, RAPTOR, CRAG, semantic cache, adaptive routing); không có flag nào của chúng đọc ở đây.
