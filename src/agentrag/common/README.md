# Module: `common` — Shared Utilities

**Vị trí:** `src/pam/common/`

Utilities dùng chung toàn hệ thống — không phụ thuộc vào bất kỳ module nghiệp vụ nào.

---

## Files

| File | Class / Function | Mô tả |
|---|---|---|
| `tracing.py` | `StageTracer` | Đo thời gian xử lý từng stage trong một request |
| `security_policy.py` | `PolicyRegistry`, `DocumentPolicy` | Access control — giới hạn context theo document_title |

---

## `StageTracer`

Đo latency theo từng bước (stage) trong pipeline, tương thích với `timings_ms` field ở response.

```python
tracer = StageTracer()
with tracer.stage("retrieval"):
    ...
with tracer.stage("answer"):
    ...
timings = tracer.to_dict()  # {"retrieval": 123.4, "answer": 56.7}
```

---

## `SecurityPolicy` / `PolicyRegistry`

Lọc kết quả retrieval theo document_title để ngăn thông tin cross-document rò rỉ.

- `DocumentPolicy` — policy cho một document: danh sách user/group được phép truy cập
- `PolicyRegistry` — tập hợp policies, expose `is_allowed(document_title, user_context)` và `filter_results(results, document_title)`

Dùng bởi `SecurityService` ở lớp service.

---

## Tương tác

| Module | Vai trò |
|---|---|
| `services.SecurityService` | Dùng `PolicyRegistry` để filter tool results |
| `agent.AgentService` | Dùng `StageTracer` (gián tiếp qua timings_ms) |
