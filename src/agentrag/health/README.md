# health — provider & infra reachability probe for the `/health/providers` endpoint

## Mục đích / Purpose
Module này thu thập một snapshot "khỏe mạnh" của các LLM/embedding provider và hạ tầng (Elasticsearch, Redis) mà backend đang cấu hình. Nó được gọi bởi endpoint `GET /health/providers` trong `main.py` để trả về: cấu hình settings có hợp lệ không (`validate_settings`), mỗi provider đã có token chưa, base_url là gì, và với provider chạy local (`ollama`) thì TCP có kết nối được không. Mục tiêu là giúp ops/dev nhanh chóng chẩn đoán cấu hình sai hoặc service down trước khi chạy ingest/chat.

It is a read-only diagnostic surface — it never mutates state and never performs real LLM calls. Reachability is checked only at the TCP-socket level.

## Plane
**Infrastructure / Execution Plane.** Pure IO + config introspection: nó đọc `Settings`, gọi `validate_settings`, và mở TCP socket. Không có prompt, không có LLM decision, không có branching nghiệp vụ — chỉ phản ánh trạng thái cấu hình/hạ tầng. Không đi qua `ServiceContainer`; được gọi trực tiếp như một function.

## Key files
| File | Responsibility |
|---|---|
| `providers.py` | Toàn bộ logic của module. Export `collect_provider_health(settings) -> dict`; các helper private build per-provider status, resolve base_url, kiểm tra token, và probe TCP socket. |

(Module chỉ có một file `.py`.)

## Public interface
Một entrypoint công khai duy nhất:

```python
from src.agentrag.health.providers import collect_provider_health
report = collect_provider_health(settings)  # settings: src.agentrag.config.Settings
```

`collect_provider_health(settings: Settings) -> dict` trả về:

```python
{
  "ok": bool,                      # True nếu validate_settings không raise
  "validation_error": str | None,  # message của exception (nếu có)
  "providers": {                   # luôn có "embedding", "extraction"; "agent" chỉ khi AGENT_PROVIDER set
    "embedding":  {role, provider, model, token_present, base_url, reachable, reachability_error},
    "extraction": {...},
    "agent":      {...},           # optional
  },
  "infra": {
    "elasticsearch": {reachable, error},
    "redis":         {reachable, error},  # chỉ có khi REDIS_URL được set
  },
}
```

Truy cập: **import trực tiếp** function (không qua container, không qua Protocol). Caller duy nhất hiện nay là `main.py::provider_health()`, map vào `GET /health/providers`; nếu `report["ok"]` là False thì endpoint raise `HTTPException(400, detail=report)`.

Private helpers (không export ra ngoài module):
- `_provider_status(role, provider, model, base_url, settings)` — build dict cho một provider.
- `_resolve_base_url(provider, explicit_base_url, settings)` — chọn base_url ưu tiên `*_BASE_URL` rồi fallback theo provider.
- `_token_present(provider, settings)` — kiểm tra biến môi trường key tương ứng có giá trị.
- `_socket_status(url_or_uri)` — mở `socket.create_connection` với timeout 1.5s; suy luận port nếu URL không có.

## Data flow
Inputs → `Settings` (object cấu hình toàn cục từ `config.py`).

Steps:
1. Gọi `validate_settings(settings)` (từ `config_validation.py`); bắt mọi `Exception` thành `validation_error` để `ok=False` thay vì làm crash request.
2. Build status cho `embedding` và `extraction`, và `agent` nếu `settings.AGENT_PROVIDER` được set (model fallback `AGENT_MODEL or EXTRACTION_MODEL`).
3. Với mỗi provider: xác định `token_present`, resolve `base_url`, và set `reachable`:
   - `ollama` (có base_url): TCP-probe base_url thật.
   - `openai` / `gemini` / `hf_inference`: `reachable = None` (không probe các SaaS endpoint).
4. Probe infra: luôn `elasticsearch` (`ELASTICSEARCH_URL`); thêm `redis` nếu `REDIS_URL` set.
5. Trả dict (không raise — quyết định HTTP status do caller `main.py` lo).

Outputs → JSON dict (ở trên).

Upstream caller: `main.py`. Downstream dependencies: `src.agentrag.config.Settings`, `src.agentrag.config_validation.validate_settings`, và stdlib `socket` / `urllib.parse.urlparse`. Module này **không** phụ thuộc vào agent/retrieval/ingestion — đứng độc lập.

## Config
Đọc (không ghi) các field sau từ `src/agentrag/config.py::Settings`:

| Nhóm | Settings được đọc |
|---|---|
| Embedding | `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_BASE_URL` |
| Extraction | `EXTRACTION_PROVIDER`, `EXTRACTION_MODEL`, `EXTRACTION_BASE_URL` |
| Agent (optional) | `AGENT_PROVIDER`, `AGENT_MODEL`, `AGENT_BASE_URL` |
| Token keys | `OPENAI_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`, `OLLAMA_API_KEY` |
| Base-url fallback | `OLLAMA_BASE_URL`, `HF_OPENAI_BASE_URL` (gemini dùng URL hard-coded `https://generativelanguage.googleapis.com/v1beta/openai/`) |
| Infra | `ELASTICSEARCH_URL`, `REDIS_URL` |

## Gotchas
- **`reachable=None` không phải lỗi.** Với `openai`/`gemini`/`hf_inference`, module cố ý không probe (tránh phụ thuộc mạng ra ngoài và đếm vào rate-limit); `None` nghĩa là "chưa kiểm tra", chỉ `ollama` mới có `True/False` thật.
- **`token_present` chỉ kiểm tra biến môi trường có giá trị — không validate token đúng/sai.** Token rác vẫn báo `True`.
- **TCP probe ≠ health thật.** `_socket_status` chỉ mở socket; ES/Redis có thể bắt port nhưng vẫn unhealthy ở tầng ứng dụng. Timeout 1.5s/host nên endpoint có thể chậm ~vài giây nếu nhiều host down.
- **`ok` chỉ phản ánh `validate_settings`**, không phản ánh `reachable`. Một provider unreachable vẫn cho `ok=True` (miễn cấu hình hợp lệ); chỉ validation error mới làm `ok=False` → `main.py` trả HTTP 400.
- **Port inference**: `_socket_status` suy ra port từ scheme (http→80, https→443, bolt/neo4j→7687). Scheme lạ + không có port → `{"reachable": None, "error": "could not infer port"}` (hiện chưa có caller neo4j/bolt, nhánh đó là dự phòng).
- Provider chưa biết trong `_token_present` mặc định trả `False` (không có nhánh → mọi provider ngoài 4 cái đã liệt kê đều coi như thiếu token).
