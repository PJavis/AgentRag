# services — Execution Plane facades + DI container (S4)

## Mục đích / Purpose

`services/` là **Execution Plane** của AgentRag: mọi IO/LLM/embedding/vision/storage/retrieval
đi qua các facade ổn định ở đây. Reasoning Plane (`agent/`, `orchestration/`)
**không** tự khởi tạo concrete class — chỉ fetch service qua `ServiceContainer` (DI singleton)
và type-hint chống lại `Protocol` trong `protocols.py`. Module này cũng chứa hai ngoại lệ có chủ đích:
`reasoning_knowledge.py` (pure Reasoning helpers — đặt ở đây vì lịch sử) và `semantic_cache.py`
(một data-structure thuần, dùng bởi retrieval). Xem `ARCHITECTURE.md` (root) cho luật chia plane.

## Plane

**Execution Plane** (chủ đạo). Hai file lệch khỏi quy ước:
`reasoning_knowledge.py` là **Reasoning Plane** (pure, no IO);
`semantic_cache.py` là **Infrastructure** (in-memory data structure, không gọi IO).

## Key files

| File | Responsibility |
|---|---|
| `container.py` | `ServiceContainer` — DI singleton, lazy-init mỗi service. `get_container()` / `set_container()` / `reset_container()` + `override()` cho test. |
| `protocols.py` | `EmbeddingProtocol`, `VisionProtocol`, `RetrievalProtocol`, `StorageProtocol`, `RerankerProtocol`, `LLMProtocol` — `runtime_checkable` contract cho Reasoning code type-hint. |
| `llm_gateway.py` | `LLMGateway` — điểm gọi LLM duy nhất. Task-based routing, large-context auto-route, vision multimodal, cost tracking. |
| `embedding_service.py` | `EmbeddingService` — dense-embedding facade + TTL cache keyed SHA-256(text); thoả `EmbeddingProtocol`. |
| `retrieval_service.py` | `RetrievalService` — hybrid-search facade, wrap `FederatedRetriever`. Filters-only; routing là việc của Reasoning. |
| `storage_service.py` | `StorageService` — read-mostly CRUD facade trên Postgres + Elasticsearch. |
| `vision_service.py` | `VisionService` — wrap `ImageParser`. `enabled=False` khi `VISION_PROVIDER` chưa set. |
| `context_assembly_service.py` | `ContextAssemblyService` — thin facade trên `agent.context.ContextAssembler` (dedup + rank + trim + reorder). |
| `security_service.py` | `SecurityService` — query-time validation + result filtering theo `DocumentPolicy`. |
| `semantic_cache.py` | `SemanticCache` — tier-2 retrieval cache keyed by query-embedding cosine similarity (LRU + TTL). |
| `knowledge_service.py` | `KnowledgeService` — legacy retrieval+tool facade (HyDE, decompose/RRF, intent-mode). Vẫn dùng bởi MCP; đừng mở rộng cho code mới. |
| `reasoning_knowledge.py` | Pure Reasoning helpers (no IO): `expand_query`, `select_retrieval_mode`, `mode_to_tool`, `normalize_tool_call`. |

## Public interface

Hầu hết truy cập **qua container**, không import concrete trực tiếp:

```python
from src.agentrag.services.container import get_container
c = get_container()

payload, latency = await c.llm.json_response(system, user, task="answer")
vecs  = await c.embedding.embed(["query a", "query b"])
hits  = await c.retrieval.search(query=q, mode="hybrid_kg", top_k=8)
route = await c.domain_router.classify(q)            # Reasoning service, hosted here for singleton sharing
docs  = await c.storage.list_documents(limit=20)
if c.vision.enabled:
    desc = await c.vision.describe(image_bytes, mime="image/jpeg")
```

Container properties (lazy): `.llm` → `LLMGateway`, `.embedding` → `EmbeddingService`,
`.vision` → `VisionService`, `.storage` → `StorageService`, `.retrieval` → `RetrievalService`,
`.domain_router` → `DomainRouter` (instanced ở đây để singleton-share giữa các caller).

`LLMGateway` (called by `agent/`, `generation/`, `orchestration/`, `ingestion/`):

| Method | Mô tả |
|---|---|
| `json_response(system, user, task="general") -> (dict, latency_ms)` | LLM → parse JSON; đo latency. |
| `json_response_multimodal(system, user_text, image_urls, task) -> (dict, latency_ms)` | Text + image URLs → routes to `settings.VISION_ANSWER_MODEL` (NOT the text answer model). Empty VISION_ANSWER_MODEL → raises `VisionDisabledError` (caller falls back to text). |
| `text_response(system, user, task="general") -> str` | Plain-text completion (dùng bởi `contextualizer`, `raptor`, fast-path `graph_service`). |
| `vision_response(system, text, image_bytes, mime, task="vision") -> (str, latency_ms)` | Single-image vision (dùng bởi `ImageParser`). |
| `vision_response_batch(system, text, images, task="vision") -> list[str]` | N ảnh trong 1 call → tiết kiệm RPM. |
| `cost_summary() -> dict` | Delegate `observability.cost.cost_summary()`. |

Test pattern (mocks):

```python
from src.agentrag.services.container import ServiceContainer
c = ServiceContainer()
c.override(retrieval=mock_retrieval, embedding=mock_embed)   # keys = property names; unknown key raises KeyError
```

`__init__.py` dùng PEP 562 lazy re-export (`ContextAssemblyService`, `KnowledgeService`,
`LLMGateway`, `SecurityService`) để import một leaf như `semantic_cache` không kéo theo
`knowledge_service → agent.tools → elasticsearch_retriever` (tránh circular import).

## Data flow

- **Reasoning → Execution:** `agent.service` fetch services từ container,
  gọi `retrieval.search()` (sau khi `domain_router.classify()` ra `system_override`), assemble
  context qua `context_assembly_service`, sinh answer qua `llm.json_response()`.
- `RetrievalService.search()` chấp nhận **hoặc** generic `filters={"systems":[…],"specialties":[…]}`
  **hoặc** explicit `system_override`/`specialty_override` (S5 UI form); nó merge `filters` →
  override rồi forward xuống `FederatedRetriever` (router=None — không tự route).
- `EmbeddingService.embed()` cache batch nhỏ (≤ `cache_max_batch`=8); batch lớn (ingestion) bypass.
  Hot paths hưởng lợi: HyDE rewrite + dense kNN trên cùng text, repeated sub-queries trong decide loop.
- `LLMGateway._resolve_client()` chọn client theo thứ tự ưu tiên: (1) large-context auto-route khi
  ước lượng token > `LLM_LARGE_CONTEXT_THRESHOLD`, (2) task routing nếu `LLM_ROUTING_ENABLED` +
  task ∈ `LLM_TASK_MODEL_MAP`, (3) default `AgentLLM`. Mọi call record cost qua `observability.cost`.

## Config

`settings.*` mà module này đọc (từ `src/agentrag/config.py`):

| Key | Default | Đọc bởi |
|---|---|---|
| `LLM_ROUTING_ENABLED` | `False` | `llm_gateway` — bật per-task model routing |
| `LLM_TASK_MODEL_MAP` | `"{}"` | `llm_gateway` — JSON map task→model (tasks: classify, decide, plan, answer, mindmap, summary, domain_router, followup, starter, …) |
| `LLM_LARGE_CONTEXT_MODEL` / `LLM_LARGE_CONTEXT_THRESHOLD` | `None` / `100000` | `llm_gateway` — auto-switch khi prompt vượt threshold tokens |
| `LLM_FALLBACK_MODEL` | `"qwen2.5:7b-instruct"` | `llm_gateway.vision_response` — fallback khi model not found |
| `LLM_COST_TRACKING_ENABLED` | `False` | gate cho ledger + `/metrics/cost` (record ở `observability.cost`) |
| `VISION_PROVIDER` / `VISION_MODEL` / `VISION_BASE_URL` | `None` | `vision_service`, `llm_gateway._get_vision_client` (fallback `EXTRACTION_PROVIDER`/`EXTRACTION_MODEL`) |
| `VISION_TIMEOUT_SECONDS` | `180` | `llm_gateway` — llava cold-start tolerance |
| `AGENT_MAX_OUTPUT_TOKENS` | `131072` | `llm_gateway` — `max_tokens` cho vision calls |
| `QUERY_REWRITE_ENABLED` / `QUERY_REWRITE_HYDE` / `QUERY_REWRITE_DECOMPOSE` | `False` / `True` / `False` | `knowledge_service` — HyDE + multi-hop decompose |
| `AGENT_TOOL_TOP_K` | `5` | `knowledge_service`, `reasoning_knowledge` — default top_k |
| `RETRIEVAL_RRF_K` | `60` | `knowledge_service._decomposed_search` — RRF constant |
| `SEMANTIC_CACHE_ENABLED` | `False` | gate (read trong `elasticsearch_retriever`, không trong `semantic_cache.py`) |
| `SEMANTIC_CACHE_THRESHOLD` | `0.97` | cosine threshold để `SemanticCache` coi là hit |
| `SEMANTIC_CACHE_TTL_SECONDS` | `120` | TTL entry |
| `SEMANTIC_CACHE_MAX_ITEMS` | `256` | LRU bound |

`EmbeddingService` cache (`cache_size=2048`, `cache_ttl_s=600`, `cache_max_batch=8`) là constructor
args, **không** phải env settings.

## Recent additions (2026-06)

- **`semantic_cache.py` (`SemanticCache`)** — tier-2 retrieval cache mới, default-OFF qua
  `SEMANTIC_CACHE_ENABLED`. KHÔNG được wire trong `services/` — nó được instantiate bởi
  `retrieval/elasticsearch_retriever.py` (`search_cached`). `get(embedding)` quét entry gần nhất
  còn hạn có `cosine >= SEMANTIC_CACHE_THRESHOLD`; hit gắn `semantic_cache_hit=True` lên payload.
  Per-worker, không distributed. Cache bypass khi query có `filters` hoặc `document_title`
  (tránh cross-scope leak). Signal `semantic_cache_hit` được `agent/graph_service.py` đọc và bubble
  lên chat response.
- **`LLMGateway.text_response` / `json_response_multimodal`** là đường dùng bởi recent RAG work:
  `ingestion/contextualizer.py` + `ingestion/raptor.py` gọi `text_response`; fast-path `fast_answer`
  node trong `graph_service.py` gọi `text_response`; CRAG/multi-hop answer trong `agent/service.py`
  gọi `json_response`/`json_response_multimodal`. Các flag gate (Contextual Retrieval, RAPTOR, CRAG,
  fast-path) nằm ở các module đó, không ở `services/`.

## Gotchas

- **`semantic_cache.py` không tự đọc settings và không tự bật.** Nó là data-structure thuần;
  gate `SEMANTIC_CACHE_ENABLED` + 3 settings được áp ở `elasticsearch_retriever.__init__`. Đừng tìm
  flag check bên trong file này.
- **Hai facade cho retrieval.** `RetrievalService` (mới, S4, container-driven, filters-only) vs
  `KnowledgeService` (legacy, ôm cả reasoning: HyDE/decompose/intent-mode, gọi `AgentTools` trực
  tiếp). Code mới dùng `RetrievalService` + `reasoning_knowledge.py` helpers; `KnowledgeService`
  còn sống vì MCP (`mcp/app.py`, `mcp/server.py`). Logic trong hai file trùng nhau (vd `_select_retrieval_mode` vs `select_retrieval_mode`) — sửa thì sửa cả hai.
- **`VisionService.describe()`/`parse_file()` raise nếu disabled.** `enabled` chỉ True khi cả
  `VISION_PROVIDER` lẫn `VISION_MODEL` set. Reasoning code phải check `c.vision.enabled` trước.
- **Cost ledger không nằm trong module này.** `LLMGateway` chỉ gọi `observability.cost.record_llm_call`
  / `cost_summary`. Ledger backing là Valkey stream (`maxlen` xấp xỉ), fallback process-local
  `deque` khi Valkey unreachable — KHÔNG phải ring buffer in-memory cố định, và KHÔNG nhất thiết mất
  khi restart. USD = provider `usage.*` khi có (`usage_source="provider"`), else char-density estimate
  (`usage_source="estimate"`). API: `GET /metrics/cost`, `GET /metrics/cost/recent`,
  `POST /metrics/cost/reset` (`adapter/routers/config.py`).
- **`domain_router` sống trong container nhưng là Reasoning service** — đặt ở đây chỉ để singleton-share.
  `RetrievalService` cố ý `router=None`: routing xảy ra ngoài rồi forward qua `system_override`.
- **`container.override()` raises `KeyError`** nếu key không khớp một property name (`llm`,
  `embedding`, `vision`, `storage`, `retrieval`, `domain_router`).
- **`__init__.py` lazy re-export** (PEP 562): import một leaf submodule không kéo theo
  `knowledge_service`. Đừng đổi sang eager import — sẽ tái lập circular import qua `agent.tools`.
