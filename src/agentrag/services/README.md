# Module: `services` — Execution Plane

**Vị trí:** `src/agentrag/services/`

> S4 — đây là **Execution Plane**. Mọi IO/LLM/embedding/vision/storage đi qua
> các facade ở đây. Reasoning Plane (`agent/`, `orchestration/`, `structured/`)
> chỉ fetch service từ `ServiceContainer`, không tự khởi tạo concrete class.
> Xem `ARCHITECTURE.md` (root) cho luật chia plane đầy đủ.

---

## Files

| File | Class / Helper | Plane | Mô tả |
|---|---|---|---|
| `protocols.py` | `EmbeddingProtocol`, `VisionProtocol`, `RetrievalProtocol`, `StorageProtocol`, `RerankerProtocol`, `LLMProtocol` | E | Contract `Protocol` (runtime_checkable) cho Reasoning code type-hint |
| `container.py` | `ServiceContainer`, `get_container()`, `set_container()`, `reset_container()` | E | Singleton DI, lazy-init từng service, `override()` cho test |
| `llm_gateway.py` | `LLMGateway` | E | Unified LLM client — task routing, cost tracking, vision multimodal |
| `embedding_service.py` | `EmbeddingService` | E | Dense embedding facade + TTL cache (S3) |
| `vision_service.py` | `VisionService` | E | Vision LLM facade — wrap `ImageParser` |
| `storage_service.py` | `StorageService` | E | CRUD facade trên Postgres + ES segments |
| `retrieval_service.py` | `RetrievalService` | E | Hybrid retrieval facade — wrap `FederatedRetriever` |
| `context_assembly_service.py` | `ContextAssemblyService` | E | Context dedup + rank + trim + lost-in-middle reorder |
| `security_service.py` | `SecurityService` | E | Query-time access control, output filtering |
| `knowledge_service.py` | `KnowledgeService` | mixed | Legacy facade — kết hợp reasoning + execution. New code dùng `reasoning_knowledge.py` + `RetrievalService` thay thế |
| `reasoning_knowledge.py` | `expand_query`, `mode_to_tool`, `select_retrieval_mode`, `normalize_tool_call` | **R** | Pure helpers (no IO) — Reasoning Plane |

E = Execution, R = Reasoning. `knowledge_service.py` còn lại từ trước S4; dùng cho code cũ, đừng mở rộng.

---

## ServiceContainer (S4)

Singleton entry — lazy. Cấu trúc:

```python
from src.agentrag.services.container import get_container

container = get_container()
hits = await container.retrieval.search(query=q, mode="hybrid")
vec  = (await container.embedding.embed([q]))[0]
route = await container.domain_router.classify(q)
```

| Property | Lazy class |
|---|---|
| `.llm` | `LLMGateway` |
| `.embedding` | `EmbeddingService` |
| `.vision` | `VisionService` |
| `.storage` | `StorageService` |
| `.retrieval` | `RetrievalService` |
| `.domain_router` | `DomainRouter` (Reasoning, instanced here for singleton sharing) |

Test pattern:

```python
from src.agentrag.services.container import ServiceContainer
c = ServiceContainer()
c.override(retrieval=mock_retrieval, embedding=mock_embed)
```

---

## EmbeddingService (S3 cache)

Wrap `build_embedding_provider()`. Per-text vectors cached in
`TTLCache(maxsize=2048, ttl=600s)`, keyed by SHA-256(text). Bypass when
batch size > `cache_max_batch` (default 8) — ingestion paths không bị
giam vào cache.

```python
svc = container.embedding
vecs = await svc.embed(["query a", "query b"])
print(svc.cache_stats)        # {"hits": …, "misses": …, "skips": …, "size": …, "hit_rate": …}
```

Hot paths benefit:

- HyDE rewrite + dense kNN call same text in different modes
- Repeated sub-queries during agent decide loop
- ES `_RESULT_CACHE` miss but query-text identical

---

## VisionService

```python
if container.vision.enabled:
    desc = await container.vision.describe(image_bytes, mime="image/jpeg")
```

Khi `VISION_PROVIDER` chưa set → `enabled=False`, gọi `describe()` raises.

---

## StorageService

Read-mostly facade — write CRUD vẫn đi qua `PostgresStore` / `ElasticsearchStore`
trực tiếp trong ingestion pipeline.

| Method | Mô tả |
|---|---|
| `get_chunks_by_hashes(hashes)` | ES bulk lookup |
| `list_documents(limit=20)` | PG document index |
| `get_document_by_title(title)` | PG document lookup |

---

## RetrievalService

Reasoning Plane gọi đây để search. Router/reranker là quyền Reasoning —
service này filter-only.

```python
hits = await container.retrieval.search(
    query=q,
    mode="hybrid_kg",
    top_k=8,
    filters={"systems": ["tim_mach"]},        # generic
    # OR
    system_override="tim_mach",                # explicit
    specialty_override=["noi"],
)
```

Nội bộ wrap `FederatedRetriever(base=ElasticsearchRetriever, router=None)`.
DomainRouter chạy ngoài (Reasoning code) rồi forward kết quả qua override.

---

## LLMGateway

Điểm duy nhất gọi LLM. Hỗ trợ task-based routing + cost tracking.

| Method | Mô tả |
|---|---|
| `json_response(system, user, task)` | LLM → parse JSON → đo latency, record cost |
| `vision_response(system, text, image_bytes, mime, task)` | Multimodal (text + image) cho `ImageParser` |
| `_resolve_client(task)` | Trả `AgentLLM` đúng model cho task |

Task routing (`LLM_TASK_MODEL_MAP`): `{"classify": "model-a", "answer": "model-b"}`.

---

## Cost tracking (S1)

Ledger ở `src/agentrag/observability/cost.py`:

- Mỗi call record `(id, timestamp, task, model, latency_ms, in_tokens, out_tokens, usd, usage_source)`.
- Ring buffer 5000 calls — clear on restart.
- USD: provider `usage.*` khi có, else char-density heuristic.
- Pricing: Gemini 2.5 / 1.5 + OpenAI 4o/4o-mini. Unknown → Gemini 2.5 Flash.

API:

```
GET  /on/api/metrics/cost              total + per-task + per-model (avg, p50, p95, USD)
GET  /on/api/metrics/cost/recent       newest-first feed, ?limit=&since=
POST /on/api/metrics/cost/reset
```

UI dashboard: `/cost` page (S1, auto-refresh 5s).

---

## SecurityService

| Method | Mô tả |
|---|---|
| `validate_chat_request(question, document_title)` | Kiểm tra request hợp lệ |
| `filter_tool_results(tool_output, document_title)` | Lọc kết quả ngoài scope |

---

## ContextAssemblyService

Merge + dedup theo `content_hash` + rank theo score + source boost + trim
theo token budget + lost-in-middle reorder.

Source boost: `structmem +0.08`, `synthesis +0.07`, `hybrid +0.06`, `sparse +0.03`.

Trim: `AGENT_MAX_CONTEXT_TOKENS` (token-aware) — fallback `AGENT_MAX_CONTEXT_CHUNKS`.

Reorder (Liu 2023): khi `AGENT_LOST_IN_MIDDLE_REORDER=true`, `[r1,r2,r3,r4,r5]` → `[r1,r3,r5,r4,r2]`.

---

## reasoning_knowledge.py (R)

Pure Reasoning helpers — không IO, không LLM:

| Function | Mô tả |
|---|---|
| `expand_query(query, intent)` | Rule-based keyword expansion từ classified intent |
| `select_retrieval_mode(intent)` | Intent → `hybrid_kg` / `hybrid` |
| `mode_to_tool(mode)` | Mode → AgentTools tool name |
| `normalize_tool_call(name, input, question, document_title, valid_tools)` | Fallback `search_hybrid_kg` khi LLM emit unknown tool |

---

## Config liên quan

| Key | Default | Mô tả |
|---|---|---|
| `LLM_ROUTING_ENABLED` | `false` | Task-based model routing |
| `LLM_TASK_MODEL_MAP` | `"{}"` | JSON map task → model. Tasks: `classify`, `decide`, `schema_discovery`, `sql_compile`, `synthesize`, `answer`, `mindmap`, `summary`, `domain_router` |
| `LLM_COST_TRACKING_ENABLED` | `false` | Bật ledger + `/metrics/cost` |
| `LLM_LARGE_CONTEXT_MODEL` / `_THRESHOLD` | `None` / `100000` | Auto-switch khi prompt vượt threshold |
| `VISION_PROVIDER` / `VISION_MODEL` / `VISION_BASE_URL` | `None` | Vision multimodal |
| `VISION_TIMEOUT_SECONDS` | `180` | Llava cold-start tolerance |
| `AGENT_MAX_CONTEXT_TOKENS` | `6000` | Packed-context budget |
| `AGENT_MAX_CONTEXT_CHUNKS` | `8` | Legacy fallback when token budget=0 |
| `AGENT_LOST_IN_MIDDLE_REORDER` | `true` | Best chunks at start + end |
| `TAGGING_ENABLED` | `true` | S5 — SectionTagger trong ingest pipeline |
| `DOMAIN_FILTER_ENABLED` | `true` | S5 — Federated filter active |
| `DOMAIN_ROUTER_CONFIDENCE_THRESHOLD` | `0.7` | S5 — top-1 vs top-K |
| `DOMAIN_ROUTER_TOP_K` | `3` | S5 — broadened federation |
