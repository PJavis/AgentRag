# generation — học liệu phái sinh từ document (mindmap + structured summary)

## Mục đích / Purpose
Sinh learning artifacts từ một document đã ingest, tách biệt khỏi vòng hỏi-đáp của agent:
**mindmap** (Mermaid + concept hierarchy) và **structured summary** (tóm tắt theo template y khoa
hoặc map-reduce toàn văn). Cả hai service đều tự retrieve chunks từ Elasticsearch rồi gọi LLM
qua `LLMGateway` — không đi qua agent loop, không có retrieval router, không có decision branching
trên prompt. Đây là các "one-shot generators" cho UI (notebook summary, mindmap view, chat tóm tắt).

## Plane
**Hybrid (chủ yếu Execution).** Hai service là IO worker stateless: ES read + LLM call + (mindmap)
cache. Chúng KHÔNG nằm trong `ServiceContainer` và KHÔNG được type qua `services/protocols.py` —
chúng tự instantiate `ElasticsearchStore()` + `LLMGateway()` trực tiếp (giống `AgentTools` instantiate
retriever trực tiếp, một boundary leak đã biết). Phần "reasoning" duy nhất nằm trong các system prompt
hard-coded (template y khoa, hướng dẫn văn phong) — không có classify/route động ở runtime.

## Key files
| File | Responsibility |
|---|---|
| `summary_service.py` | `SummaryService` — 3 đường sinh tóm tắt: per-section template (`generate`), quick-review 1-call (`_quick_review`), và map-reduce toàn văn (`generate_full` / `iter_sections`). |
| `mindmap_service.py` | `MindmapService` — sinh Mermaid mindmap + concept list, cache Valkey/Redis TTL 24h, có `invalidate()`. |
| `__init__.py` | Rỗng (không re-export — caller import trực tiếp module path). |

## Public interface

Truy cập **bằng direct import** (không qua container):

```python
from src.agentrag.generation.summary_service import SummaryService
from src.agentrag.generation.mindmap_service import MindmapService
```

### `SummaryService`
```python
# Template y khoa 9 mục (overview + sections song song). REST path.
async def generate(document_title: str,
                   style: Literal["study_note","clinical","quick_review"] = "study_note"
                   ) -> dict   # {title, style, overview, sections[]}

# Map-reduce TOÀN VĂN: gom mọi text segment thành các page-span, tóm tắt từng span.
# Thấy 100% nội dung (không phải top-K). Dùng cho chat regenerate (non-stream).
async def generate_full(document_title, on_progress=None, char_budget=14000,
                        max_batches=24, concurrency=8) -> dict  # {title, style:"full", sections[], batch_count}

# Như generate_full nhưng là async generator, yield ("meta",{total}) rồi
# ("section", sec) theo PAGE ORDER khi mỗi batch xong → chat stream live.
async def iter_sections(document_title, char_budget=14000, max_batches=24, concurrency=8)
```

`study_note` và `clinical` **hành xử y hệt nhau** — cả hai đều chạy template `_MEDICAL_TEMPLATE_VI`
(9 heading tiếng Việt). `style` chỉ được trả lại trong response để consumer biết tên; không có branch
nào khác biệt theo style trừ `quick_review`.

### `MindmapService`
```python
async def generate(document_title: str, focus_topic: str | None = None,
                   max_depth: int = 3) -> dict   # {mermaid, concepts[], cached}
async def invalidate(document_title: str) -> None  # xóa mọi cache key của document
```

### REST endpoints (`main.py`, không phải router trong `adapter/routers/`)
| Route | Service call |
|---|---|
| `POST /generate/mindmap` | `MindmapService().generate(document_title, focus_topic, max_depth)` |
| `POST /generate/summary` | `SummaryService().generate(document_title, style)` — `style` validate ∈ {study_note, clinical, quick_review} |

`POST /metrics` cũng ở `main.py` (LLM cost) nhưng không thuộc module này.

## Data flow

**Mindmap** (`main.py /generate/mindmap` → `MindmapService.generate`):
```
generate(title, focus_topic, max_depth)
  → cache_get (Valkey, key=sha256("title|focus|depth"))  ── hit → return {...,"cached":true}
  → ElasticsearchStore.sparse_search(query=focus_topic|title, top_k=30, document_title=title)
  → _build_context: 30 chunks, content cắt 600 ký tự, kèm section_path
  → LLMGateway.json_response(task="mindmap")  → {mermaid, concepts}
  → cache_set (key + thêm vào doc-index set để invalidate)  → return
```
Document rỗng (không chunk) → trả mermaid stub `mindmap\n  root((Title))\n    Không tìm thấy nội dung`.

**Summary — per-section template** (`/generate/summary`, hoặc gọi `generate` trực tiếp):
```
generate(title, style)  ── style=="quick_review" → _quick_review (1 LLM call, 30 chunks)
  └─ asyncio.gather:
       _generate_overview: sparse_search(top_k=10) → LLM(task="summary") → overview prose
       _generate_sections: _fetch_image_chunks once, rồi với MỖI heading trong 9-mục template:
           sparse_search(query=heading, top_k=15) → lọc bỏ image chunk
           → LLM(task="summary") → {summary, key_points}
           → gắn images theo page-overlap (image chunk có page nằm trong page của section)
       (section trống bị bỏ → document không nói về mục đó)
```

**Summary — map-reduce toàn văn** (`generate_full` / `iter_sections`, đường ĐANG DÙNG cho chat):
```
fetch_all_segments(title)  ── ES search_after, vượt cửa sổ 10k, sort page→position, loại image
  → _batch_chunks: gom theo page order thành span ~char_budget; nếu doc khổng lồ thì
    nới budget để batch ≤ max_batches (chặn số LLM call/chi phí)
  → mỗi span: _summarize_batch → LLM(task="answer")  ← CỐ Ý dùng model FAST (flash),
    không phải task="summary"/pro: nhiều call song song, chỉ condense text given
  → yield/return sections theo PAGE ORDER, batch nào lỗi thì skip
```

**Upstream callers:** `main.py` (2 REST endpoint) và `adapter/routers/chat.py`. Chat router gọi
map-reduce path khi `_is_summary_request(message)` và resolve được đúng 1 document:
`execute_chat_stream` dùng `iter_sections` (stream từng section qua SSE), `regenerate` dùng
`generate_full`. Chat router render kết quả qua các helper `_summary_section_to_markdown` /
`_summary_full_to_markdown` của chính nó, gắn `reasoning_path="summary_mapreduce"`.

**Downstream deps:** `ingestion.stores.elasticsearch_store.ElasticsearchStore`
(`sparse_search`, `fetch_all_segments`); `services.llm_gateway.LLMGateway` (`json_response`);
`MindmapService` thêm `redis.asyncio` (Valkey) + `config.settings.REDIS_URL`.

## Config
| Setting (`config.py`) | Dùng ở đâu |
|---|---|
| `REDIS_URL` | `MindmapService._client()` — nếu None thì cache no-op (vẫn chạy, không cache). |
| `ELASTICSEARCH_URL` | Gián tiếp qua `ElasticsearchStore()`. |
| `RETRIEVAL_TOP_K` | Default của `sparse_search` khi `top_k` không truyền (các call ở đây đều truyền top_k tường minh). |
| `LLM_ROUTING_ENABLED` + `LLM_TASK_MODEL_MAP` | Quyết định model cho task `"mindmap"` / `"summary"` / `"answer"` qua `LLMGateway._resolve_client(task)`. Map riêng cho từng task nếu muốn (vd batch dùng flash). |

## Gotchas
- **Mindmap cache giờ là Valkey/Redis, không còn in-process dict** (README cũ sai). Survive restart,
  share giữa worker. Cache key = `sha256("title|focus|max_depth")`; mỗi document có thêm một
  Redis SET index (`_doc_index_key`) gom mọi cache key của nó để `invalidate()` xóa sạch.
  **Phải gọi `MindmapService().invalidate(title)` sau khi re-ingest** nếu không UI thấy mindmap cũ tới 24h.
- **Redis lỗi → tự disable cache, không crash**: `RedisError` set `_valkey_disabled=True` (class-level)
  và mọi call sau đó bỏ qua cache cho tới khi process restart.
- **`task="answer"` trong `_summarize_batch` là cố ý, không phải bug.** Map-reduce phát hàng chục call
  song song nên dùng model nhanh; nếu route batch sang pro model sẽ chậm/đắt. Overview + per-section
  vẫn dùng `task="summary"`.
- **Per-section `generate()` là top-K, KHÔNG phải full coverage**: mỗi heading chỉ lấy top_k=15 chunk.
  Đường "thấy 100% trang" là `generate_full` / `iter_sections` (qua `fetch_all_segments`). Chat đã
  chuyển sang map-reduce; per-section template chủ yếu còn phục vụ REST `/generate/summary`.
- **`important_terms` chỉ còn để back-compat**: prompt section hiện không yêu cầu term/definition list,
  thay vào đó bold inline; `_summarize_section` vẫn `result.get("important_terms", [])` nhưng thường rỗng.
- **`iter_sections` yield theo page order, không theo thứ tự hoàn thành**: tất cả batch chạy song song
  (`create_task`) nhưng `await` lần lượt theo index → section đầu phải xong trước khi stream section sau.
- **Không có Protocol / container override**: test phải patch `ElasticsearchStore` / `LLMGateway`
  ở module path của service, không inject qua `container.override(...)`.
- **LLM trả JSON, không markdown fence**: mọi system prompt kết thúc bằng "Return ONLY JSON, no markdown
  fences"; parse JSON là do `LLMGateway.json_response`. Field thiếu được `.get(..., default)` an toàn.
