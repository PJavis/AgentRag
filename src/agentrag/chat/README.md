# chat — Conversation persistence + semantic chat memory (Chat StructMem)

## Mục đích / Purpose
Module này sở hữu **state hội thoại** của AgentRag: lưu/đọc conversations và messages
(`history.py`), và một lớp **semantic conversation memory** (`structmem.py`) trích xuất
"factual" + "relational" entries từ mỗi lượt hội thoại, embed + index vào Elasticsearch,
rồi retrieve lại theo ngữ nghĩa để inject vào prompt của agent. Lý do tồn tại: thay thế
sliding-window flat history bằng semantic recall khi hội thoại dài, đồng thời tách phần
lưu trữ bền vững (PG + Redis) khỏi reasoning của agent.

## Plane
**Mixed — chủ yếu Execution / Infrastructure, với một mảng Reasoning bên trong `structmem.py`.**

- `history.py` (`ConversationStore`) là **Execution / Infrastructure** thuần: chỉ làm IO
  (PostgreSQL + Redis cache), không có prompt, không branching theo nội dung.
- `structmem.py` (`ChatMemoryService`) là một **boundary leak có chủ đích**: nó vừa giữ
  prompts + LLM extraction/synthesis (Reasoning) vừa tự gọi IO (ES, embedder, LLM client)
  trực tiếp — KHÔNG đi qua `ServiceContainer`. Giống `KnowledgeService`, đây là một
  reasoning-service tự-chứa, không phải worker thuần.
- `memory_jobs.py` chỉ là một dataclass payload (Infrastructure).

## Key files
| File | Responsibility |
|---|---|
| `history.py` | `ConversationStore` — CRUD conversations + messages. Two-tier: Redis cache (TTL) trước, PostgreSQL là source of truth. |
| `structmem.py` | `ChatMemoryService` — dual-perspective extraction (factual + relational), cross-turn consolidation/synthesis, semantic retrieve. Tự quản ES index + embedder + LLM client. |
| `memory_jobs.py` | `ChatMemoryJob` — frozen dataclass mô tả payload của ARQ `chat_memory` job. Hiện **chưa có caller** (worker nhận kwargs trực tiếp); giữ làm typed contract. |

## Public interface

### `ConversationStore` (history.py)
Import trực tiếp (`from src.agentrag.chat.history import ConversationStore`), KHÔNG qua
container. Callers: `adapter/routers/chat.py`, `adapter/admin.py`, `cli/chat.py`,
`cli/conversations.py`. Tất cả method là `async`.

| Method | Notes |
|---|---|
| `create_conversation(title=None, extra_metadata=None, user_id=None)` | `user_id` được parse thành UUID; `"anonymous"`/invalid → `None`. |
| `get_conversation(conversation_id)` | `None` nếu id không phải UUID hợp lệ hoặc không tồn tại. |
| `get_or_create_conversation(conversation_id, title=None)` | Tạo mới nếu id rỗng hoặc không tìm thấy. |
| `append_message(conversation_id, role, content, citations=None, tool_trace=None, timings_ms=None, extra_metadata=None)` | Ghi PG → **invalidate** Redis cache. `citations`/`tool_trace`/`timings_ms` lưu JSON cột. |
| `list_messages(conversation_id, limit=20)` | Redis-first, fallback PG → write-back cache. Trả `[-limit:]`. |
| `delete_message(conversation_id, message_id)` | Dùng bởi **regenerate flow** để bỏ assistant turn cũ trước khi chạy lại agent. |
| `delete_conversation(conversation_id)` | Xoá messages + conversation + cache. |
| `list_conversations(limit=20)` | Mới nhất trước (`created_at desc`). |

### `ChatMemoryService` (structmem.py)
Import trực tiếp, instantiate per-call. Callers: `agent/service.py::_retrieve_memory`,
`agent/graph_service.py::memory` node (cùng qua `_retrieve_memory`), và
`worker/functions.py::chat_memory`.

| Method | Notes |
|---|---|
| `process_turn(conversation_id, user_message, assistant_message, turn_id, turn_timestamp)` | 2 LLM calls song song (factual + relational) → embed → index. No-op nếu không trích được entry. |
| `retrieve(conversation_id, query, top_k=None)` → `list[dict]` | KNN trên `kind=entry` (k = `CHAT_MEMORY_TOP_K`) + `kind=synthesis` (k=3), merge, sort theo score, cắt top-k. Mọi lỗi → `[]`. |
| `count_unconsolidated(conversation_id)` → `int` | Đếm `kind=entry, consolidated=false`. Lỗi/missing index → `0`. |
| `consolidate(conversation_id)` | Cross-turn synthesis → index `kind=synthesis`, rồi mark buffer `consolidated=true`. |

## Data flow

**Đọc (retrieve, mỗi request):**
```
POST /chat → AgentService.chat_stream / GraphAgentService(memory node)
  └─ _retrieve_memory(conversation_id, question)   [chỉ khi CHAT_STRUCTMEM_ENABLED]
       └─ ChatMemoryService.retrieve() → list[dict]
            → inject vào prompt _decide()/_answer() dưới key "conversation_memory"
```
Song song, agent đọc flat history qua `ConversationStore.list_messages` (Redis→PG) cho
sliding window (`CHAT_HISTORY_WINDOW`).

**Ghi (sau mỗi turn):**
```
assistant message hoàn tất
  ├─ ConversationStore.append_message()  → PG + invalidate Redis
  └─ ARQ "chat_memory" job (worker/functions.py::chat_memory, kwargs)
        └─ ChatMemoryService.process_turn()  → extract/embed/index entries
        └─ count_unconsolidated() ≥ CHAT_MEMORY_CONSOLIDATION_THRESHOLD?
              └─ consolidate()  → LLM synthesis → index kind=synthesis
```

**Upstream callers:** API routers (`adapter/routers/chat.py`, `adapter/admin.py`), CLI
(`cli/chat.py`, `cli/conversations.py`), agent loop (`agent/service.py`,
`agent/graph_service.py`), ARQ worker (`worker/functions.py`, registered in
`worker/settings.py`).
**Downstream deps:** `database` (`AsyncSessionLocal`, models `Conversation` / `ChatMessage`),
Redis, Elasticsearch, `ingestion/embedders/factory.build_embedding_provider`, OpenAI-compatible
LLM client (`EXTRACTION_*` config).

## Storage layout

**PostgreSQL** (durable source of truth): `Conversation`, `ChatMessage`
(`database/models.py`). Messages giữ `citations`, `tool_trace`, `timings_ms`, `extra_metadata`
JSON.

**Redis** (cache, fail-open): key
`agentrag:conversation:{conversation_id}:messages:v1`, TTL = `CHAT_REDIS_TTL_SECONDS`.
Mọi `RedisError` → set `self._redis = None` và fallback PG; request không crash.

**Elasticsearch** — một index duy nhất `CHAT_MEMORY_INDEX` (default `agentrag_memory_chat`),
phân biệt bằng field **`kind` ∈ {entry, synthesis}** (KHÔNG còn 2 index riêng):

| kind | Fields đặc trưng |
|---|---|
| `entry` | `turn_id, turn_timestamp, entry_type (factual\|relational), content, subject, source_entity, target_entity, relation_type, confidence, consolidated (bool), embedding` |
| `synthesis` | `content, hypothesis_type, supporting_entry_ids[], reasoning, confidence, created_at, embedding` |

Index được tạo lazy qua `_ensure_indices(dims)` ở lần index đầu (dims = số chiều embedding thực tế).

## Config
Tất cả từ `src/agentrag/config.py` (`settings.*`):

| Key | Default | Mô tả |
|---|---|---|
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Bật/tắt Redis cache. `None` → bỏ cache, dùng PG thuần. |
| `CHAT_REDIS_TTL_SECONDS` | `300` | TTL cache messages. |
| `CHAT_HISTORY_WINDOW` | `10` | Sliding-window flat history (đọc bởi agent, không bởi module này trực tiếp). |
| `CHAT_STRUCTMEM_ENABLED` | `True` | Bật semantic chat memory (extraction + retrieve). |
| `CHAT_MEMORY_INDEX` | `agentrag_memory_chat` | ES index (single, dùng `kind`). |
| `CHAT_MEMORY_CONSOLIDATION_THRESHOLD` | `10` | Số unconsolidated entries trước khi `consolidate()` chạy. |
| `CHAT_MEMORY_TOP_K` | `8` | Số entries retrieve mỗi lượt (mặc định của `retrieve`). |
| `STRUCTMEM_CONSOLIDATION_HISTORY_TOP_K` | `15` | k cho historical-seed search khi consolidate (dùng chung với doc StructMem). |
| `EXTRACTION_PROVIDER` / `EXTRACTION_MODEL` / `EXTRACTION_BASE_URL` | `ollama` / `llama3.1:8b...` / `None` | LLM backend cho extraction + synthesis. `_build_llm_client()` chọn `openai\|ollama\|gemini\|hf_inference`. |
| `ELASTICSEARCH_URL` | — | ES cluster cho memory index. |
| Embedding (`EMBEDDING_*`) | — | Embedder qua `build_embedding_provider(settings)`. |

## Gotchas
- **Single ES index, không phải hai.** Code cũ/README cũ nói tới
  `agentrag_chat_entries` + `agentrag_chat_synthesis`; thực tế giờ là MỘT index
  `agentrag_memory_chat` với discriminator `kind`. `CHAT_MEMORY_SYNTHESIS_INDEX` không
  còn tồn tại.
- **`CHAT_STRUCTMEM_ENABLED` mặc định `True`** (không phải off). Tắt phải set explicit.
- **`ChatMemoryJob` hiện chưa được dùng** — ARQ `chat_memory` worker nhận kwargs rời, không
  nhận dataclass. Dataclass giữ làm typed contract; nếu thêm enqueue site, cân nhắc dùng nó.
- **`ChatMemoryService` tự gọi IO, KHÔNG qua `ServiceContainer`** — đây là boundary leak có
  chủ đích (tương tự `KnowledgeService`). Test mock bằng patch ES/LLM/embedder trực tiếp,
  không qua `container.override(...)`.
- **`process_turn` không index có thứ tự đảm bảo** — index từng doc bằng vòng lặp `await`
  (không bulk); ES auto-id, không idempotent nếu job chạy lại → có thể tạo entry trùng. Job
  được coi "best effort", lỗi LLM/ES được nuốt (logged ở mức debug/warning).
- **`consolidate` ghi synthesis trước rồi mới mark buffer `consolidated=true`** từng doc một;
  nếu crash giữa chừng, một số entry vẫn `consolidated=false` và sẽ được consolidate lại lượt
  sau (an toàn nhưng có thể sinh synthesis trùng lặp).
- **Redis fail-open một chiều**: khi gặp `RedisError`, `self._redis` bị set `None` cho phần
  đời còn lại của instance đó — các call sau bỏ qua Redis hoàn toàn cho tới khi tạo
  `ConversationStore` mới.
- **`list_messages` đọc cache toàn bộ rồi cắt `[-limit:]`** — cache lưu full list, không phải
  trang; conversation rất dài vẫn nạp hết vào memory.
- **`retrieve` lỗi → trả `[]` im lặng** (log warning). Agent vẫn chạy được nhưng mất memory
  context; không có signal nào nổi lên response để phân biệt "không có memory" vs "ES down".
