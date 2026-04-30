# Module: `chat` — Conversation & Chat StructMem

**Vị trí:** `src/agentrag/chat/`

Quản lý hội thoại, lịch sử tin nhắn, và bộ nhớ hội thoại semantic (Chat StructMem).

---

## Files

| File | Class | Mô tả |
|---|---|---|
| `history.py` | `ConversationStore` | CRUD conversations và messages (Redis cache + PostgreSQL) |
| `structmem.py` | `ChatMemoryService` | Dual-perspective memory extraction + semantic retrieval |
| `memory_jobs.py` | `ChatMemoryJob` | Dataclass cho ARQ job payload |

---

## `ConversationStore` — history.py

Two-tier storage: Redis cache (fast, TTL) + PostgreSQL (persistent, source of truth).

| Method | Mô tả |
|---|---|
| `create_conversation(title, extra_metadata)` | Tạo conversation mới |
| `get_conversation(conversation_id)` | Lấy conversation theo ID |
| `get_or_create_conversation(conversation_id, title)` | Lấy hoặc tạo mới |
| `append_message(conversation_id, role, content, ...)` | Thêm message → invalidate Redis cache |
| `list_messages(conversation_id, limit)` | Ưu tiên Redis, fallback PostgreSQL |
| `list_conversations(limit)` | Liệt kê conversations mới nhất |
| `delete_conversation(conversation_id)` | Xóa PostgreSQL + Redis cache |

### Cơ chế cache

```
list_messages():
  Redis HIT  → trả về cached list
  Redis MISS → query PostgreSQL → ghi Redis (TTL = CHAT_REDIS_TTL_SECONDS)

append_message():
  ghi PostgreSQL → xóa Redis cache (invalidate)
```

Cache key: `agentrag:conversation:{conversation_id}:messages:v1`

Redis lỗi → tự fallback PostgreSQL, không crash request.

---

## `ChatMemoryService` — structmem.py

Bộ nhớ hội thoại semantic, áp dụng StructMem cho chat history. Thay thế sliding-window flat history bằng semantic retrieval khi `CHAT_STRUCTMEM_ENABLED=true`.

### Dual-perspective extraction (per turn)

Mỗi `(user_message, assistant_message)` → 2 LLM calls song song:

**Factual entries** — sự kiện, thông tin được xác nhận trong lượt hội thoại:
```json
{"content": "User xác nhận deadline là ngày 15/3",
 "subject": "deadline", "confidence": "high"}
```

**Relational entries** — cách các chủ đề kết nối, intent người dùng:
```json
{"content": "Câu hỏi về deadline tiếp nối cuộc thảo luận về project scope",
 "source_entity": "deadline", "target_entity": "project scope",
 "relation_type": "follow_up", "confidence": "high"}
```

### Cross-turn consolidation

Trigger khi `count_unconsolidated >= CHAT_MEMORY_CONSOLIDATION_THRESHOLD`:

```
unconsolidated entries
  ├──▶ embed → cosine search → historical seeds
  ├──▶ LLM synthesis → cross-turn patterns + user goals
  ├──▶ index → agentrag_chat_synthesis
  └──▶ mark entries consolidated=true
```

### API

| Method | Mô tả |
|---|---|
| `process_turn(conversation_id, user_message, assistant_message, turn_id, turn_timestamp)` | Extract + embed + index entries từ một turn |
| `retrieve(conversation_id, query, top_k)` | KNN search trên entries + synthesis, trả về `list[dict]` |
| `count_unconsolidated(conversation_id)` | Đếm entries chưa consolidated |
| `consolidate(conversation_id)` | Chạy cross-turn synthesis |

### Injection vào AgentService

```python
# AgentService._retrieve_memory()
memory_context = await ChatMemoryService().retrieve(conversation_id, question)
# Được inject vào _decide() và _answer() user prompts
# dưới key "conversation_memory"
```

---

## ES Indices

### `agentrag_chat_entries`

```
conversation_id, turn_id, turn_timestamp, entry_type (factual|relational),
content, subject, source_entity, target_entity, relation_type,
confidence, consolidated (bool), embedding (dense_vector)
```

### `agentrag_chat_synthesis`

```
conversation_id, content, hypothesis_type, supporting_entry_ids,
confidence, reasoning, created_at, embedding (dense_vector)
```

---

## Luồng tổng thể

```
POST /chat (với CHAT_STRUCTMEM_ENABLED=true)
  │
  ├── ChatMemoryService.retrieve() → memory_context
  ├── AgentService.chat() với memory_context injected
  ├── Lưu assistant message vào ConversationStore
  └── ARQ: enqueue ChatMemoryJob
              │
              └── [worker] ChatMemoryService.process_turn()
                          → count_unconsolidated()
                          → [if ≥ threshold] consolidate()
```

---

## Config liên quan

| Key | Default | Mô tả |
|---|---|---|
| `CHAT_HISTORY_WINDOW` | `10` | Số messages tối đa trong agent context (sliding-window) |
| `CHAT_REDIS_TTL_SECONDS` | `300` | TTL cache Redis |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis URL |
| `CHAT_STRUCTMEM_ENABLED` | `false` | Bật Chat StructMem |
| `CHAT_MEMORY_INDEX` | `agentrag_chat_entries` | ES index cho entries |
| `CHAT_MEMORY_SYNTHESIS_INDEX` | `agentrag_chat_synthesis` | ES index cho synthesis |
| `CHAT_MEMORY_CONSOLIDATION_THRESHOLD` | `10` | Số turns trước khi consolidate |
| `CHAT_MEMORY_TOP_K` | `8` | Số entries retrieve mỗi lượt |
