# Module: `chat` — Conversation & Message History

**Vị trí:** `src/pam/chat/`

Quản lý hội thoại và lịch sử tin nhắn với two-tier storage: Redis cache (fast, TTL) + PostgreSQL (persistent, source of truth).

---

## Files

| File | Class | Mô tả |
|---|---|---|
| `history.py` | `ConversationStore` | CRUD đầy đủ cho conversations và messages |

---

## `ConversationStore` — Methods

| Method | Mô tả |
|---|---|
| `create_conversation(title, extra_metadata)` | Tạo conversation mới, trả về `{conversation_id, title, created_at}` |
| `get_conversation(conversation_id)` | Lấy thông tin conversation theo ID |
| `get_or_create_conversation(conversation_id, title)` | Lấy nếu tồn tại, tạo mới nếu không |
| `append_message(conversation_id, role, content, citations, tool_trace, timings_ms, extra_metadata)` | Thêm message vào DB, xóa Redis cache |
| `list_messages(conversation_id, limit)` | Lấy messages — ưu tiên Redis, fallback PostgreSQL |
| `list_conversations(limit)` | Liệt kê conversations mới nhất |
| `delete_conversation(conversation_id)` | Xóa conversation + tất cả messages, xóa Redis cache |

---

## Cơ chế cache

```
list_messages():
  Redis HIT  → trả về cached list (slice [-limit:])
  Redis MISS → query PostgreSQL → ghi Redis (TTL = CHAT_REDIS_TTL_SECONDS)

append_message():
  ghi PostgreSQL → xóa Redis cache (invalidate)

delete_conversation():
  xóa PostgreSQL → xóa Redis cache
```

Cache key format: `pam:conversation:{conversation_id}:messages:v1`

Redis lỗi (RedisError) → tự tắt Redis, tiếp tục với PostgreSQL — không crash request.

---

## Schema PostgreSQL

```
conversations
  id (UUID PK), title, extra_metadata (JSON), created_at, updated_at

chat_messages
  id (UUID PK), conversation_id (FK), role, content (Text),
  citations (JSON), tool_trace (JSON), timings_ms (JSON),
  extra_metadata (JSON), created_at
```

---

## Tương tác

| Module | Vai trò |
|---|---|
| `database.AsyncSessionLocal` | Session PostgreSQL async |
| `database.models.Conversation` | ORM model |
| `database.models.ChatMessage` | ORM model |
| `redis.asyncio.Redis` | Cache layer |
| `main.py` | Gọi tại `/chat`, `/chat/stream`, `/conversations/*` |

---

## Config liên quan

| Key | Default | Mô tả |
|---|---|---|
| `CHAT_HISTORY_WINDOW` | `10` | Số messages tối đa đưa vào agent context |
| `CHAT_REDIS_TTL_SECONDS` | `300` | TTL cache Redis (giây) |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis connection URL (None = bỏ qua cache) |
