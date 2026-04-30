# Module: `cli` — Command-Line Interface

**Vị trí:** `src/agentrag/cli/`

CLI tương tác theo phong cách Claude CLI, xây dựng bằng **Typer + Rich**. Hỗ trợ chat streaming với spinner, quản lý conversations, và persistent active-conversation state.

---

## Files

| File | Mô tả |
|---|---|
| `app.py` | CLI entry point — đăng ký commands |
| `chat.py` | Interactive chat loop với SSE streaming và Rich display |
| `conversations.py` | Conversation management: list, new, switch, delete, show |
| `state.py` | Persistent state (`~/.agentrag/state.json`) |

---

## Cách chạy

```bash
# Từ root project
python cli.py chat
python cli.py conversations list

# Hoặc nếu cài package
agentrag chat
agentrag conversations list
```

---

## `chat` command

```bash
python cli.py chat [OPTIONS]

Options:
  --new / --no-new          Tạo conversation mới thay vì dùng conversation hiện tại
  --title TEXT              Tiêu đề cho conversation mới
  --document TEXT           Lọc theo document_title (tùy chọn)
  --conversation-id TEXT    ID conversation cụ thể để tiếp tục
```

**Ví dụ:**
```bash
# Chat với conversation đang active
python cli.py chat

# Bắt đầu conversation mới về một document
python cli.py chat --new --title "Phân tích Q4" --document "annual_report"

# Tiếp tục conversation cụ thể
python cli.py chat --conversation-id "abc123"
```

### Inline commands trong chat

| Command | Mô tả |
|---|---|
| `/new [title]` | Tạo conversation mới |
| `/switch <id>` | Chuyển sang conversation khác (prefix matching) |
| `/list` | Liệt kê tất cả conversations |
| `/clear` | Xóa màn hình |
| `exit` / `quit` | Thoát |

### Display

- **Spinner** hiển thị bước xử lý (classify → retrieve → decide → answer)
- **Streaming tokens** hiển thị real-time qua SSE
- **Citations table** ở cuối mỗi câu trả lời (document § section)
- **Reasoning path badge**: `[structured]` hoặc `[semantic]`

---

## `conversations` commands

```bash
# Liệt kê conversations (Rich table)
python cli.py conversations list [--limit 20]

# Tạo conversation mới
python cli.py conversations new [--title "Tên conversation"]

# Chuyển active conversation (chấp nhận prefix ID)
python cli.py conversations switch <id_prefix>

# Xóa conversation (yêu cầu confirm)
python cli.py conversations delete <id_prefix>

# Xem messages trong conversation
python cli.py conversations show <id_prefix> [--limit 20]
```

**Ví dụ `list` output:**
```
  ID        Title              Messages  Created
  ────────  ─────────────────  ────────  ──────────
  abc12345  Phân tích Q4       12        2h ago
  def67890  Project X review   3         1d ago
  ★ ghi111  (active)           8         3h ago
```

---

## `state.py` — Persistent State

Lưu active conversation vào `~/.agentrag/state.json`:

```python
from src.agentrag.cli.state import get_active_conversation, set_active_conversation

conversation_id, title = get_active_conversation()
set_active_conversation("uuid-...", "Conversation Title")
clear_active_conversation()
```

State persist giữa các lần chạy CLI. Khi conversation bị xóa, state tự reset về `None`.

---

## `chat.py` — SSE Streaming

Internal: `_stream_turn()` parse SSE events từ `/chat/stream`:

| Event | Xử lý |
|---|---|
| `status` | Cập nhật spinner text ("Retrieving...", "Thinking...") |
| `token` | Append vào buffer, hiển thị dần trong Live block |
| `done` | Hiển thị citations table, trả về full answer + metadata |
| `error` | Hiển thị error message |

---

## Tương tác

| Module | Vai trò |
|---|---|
| `main.py` | API server mà CLI gọi qua httpx |
| `/chat/stream` | SSE endpoint cho streaming chat |
| `/conversations/*` | CRUD endpoints cho conversation management |
| `~/.agentrag/state.json` | Persistent active conversation state |

---

## Config liên quan

CLI tự động kết nối đến `http://127.0.0.1:8000`. Để override:

```bash
AGENTRAG_API_URL=http://my-server:8000 python cli.py chat
```
