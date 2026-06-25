# cli — Interactive Typer + Rich command-line front-end for the agent

## Mục đích / Purpose
CLI tương tác kiểu Claude CLI, xây bằng **Typer + Rich**. Cung cấp một `chat` REPL streaming (spinner + token streaming + citations) và một nhóm lệnh `conversations` để CRUD hội thoại. CLI gọi agent **trong tiến trình** (in-process) qua `get_agent_service()` — KHÔNG gọi HTTP API. Active-conversation được persist vào `~/.agentrag/state.json` để giữ ngữ cảnh giữa các lần chạy.

## Plane
**Infrastructure** (presentation / entrypoint). Đây là lớp trình bày: nó không chứa logic reasoning hay IO worker, mà chỉ điều phối input người dùng → gọi `AgentService.chat_stream(...)` (Reasoning Plane) và `ConversationStore` (Execution Plane), rồi render kết quả bằng Rich.

## Key files
| File | Responsibility |
|---|---|
| `app.py` | Typer entrypoint. Tạo `cli_app`, đăng ký `chat` command và sub-app `conversations`. `main()` là console-script target. |
| `chat.py` | Interactive chat REPL: vòng lặp `_chat_loop`, parse SSE (`_parse_sse`), stream một lượt (`_stream_turn`), render answer/citations, xử lý inline-commands. |
| `conversations.py` | Sub-app Typer quản lý hội thoại: `list`, `new`, `switch`, `delete`, `show`. Mỗi command bọc một coroutine `asyncio.run(...)`. |
| `state.py` | Persistent local state ở `~/.agentrag/state.json`: `load_state`/`save_state`, `get/set/clear_active_conversation`. |
| `__init__.py` | Rỗng (package marker). |

## Public interface
Module được dùng như một **entrypoint**, không phải import qua `ServiceContainer`.

- `app.main()` — gọi `cli_app()`. Đây là target của console script `agentrag` (xem `pyproject.toml [project.scripts] agentrag = "agentrag.cli.app:main"`) và của file root `cli.py` (`from src.agentrag.cli.app import main`).
- `app.cli_app: typer.Typer` — Typer app với 2 nhánh: command `chat` và sub-app `conversations`.
- `chat.chat(new, title, document, conversation_id)` — hàm command cho `agentrag chat` (xem options bên dưới).
- `conversations.app: typer.Typer` — sub-app, gắn vào `cli_app` với tên `conversations`.
- `state.get_active_conversation() -> tuple[str|None, str|None]`, `state.set_active_conversation(conversation_id, title)`, `state.clear_active_conversation()` — dùng nội bộ bởi cả `chat.py` và `conversations.py`.

Cách chạy:
```bash
python cli.py chat                 # qua file root cli.py
agentrag chat                      # nếu đã cài package (console script)
agentrag conversations list
```

## Data flow
1. `chat` command → `_chat_loop(conversation_id, document_title)`.
2. `_chat_loop` tạo `ConversationStore` (từ `src.agentrag.chat.history`), resolve active conversation qua `state`, và lấy agent qua `get_agent_service()` (từ `src.agentrag.agent.factory` → trả về `GraphAgentService`, delegate stream về inner `AgentService`).
3. Mỗi lượt: lưu user message (`store.append_message`), lấy lịch sử (`store.list_messages(limit=settings.CHAT_HISTORY_WINDOW)`), rồi gọi `agent.chat_stream(question, document_title, chat_history)`.
4. `chat_stream` yield các dòng SSE; `_stream_turn` parse bằng `_parse_sse` và render trong một `rich.live.Live` block với `Spinner`.
5. Sau khi xong: render answer (`rich.markdown.Markdown`), citations table, và nếu có thì in `SQL: ...` / `path: ...`; cuối cùng lưu assistant message kèm citations.

Upstream caller: người dùng / file root `cli.py` / console script.
Downstream deps: `agent.factory.get_agent_service`, `agent.graph_service.GraphAgentService.chat_stream` (delegate sang `agent.service.AgentService`), `chat.history.ConversationStore`, `config.settings`.

### SSE events mà `_stream_turn` xử lý
Khớp với các event do `AgentService.chat_stream` phát ra (`event: <type>\ndata: <json>\n\n`):

| Event | Xử lý trong CLI |
|---|---|
| `status` | Cập nhật spinner text `"{step}…"` (vd. `chitchat`, `retrieve`, `decide`, `tool`, `answer`). |
| `token` | Append vào buffer; hiển thị dần trong Live block. |
| `done` | Đọc `citations`; lưu `done_data` (gồm `reasoning_path`, `sql_query` nếu có). |
| `error` | In ra `err_console` (stderr, đỏ). |

`reasoning_path` có thể là `chitchat` / `semantic`. CLI chỉ in badge `path:` khi path khác `semantic`; in `SQL:` khi `done` có `sql_query`.

### Inline commands trong REPL (`chat.py`)
| Command | Hành vi thực tế |
|---|---|
| `/new [title]` | Tạo conversation mới (`store.create_conversation`) và set active. |
| `/switch <id-prefix>` | Prefix-match qua `list_conversations(limit=100)`; báo lỗi nếu không khớp hoặc mơ hồ. |
| `/list` | Liệt kê tối đa 20 conversation, đánh dấu `✦` cái đang active. |
| `/clear` | **Bắt đầu một conversation mới rỗng** (không phải xoá màn hình). |
| `exit` / `quit` / `/exit` / `/quit` | Thoát REPL. |

### `chat` command options
| Option | Mặc định | Ý nghĩa |
|---|---|---|
| `--new` / `-n` | `False` | Tạo conversation mới trước khi vào REPL. |
| `--title` / `-t` | `""` | Tiêu đề cho conversation mới (chỉ dùng với `--new`). |
| `--document` / `-d` | `""` | Scope retrieval theo một `document_title`. |
| `--id` | `""` | Resume một conversation theo ID. |

### `conversations` sub-commands
| Command | Hành vi |
|---|---|
| `list [--limit/-n 20]` | Rich table: marker active, short-id (8 ký tự), title, "created" tương đối (`_fmt_time`). |
| `new [TITLE]` | Tạo + set active; in ID đầy đủ. |
| `switch <id-prefix>` | Prefix-match; `typer.Exit(1)` nếu không khớp / mơ hồ. |
| `delete <id-prefix> [--yes/-y]` | Prefix-match; confirm (trừ khi `-y`); xoá; nếu trùng active thì `clear_active_conversation()`. |
| `show` | In active conversation hiện tại (id + title) từ `state`. Không nhận tham số. |

## Config
| Setting | Dùng ở đâu |
|---|---|
| `settings.CHAT_HISTORY_WINDOW` (mặc định `10`) | `chat.py` — số message lịch sử nạp vào mỗi lượt (`store.list_messages(limit=...)`). |

Không có biến môi trường HTTP nào (vd. `AGENTRAG_API_URL`): CLI chạy agent in-process, không gọi server. Storage/DB do `ConversationStore` và cấu hình của nó quyết định, không phải module này.

## Gotchas
- **In-process, không HTTP.** CLI gọi thẳng `get_agent_service()` và `ConversationStore` trong cùng tiến trình. Mọi cấu hình DB/LLM (env) phải có sẵn ở môi trường chạy CLI; không có web server trung gian.
- **Streaming structured path là giả lập.** `GraphAgentService.chat_stream` chưa stream thật — nó delegate về inner `AgentService`, và path `structured` phát token bằng cách lặp từng ký tự của câu trả lời đã hoàn chỉnh (không phải token-by-token thật từ LLM).
- **`/clear` không xoá màn hình** — nó tạo conversation mới. `--clear`/xoá màn hình không tồn tại.
- **`conversations show` chỉ hiện active conversation**, KHÔNG liệt kê message trong một hội thoại (đừng nhầm với hành vi mô tả ở README cũ).
- **State có thể trỏ tới conversation đã xoá** nếu xoá ở nơi khác; `state.py` chỉ tự reset khi chính lệnh `delete` thấy id trùng active. `load_state` nuốt mọi lỗi JSON và trả `{}`.
- **Mỗi sub-command tự `asyncio.run(...)`**, nên không gọi chúng từ bên trong một event loop đang chạy.
- **Spinner dùng `Live(transient=True)`**: trong khi stream, answer hiển thị trong spinner text; bản render Markdown sạch chỉ in lại sau khi `done`.
