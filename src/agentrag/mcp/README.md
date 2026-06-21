# mcp — expose AgentRag retrieval as Model Context Protocol tools

## Mục đích / Purpose
Phơi bày (expose) khả năng cốt lõi của AgentRag — **hybrid retrieval** — dưới dạng MCP tools để các MCP-compatible client (Claude Desktop, Claude Code, MCP Inspector, ...) gọi vào. Module mount một `FastMCP` app vào FastAPI tại `/mcp` qua **streamable HTTP transport**. Nó là một lớp adapter mỏng: không chứa business logic riêng, chỉ wrap các service hiện có và áp `SecurityService.filter_tool_results` lên mọi tool response.

## Plane
**Infrastructure / transport adapter.** Không thuộc Reasoning hay Execution Plane — nó là một *protocol surface* (giống `api/`) đứng trước cả hai: tool `search` gọi xuống Execution Plane (`KnowledgeService`).

## Key files
| File | Responsibility |
|---|---|
| `app.py` | `FastMCP("AgentRag")` instance + 1 tool function (`search`) decorated với `@mcp.tool()`. Đây là entrypoint thực tế được mount vào FastAPI. Services lazy-init qua singleton dict `_svc` / helper `_services()`. |
| `server.py` | Class `MCPServer` — adapter thủ công (không dùng FastMCP) phơi `list_tools()` + `handle_tool_call(name, input)` cùng hằng `TOOL_DEFINITIONS` (JSON-schema cho tool `search`). Hiện **không được import ở đâu ngoài module**; là biến thể standalone / dự phòng cho client tự gọi qua Python. |
| `__init__.py` | Rỗng (chỉ đánh dấu package). |

## Public interface
- **`app.mcp`** — đối tượng `FastMCP`. Được mount trong `main.py`:
  ```python
  from src.agentrag.mcp.app import mcp
  app.mount("/mcp", mcp.streamable_http_app())   # main.py:17,52
  ```
  Endpoint: `http://localhost:8000/mcp`. Một tool đăng ký:
  - `search(query: str, document_title: str | None = None, top_k: int = 5) -> str`

  Trả về **JSON string** (`json.dumps(..., ensure_ascii=False)`), không phải dict — vì MCP tool output là text.

- **`server.MCPServer`** — gọi trực tiếp từ Python (không qua HTTP):
  ```python
  from src.agentrag.mcp.server import MCPServer
  server = MCPServer()
  await server.handle_tool_call("search", {"query": "...", "document_title": None})
  ```
  `handle_tool_call` trả về **dict** (kèm khoá `"tool"`), khác `app.py` trả JSON string.

Module **không** đi qua `ServiceContainer` DI — nó tự khởi tạo `KnowledgeService`, `SecurityService` (`server.py` còn khởi tạo `LLMGateway`) — đây là khác biệt so với agent/api plane.

## Data flow
**Tool `search`:**
MCP client → `search()` → `_services()` → `KnowledgeService.bootstrap_search(query, document_title, top_k)` (hybrid BM25 + dense + graph, có thể kèm HyDE / decompose / rerank tùy flag của KnowledgeService) → `SecurityService.filter_tool_results(tool_output, document_title)` → ánh xạ mỗi result thành `{content, document_title, section_path, content_hash, score}` (score = `r["score"]` hoặc fallback `r["rrf_score"]`) → JSON string `{query, results: [...]}`.

**Upstream callers:** bất kỳ MCP client nào nối tới `/mcp`. **Downstream deps:** `services.KnowledgeService`, `services.SecurityService`, `services.LLMGateway`.

## Config
Module này **không đọc trực tiếp** bất kỳ `settings.*` flag nào. Hành vi retrieval (top_k mặc định `AGENT_TOOL_TOP_K`, HyDE `QUERY_REWRITE_HYDE`, decompose `QUERY_REWRITE_DECOMPOSE`, ...) được quyết định bên trong các service được wrap, không phải ở đây.

## Gotchas
- **Lazy init (chỉ ở `app.py`):** services khởi tạo lần đầu khi tool được gọi, không phải lúc import/mount — tránh mở kết nối DB/ES nếu MCP không bao giờ được dùng. `MCPServer` (`server.py`) thì **eager**: khởi tạo hết trong `__init__`.
- **`top_k`:** `app.py` mặc định `5`; nhưng `KnowledgeService.bootstrap_search` coi `top_k=None`/falsy → dùng `settings.AGENT_TOOL_TOP_K`. Giá trị `top_k` truyền vào không phải lúc nào cũng là số chunk cuối cùng.
- **Hai code path song song:** `app.py` (FastMCP, trả JSON string) và `server.py` (`MCPServer`, trả dict) có cùng logic nhưng output khác kiểu. Sửa một bên nhớ đồng bộ bên kia — chỉ `app.py` đang thực sự được mount.
- **Security filter luôn áp dụng** lên kết quả `search`; nếu `document_title=None` thì lớp filter chủ yếu là pass-through (chỉ áp policy khi có document scope khớp registry).

## Sử dụng / Usage
Claude Code / Claude Desktop (`.mcp.json` hoặc `~/.claude.json`):
```json
{ "mcpServers": { "agentrag": { "type": "http", "url": "http://localhost:8000/mcp" } } }
```
MCP Inspector:
```bash
npx @modelcontextprotocol/inspector http://localhost:8000/mcp
```
