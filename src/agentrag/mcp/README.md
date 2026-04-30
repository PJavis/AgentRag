# Module: `mcp` — Model Context Protocol Server

**Vị trí:** `src/agentrag/mcp/`

Expose AgentRag tools qua **FastMCP** (Model Context Protocol). Được mount vào FastAPI app tại `/mcp` với **streamable HTTP transport** — tương thích với Claude Desktop, Claude Code, và bất kỳ MCP-compatible client nào.

---

## Files

| File | Class | Mô tả |
|---|---|---|
| `app.py` | `mcp` (FastMCP instance) | Tool definitions và lazy-initialized services |
| `server.py` | `MCPServer` | Wrapper class cho backward compatibility |

---

## Mounting

```python
# main.py
from src.agentrag.mcp.app import mcp
app.mount("/mcp", mcp.streamable_http_app())
```

Endpoint: `http://localhost:8000/mcp`

---

## Tools

### `search`

```
Search the AgentRag knowledge base using hybrid retrieval (BM25 + dense + StructMem knowledge graph).
```

**Parameters:**
| Tham số | Type | Mô tả |
|---|---|---|
| `query` | `str` | Câu truy vấn |
| `document_title` | `str \| None` | Lọc theo tài liệu (optional) |
| `top_k` | `int` | Số kết quả (default: 5) |

**Returns:** JSON string chứa list kết quả với `content`, `document_title`, `section_path`, `score`, `source`.

---

### `structured_query`

```
Answer structured questions using SQL reasoning over extracted tabular data.
Best for: comparison, aggregation, ranking, multi-filter queries.
```

**Parameters:**
| Tham số | Type | Mô tả |
|---|---|---|
| `question` | `str` | Câu hỏi |
| `document_title` | `str \| None` | Lọc theo tài liệu (optional) |
| `query_type` | `str` | `"comparison"` / `"aggregation"` / `"ranking"` |

**Returns:** JSON string với `answer`, `sql_query`, `reasoning_path`.

---

## Sử dụng với Claude Code

Thêm vào `~/.claude.json` hoặc `.mcp.json`:

```json
{
  "mcpServers": {
    "agentrag": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

---

## Sử dụng với MCP Inspector

```bash
npx @modelcontextprotocol/inspector http://localhost:8000/mcp
```

---

## Lazy initialization

Services (`KnowledgeService`, `SecurityService`, `StructuredReasoningPipeline`) được khởi tạo lần đầu khi tool được gọi, không phải khi server start — tránh tốn tài nguyên nếu MCP không được dùng.

```python
_svc: dict[str, Any] = {}

def _services():
    if "knowledge" not in _svc:
        _svc["knowledge"] = KnowledgeService()
        _svc["security"] = SecurityService()
        _svc["pipeline"] = StructuredReasoningPipeline(...)
    return _svc["knowledge"], _svc["security"], _svc["pipeline"]
```

---

## Tương tác

| Module | Vai trò |
|---|---|
| `services.KnowledgeService` | Hybrid retrieval cho tool `search` |
| `services.SecurityService` | Filter results (không có document scope → pass-through) |
| `structured.StructuredReasoningPipeline` | SQL reasoning cho tool `structured_query` |
| `main.py` | Mount MCP app vào FastAPI |
