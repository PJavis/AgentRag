# Module: `services` — Service Layer

**Vị trí:** `src/agentrag/services/`

Lớp trung gian kết nối agent/structured pipeline với retrieval và LLM. Bao gồm unified LLM gateway với task routing, knowledge retrieval facade, và access control.

---

## Files

| File | Class | Mô tả |
|---|---|---|
| `llm_gateway.py` | `LLMGateway` | Unified LLM client — task routing, cost tracking |
| `knowledge_service.py` | `KnowledgeService` | Retrieval facade — bootstrap, tool dispatch, normalization |
| `security_service.py` | `SecurityService` | Query-time access control và result filtering |
| `context_assembly_service.py` | `ContextAssemblyService` | Wrapper cho context dedup + rank + trim |

---

## `LLMGateway`

Điểm duy nhất để gọi LLM trong toàn hệ thống. Hỗ trợ task-based routing (gọi model khác nhau tùy task) và cost tracking.

```python
answer, latency_ms = await gateway.json_response(
    system_prompt=..., user_prompt=..., task="answer"
)
```

| Method | Mô tả |
|---|---|
| `json_response(system, user, task)` | Gọi LLM, parse JSON, đo latency |
| `_resolve_client(task)` | Trả về `AgentLLM` instance đúng model cho task |
| `cost_summary()` | Tổng token + chi phí ước tính (khi `LLM_COST_TRACKING_ENABLED=true`) |

**Task routing** (`LLM_TASK_MODEL_MAP`): map JSON `{"classify": "model-a", "answer": "model-b"}` — task không có trong map dùng default model.

---

## `KnowledgeService`

Facade giữa agent và retrieval. Quản lý tool registry, normalize input, và dedup calls.

| Method | Mô tả |
|---|---|
| `bootstrap_search(query, document_title, intent)` | Warm retrieval đầu tiên — expand query nếu có intent |
| `execute_tool(tool_name, tool_input, question, document_title)` | Dispatch đến retriever với đúng params |
| `normalize_tool_call(tool_name, tool_input, question, document_title)` | Chuẩn hóa tool name + input trước khi execute |
| `fingerprint_call(tool_name, tool_input)` | SHA256 hash để dedup tool calls |
| `describe_tools()` | Mô tả tools dạng text cho LLM _decide prompt |

---

## `SecurityService`

Access control ở query time. Hoạt động theo document_title scope.

| Method | Mô tả |
|---|---|
| `validate_chat_request(question, document_title)` | Kiểm tra request hợp lệ trước khi xử lý |
| `filter_tool_results(tool_output, document_title)` | Xóa kết quả không thuộc document_title (nếu có scope) |

---

## `ContextAssemblyService`

Assemble + dedup + rank + trim kết quả từ nhiều tool calls thành packed context.

| Method | Mô tả |
|---|---|
| `assemble(question, tool_outputs)` | Merge, dedup theo content_hash, rank theo score + source boost, trim theo AGENT_MAX_CONTEXT_CHUNKS |

Source boost: `structmem +0.08`, `synthesis +0.07`, `hybrid +0.06`, `sparse +0.03`

---

## Tương tác

| Module | Vai trò |
|---|---|
| `agent.AgentService` | Gọi KnowledgeService, SecurityService, ContextAssemblyService, LLMGateway |
| `structured.*` | Gọi LLMGateway cho classify/extract/synthesize |
| `retrieval.ElasticsearchRetriever` | KnowledgeService gọi để thực hiện search |
| `main.py` | Expose `LLMGateway.cost_summary()` qua `/metrics` |

---

## Config liên quan

| Key | Default | Mô tả |
|---|---|---|
| `LLM_ROUTING_ENABLED` | `false` | Bật task-based model routing |
| `LLM_TASK_MODEL_MAP` | `"{}"` | JSON map task → model name |
| `LLM_COST_TRACKING_ENABLED` | `false` | Bật cost tracking |
| `AGENT_MAX_CONTEXT_CHUNKS` | `8` | Giới hạn chunks trong packed context |
