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
| `vision_response(system, text, image_bytes, mime_type, task)` | Multimodal call (text + image) — dùng cho ImageParser. Provider lấy từ `VISION_PROVIDER` / `VISION_MODEL`, fallback `EXTRACTION_*`. Hỗ trợ `openai`, `gemini`, `ollama` (llava qua OpenAI-compat). |
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
| `assemble(question, tool_outputs)` | Merge, dedup theo content_hash, rank theo score + source boost, trim theo token budget (hoặc chunk-count fallback), reorder lost-in-middle |

Source boost: `structmem +0.08`, `synthesis +0.07`, `hybrid +0.06`, `sparse +0.03`

**Trim strategy:** when `AGENT_MAX_CONTEXT_TOKENS > 0`, keep adding ranked
chunks until accumulated content tokens (char-density estimate) exceed the
budget. Falls back to `AGENT_MAX_CONTEXT_CHUNKS` cap when budget is 0.

**Reorder (Liu 2023 — lost in the middle):** when
`AGENT_LOST_IN_MIDDLE_REORDER=true`, ranked list `[r1, r2, r3, r4, r5]`
becomes `[r1, r3, r5, r4, r2]` — best at start AND end, weaker in the middle.
Empirically improves LLM attention on long contexts.

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
| `LLM_TASK_MODEL_MAP` | `"{}"` | JSON map task → model name. Tasks: `classify`, `decide`, `schema_discovery`, `sql_compile`, `synthesize`, `answer`, `mindmap`, `summary` |
| `LLM_COST_TRACKING_ENABLED` | `false` | Bật cost tracking (`GET /on/api/metrics/cost`) |
| `LLM_LARGE_CONTEXT_MODEL` | `None` | Auto-override khi prompt > threshold (open-notebook pattern). Bỏ trống = disabled |
| `LLM_LARGE_CONTEXT_THRESHOLD` | `100000` | Token cutoff để switch sang large-context model |
| `VISION_TIMEOUT_SECONDS` | `180` | Timeout cho vision LLM calls (llava cold-start có thể >60s) |
| `AGENT_MAX_CONTEXT_CHUNKS` | `8` | Legacy chunk-count cap (used when token budget = 0) |
| `AGENT_MAX_CONTEXT_TOKENS` | `6000` | Token-aware packed-context budget |
| `AGENT_LOST_IN_MIDDLE_REORDER` | `true` | Best chunks at start + end of packed context |
| `VISION_PROVIDER` | `None` | Provider cho `vision_response()`: `openai` / `gemini` / `ollama` |
| `VISION_MODEL` | `None` | Model cho vision calls (gpt-4o / gemini-1.5-flash / llava:13b) |
| `VISION_BASE_URL` | `None` | Override endpoint (chủ yếu cho Ollama) |

## Cost tracking

Process-global LLM ledger lives in `src/agentrag/observability/cost.py`.
Every call from `AgentLLM` (json/text/stream) and `LLMGateway.vision_response`
auto-records `(task, model, latency_ms, in_tokens, out_tokens, usd)` when
`LLM_COST_TRACKING_ENABLED=true`.

- Token counts: prefer provider `usage.prompt_tokens` / `completion_tokens` when
  the OpenAI-compat response surfaces them; otherwise char-density heuristic.
- USD estimate: per-model price table (Gemini 2.5 / 1.5 family, OpenAI 4o/4o-mini).
  Unknown models default to `gemini-2.5-flash` pricing.
- In-memory ring buffer of 5000 last calls. Cleared on process restart.

`LLMGateway.cost_summary()` aggregates by task + by model. Surfaced via
`GET /on/api/metrics/cost`; reset via `POST /on/api/metrics/cost/reset`.
