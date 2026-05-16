# Module: `agent` — Semantic Reasoning Loop (Reasoning Plane)

**Vị trí:** `src/agentrag/agent/`

> S4 — Reasoning Plane. `AgentService.__init__` fetch services qua
> `ServiceContainer.get_container()` (không tự `new ElasticsearchRetriever()`).
> Đọc `ARCHITECTURE.md` cho luật chia plane.

Vòng lặp suy luận ngữ nghĩa chính. Nhận câu hỏi, tự chọn tool retrieval cần gọi, tích lũy context qua nhiều bước, rồi sinh câu trả lời cuối. Hỗ trợ cả blocking (`chat`) và streaming SSE (`chat_stream`).

Output chat response bao gồm `tool_trace`, `timings_ms`, `reasoning_path`,
`plan_subqueries`, `sql_query` — adapter persist trên assistant message để
UI `TraceDialog` (S2) hiển thị.

---

## Files

| File | Class / Function | Mô tả |
|---|---|---|
| `service.py` | `AgentService` | Orchestrator chính — quản lý toàn bộ vòng lặp |
| `tools.py` | `ToolRegistry` | Khai báo và dispatch retrieval tools |
| `context.py` | `ContextAssemblyService` | Dedup, rank, trim context trước khi đưa vào LLM |
| `llm.py` | `AgentLLM` | Wrapper OpenAI-compatible client cho decision + streaming |

---

## Luồng xử lý

```
question
  │
  ├─[chit-chat heuristic]──▶ _is_chitchat()  → reply via cheap routing model, skip retrieval
  │
  ├─[STRUCTURED_REASONING_ENABLED]──▶ QueryIntentClassifier.classify()
  │         │  intent == "structured"
  │         └──▶ StructuredReasoningPipeline.run() ──▶ return kết quả SQL
  │
  ├─[AGENT_PLAN_THEN_EXECUTE_ENABLED, len≥trigger]──▶ _plan_subqueries()
  │         │ multi_step=true
  │         └──▶ asyncio.gather(bootstrap_search(sq) for sq in subqueries)
  │
  ├──▶ KnowledgeService.bootstrap_search()       ← bootstrap on original question
  │
  └──▶ for step in range(AGENT_MAX_STEPS - 1):
          _decide()                               ← LLM chọn tool tiếp theo
          KnowledgeService.execute_tool()         ← chạy tool
          SecurityService.filter_tool_results()   ← lọc theo document_title
       │
       ▼
       ContextAssemblyService.assemble()          ← dedup + rank + token-budget trim + LiM reorder
       _answer()                                  ← LLM sinh answer + citations
      [AGENT_SELF_CRITIQUE_ENABLED, thin retrieval] ──▶ _self_critique() → maybe revise
       _ground_citations()                        ← validate citations vs context
```

---

## API chính

### `AgentService.chat(question, document_title, chat_history) → dict`

Blocking. Trả về:

```json
{
  "question": "...",
  "document_title": "...",
  "answer": "Van **hai lá** nằm giữa tâm nhĩ trái và tâm thất trái...",
  "highlights": [
    "Van hai lá gồm 2 lá van và bộ máy dưới van",
    "Hở van hai lá độ 3-4 cần phẫu thuật"
  ],
  "citations": [
    {
      "document_title": "...",
      "section_path": "Chương 3 / Hệ tim mạch",
      "content_hash": "...",
      "page_start": 47,
      "page_end": 48,
      "excerpt": "Van hai lá nằm giữa..."
    }
  ],
  "reasoning_path": "semantic | structured",
  "sql_query": null,
  "tool_trace": [...],
  "context": [...],
  "timings_ms": {"total": 0, "decide": 0, "tool": 0, "assemble": 0, "answer": 0}
}
```

**Page-aware citations**: nếu document được parse bằng `PDFParser` (PyMuPDF), `page_start`/`page_end` sẽ có giá trị → frontend hiện trang chính xác như NotebookLM. Markdown/DOCX không có page info → các trường này = `null`.

**Highlights**: 3-5 điểm quan trọng nhất sinh trực tiếp trong answer prompt (term + bullet). Câu trả lời cũng dùng `**bold**` cho thuật ngữ quan trọng.

### `AgentService.chat_stream(question, document_title, chat_history) → AsyncIterator[str]`

SSE generator. Mỗi yield là `"event: <type>\ndata: <json>\n\n"`.

| Event | Payload | Khi nào |
|---|---|---|
| `status` | `{"step": "classify\|retrieve\|decide\|tool\|answer"}` | Đầu mỗi bước xử lý |
| `token` | `{"text": "..."}` | Mỗi token LLM sinh |
| `done` | `{citations, reasoning_path, sql_query, tool_trace}` | Kết thúc |
| `error` | `{"message": "..."}` | Bất kỳ exception |

---

## `AgentLLM`

Wrapper `AsyncOpenAI`. Tự resolve backend từ `AGENT_PROVIDER` / `EXTRACTION_PROVIDER`.

| Method | Mô tả |
|---|---|
| `json_response(system, user)` | Gọi LLM với `response_format=json_object` |
| `stream_text(system, user)` | Stream raw tokens qua `stream=True` |

---

## Tương tác

| Module | Vai trò |
|---|---|
| `services.KnowledgeService` | Bootstrap + execute retrieval tools |
| `services.ContextAssemblyService` | Assemble + rank + trim context |
| `services.SecurityService` | Filter results theo document_title |
| `services.LLMGateway` | json_response cho decide + answer |
| `structured.StructuredReasoningPipeline` | SQL reasoning khi intent = structured |
| `structured.QueryIntentClassifier` | Phân loại intent câu hỏi |

---

## Config liên quan

| Key | Default | Mô tả |
|---|---|---|
| `AGENT_MAX_STEPS` | `4` | Số bước tool tối đa mỗi request |
| `AGENT_TOOL_TOP_K` | `5` | top_k mặc định khi LLM không chỉ định |
| `AGENT_MAX_CONTEXT_CHUNKS` | `8` | Legacy chunk-count cap (used when token budget = 0) |
| `AGENT_MAX_CONTEXT_TOKENS` | `6000` | Token-aware budget for packed context |
| `AGENT_LOST_IN_MIDDLE_REORDER` | `true` | Reorder packed context: best at start + end |
| `AGENT_SELF_CRITIQUE_ENABLED` | `false` | 2nd LLM call verifies draft against context |
| `AGENT_SELF_CRITIQUE_RRF_THRESHOLD` | `0.05` | Critique only when top RRF below threshold |
| `AGENT_PLAN_THEN_EXECUTE_ENABLED` | `true` | Planner decomposes multi-hop → parallel sub-retrieval |
| `AGENT_PLAN_TRIGGER_MIN_CHARS` | `60` | Skip planner for shorter questions |
| `AGENT_PLAN_MAX_SUBQUERIES` | `4` | Cap on sub-queries per plan |
| `STRUCTURED_REASONING_ENABLED` | `true` | Bật/tắt nhánh SQL reasoning |
| `AGENT_PROVIDER` | (fallback EXTRACTION_PROVIDER) | LLM provider cho agent |
| `AGENT_MODEL` | (fallback EXTRACTION_MODEL) | LLM model cho agent |
| `AGENT_TEMPERATURE` | (fallback EXTRACTION_TEMPERATURE) | Temperature cho agent calls |
| `AGENT_BACKEND` | `loop` | `loop` = legacy `AgentService.chat()`; `langgraph` = `GraphAgentService` (13-node StateGraph w/ checkpoint+replay) |

## Backends

- **`loop` (default)** — `service.AgentService.chat()`, hand-rolled. Battle-tested.
- **`langgraph`** — `graph_service.GraphAgentService`, wraps existing node helpers in a `StateGraph` with `InMemorySaver`. Same logic, exposes per-node state for resume / inspection. Select via `agent.factory.get_agent_service()`.

## Chit-chat fast-path

`_is_chitchat()` is a rule-based detector — short messages (≤60 chars) containing
greeting/thanks tokens (`hi`, `chào`, `thanks`, `cảm ơn`, `how are you`, ...)
**and** no information-request signal (`?`, `tại sao`, `what`, `tóm tắt`, ...)
get a brief warm reply via the `classify` task client (cheapest routing model).
No retrieval, no citations. `reasoning_path: "chitchat"` in the response.

## Self-critique pass

When `AGENT_SELF_CRITIQUE_ENABLED=true` and top retrieval RRF score is below
`AGENT_SELF_CRITIQUE_RRF_THRESHOLD`, the draft answer is sent back to the
`decide` task client for audit. Returns strict JSON:

```json
{"verdict": "ok|unsupported|sycophantic", "issues": [...], "revised": "..."}
```

When verdict is `unsupported` / `sycophantic`, the revised answer replaces the
draft before being returned to the user. The `critique` field in the chat
response surfaces the verdict for debugging.

## Plan-then-execute

For questions ≥ `AGENT_PLAN_TRIGGER_MIN_CHARS` chars, the agent calls a
planner LLM first:

```json
{"multi_step": true, "subqueries": ["...", "...", "..."]}
```

Each sub-query is dispatched to `KnowledgeService.bootstrap_search` in
parallel via `asyncio.gather`. The original-question bootstrap still runs
afterwards as a safety net. The reactive `_decide` loop then usually
short-circuits because evidence is pre-collected. Result includes
`plan_subqueries: [...]` and `timings_ms.plan`.
