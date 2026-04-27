# Module: `agent` — Semantic Reasoning Loop

**Vị trí:** `src/pam/agent/`

Vòng lặp suy luận ngữ nghĩa chính. Nhận câu hỏi, tự chọn tool retrieval cần gọi, tích lũy context qua nhiều bước, rồi sinh câu trả lời cuối. Hỗ trợ cả blocking (`chat`) và streaming SSE (`chat_stream`).

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
  ├─[STRUCTURED_REASONING_ENABLED]──▶ QueryIntentClassifier.classify()
  │         │  intent == "structured"
  │         └──▶ StructuredReasoningPipeline.run() ──▶ return kết quả SQL
  │
  ├──▶ KnowledgeService.bootstrap_search()       ← warm retrieval bước đầu
  │
  └──▶ for step in range(AGENT_MAX_STEPS - 1):
          _decide()                               ← LLM chọn tool tiếp theo
          KnowledgeService.execute_tool()         ← chạy tool
          SecurityService.filter_tool_results()   ← lọc theo document_title
       │
       ▼
       ContextAssemblyService.assemble()          ← dedup + rank + trim
       _answer()                                  ← LLM sinh answer + citations
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
  "answer": "...",
  "citations": [{"document_title": "...", "section_path": "...", "content_hash": "..."}],
  "reasoning_path": "semantic | structured",
  "sql_query": null,
  "tool_trace": [...],
  "context": [...],
  "timings_ms": {"total": 0, "decide": 0, "tool": 0, "assemble": 0, "answer": 0}
}
```

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
| `AGENT_MAX_CONTEXT_CHUNKS` | `8` | Số chunks tối đa trong packed context |
| `STRUCTURED_REASONING_ENABLED` | `true` | Bật/tắt nhánh SQL reasoning |
| `AGENT_PROVIDER` | (fallback EXTRACTION_PROVIDER) | LLM provider cho agent |
| `AGENT_MODEL` | (fallback EXTRACTION_MODEL) | LLM model cho agent |
| `AGENT_TEMPERATURE` | (fallback EXTRACTION_TEMPERATURE) | Temperature cho agent calls |
