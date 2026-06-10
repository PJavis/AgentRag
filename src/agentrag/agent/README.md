# agent — semantic reasoning loop (Reasoning Plane)

**Vị trí:** `src/agentrag/agent/`

## Mục đích / Purpose

Vòng lặp suy luận chính của AgentRag. Nhận câu hỏi của người dùng, phân loại
intent, tự chọn tool retrieval cần gọi, tích lũy context qua nhiều bước, rồi
sinh câu trả lời cuối kèm citations. Đây là "bộ não" quyết định *WHAT to do* —
nó không tự gọi IO mà điều phối các execution-plane services. Có cả blocking
(`GraphAgentService.chat`) và streaming SSE (`chat_stream`).

## Plane

**Reasoning Plane.** Module này sở hữu state machine (LangGraph), prompts,
decision loops và branching. Nó KHÔNG khởi tạo concrete IO classes — mọi
execution service được lấy qua `ServiceContainer.get_container()` trong
`AgentService.__init__`. (Ngoại lệ tồn dư: `agent/tools.py::AgentTools` vẫn
`new ElasticsearchRetriever()` trực tiếp — boundary leak đã ghi nhận trong
`ARCHITECTURE.md`.)

## Key files

| File | Class / Function | Responsibility |
|---|---|---|
| `graph_service.py` | `GraphAgentService`, 16-node `StateGraph` | **Orchestrator chính.** LangGraph state machine wiring các phase; mỗi node gọi helper trên `_INNER = AgentService()`. Resumable per `thread_id = conversation_id` (InMemorySaver). |
| `service.py` | `AgentService` | Helper container — security/knowledge/classifier/structured pipeline + các pha `_decide`/`_answer`/`_plan_subqueries`/`_critique`/`_build_packed_citations`. Cũng có `chat_stream` (đường streaming SSE riêng, không qua graph). |
| `context.py` | `ContextAssembler`, `_lost_in_middle_reorder` | Dedup → rank → token-budget trim → global rerank → citation-pack context trước khi đưa vào LLM. |
| `llm.py` | `AgentLLM` | Wrapper `AsyncOpenAI` (OpenAI-compatible). Resolve backend/provider, json/text/stream/multimodal responses, sticky model fallback. |
| `tools.py` | `AgentTools` | Tool registry — dispatch search_sparse/dense/hybrid/hybrid_kg + segment/chunk lookups. Đọc S5 `domain_filter` qua ContextVar. |
| `factory.py` | `get_agent_service()` | Stable entrypoint — trả về `GraphAgentService` (single backend shim). |
| `followups.py` | `generate_followups()` | Sau khi trả lời, 1 LLM call rẻ (task `followup`) đề xuất 3 câu hỏi tiếp. TTLCache 5 phút. |
| `starters.py` | `generate_starters()` | Empty-state starter questions từ title + summary (task `starter`). TTLCache 10 phút. |

## Public interface

Callers (adapter routers, CLI) chỉ import **qua factory**:

```python
from src.agentrag.agent.factory import get_agent_service
agent = get_agent_service()          # → GraphAgentService
result = await agent.chat(question, document_title, chat_history,
                          conversation_id, domain_filter, verbosity)
```

- `GraphAgentService.chat(...) -> dict` — blocking. Trả về `answer`, `citations`,
  `tool_trace`, `reasoning_path`, `sql_query`, `highlights`, `timings_ms`,
  `context` (= packed_context, cho RAGAS eval), và 3 UI signals
  `semantic_cache_hit` / `retrieval_mode` / `domain_route` (xem `_message_signals`).
- `GraphAgentService.chat_stream(...) -> AsyncIterator[str]` — SSE. **Chưa
  port sang graph**; nó delegate thẳng tới `AgentService.chat_stream` (loop
  thủ công). Events: `status` / `token` / `done` / `error`.
- `generate_followups(...)` / `generate_starters(...)` — gọi trực tiếp từ
  `adapter/routers/chat.py`, nhận `llm_gateway` injected.

`AgentService` không phải public — public surface của nó (helper methods) chỉ
được graph nodes tiêu thụ qua `_INNER`.

## Data flow

Upstream callers: `adapter/routers/chat.py`, `adapter/routers/search.py`,
`cli/chat.py`.

LangGraph node sequence (`graph_service._build_graph`):

```
validate → memory → chitchat_check
  ├─[chitchat]→ chitchat_answer → END
  └→ classify
       ├─[structured]→ structured_run ─[ok]→ ground
       │                              └[fallback]→ semantic_plan
       ├─[adaptive fast-path]→ fast_answer → critique
       └→ semantic_plan → bootstrap → decide
                                       ├─[more]→ tool_exec → decide  (loop ≤ AGENT_MAX_STEPS)
                                       └─[done]→ assemble → answer → critique
   critique ─[grounded]→ ground → END
            └[not grounded]→ corrective_retrieve → critique  (≤ AGENT_CRITIQUE_MAX_RETRIES)
```

Per-node work:
- `validate` → `SecurityService.validate_chat_request`
- `memory` → `ChatMemoryService.retrieve` (chỉ khi `CHAT_STRUCTMEM_ENABLED`)
- `classify` → `QueryIntentClassifier.classify` (gate `STRUCTURED_REASONING_ENABLED`)
- `structured_run` → `StructuredReasoningPipeline.run` (đường SQL)
- `semantic_plan` → `AgentService._plan_subqueries` (decompose multi-hop)
- `bootstrap` → `KnowledgeService.bootstrap_search` (+ sub-query fan-out, parallel hoặc multi-hop-chained) → `SecurityService.filter_tool_results`
- `decide` → `AgentService._decide` (LLM tự-phản tỉnh chọn tool kế tiếp)
- `tool_exec` → `KnowledgeService.normalize_tool_call` + `execute_tool` → filter
- `assemble` → `ContextAssembler.assemble`
- `answer` → `AgentService._answer` (LLM synthesis, multimodal nếu có image segments)
- `critique` → `AgentService._critique` (CRAG, no extra LLM call)
- `ground` → `AgentService._build_packed_citations` + `_attach_source_ids`

Downstream deps (qua `ServiceContainer` / services package): `KnowledgeService`,
`ContextAssemblyService`, `SecurityService`, `LLMGateway`,
`StructuredReasoningPipeline`, `QueryIntentClassifier`, `ChatMemoryService`.
`ContextAssembler` còn pull `retrieval.reranker.LLMReranker` trực tiếp cho
global rerank pass.

### Citation contract (quan trọng)

Answer prompt cite theo **source number** `[n]` = vị trí trong
`packed_context` (1-based). `_build_packed_citations` trả về **toàn bộ** danh
sách packed (mỗi entry tagged `source = n`), nên mọi `[n]` đều resolve được —
khác với việc ground subset citation tự do của model. `_lost_in_middle_reorder`
chỉ áp lên **bản prompt**, packed_context trả về vẫn giữ relevance-order để eval
+ `[n]` map đúng `retrieval_context[n-1]`.

## Config

| Key | Default | Mô tả |
|---|---|---|
| `AGENT_MAX_STEPS` | `4` | Số bước decide→tool tối đa mỗi request |
| `AGENT_TOOL_TOP_K` | `5` | top_k mặc định khi LLM không chỉ định |
| `AGENT_MAX_CONTEXT_CHUNKS` | `8` | Chunk-count cap (chỉ dùng khi token budget = 0) |
| `AGENT_MAX_CONTEXT_TOKENS` | `6000` | Token-aware budget cho packed context (ưu tiên) |
| `AGENT_MAX_OUTPUT_TOKENS` | `131072` | max_tokens cho mọi `AgentLLM` call |
| `AGENT_LOST_IN_MIDDLE_REORDER` | `true` | Reorder prompt copy: best ở đầu + cuối |
| `AGENT_PLAN_THEN_EXECUTE_ENABLED` | `true` | Planner decompose → parallel sub-retrieval |
| `AGENT_PLAN_TRIGGER_MIN_CHARS` | `60` | Skip planner cho câu ngắn (trừ summary intent) |
| `AGENT_PLAN_MAX_SUBQUERIES` | `4` | Cap sub-queries mỗi plan |
| `STRUCTURED_REASONING_ENABLED` | `true` | Bật nhánh classify + SQL reasoning |
| `CHAT_STRUCTMEM_ENABLED` | `true` | Semantic chat memory thay sliding-window history |
| `RETRIEVAL_RERANK_ENABLED` | `false` | Global cross-encoder rerank trong `ContextAssembler` |
| `CRAG_ENABLED` | `false` | Bật critique + corrective re-retrieve loop |
| `CRAG_MIN_HITS` | `1` | Số passage tối thiểu để coi là đủ context |
| `CRAG_GROUNDING_ENABLED` | `true` | Critique fail nếu thiếu citation / answer mơ hồ |
| `AGENT_CRITIQUE_MAX_RETRIES` | `1` | Số lần corrective_retrieve tối đa |
| `AGENT_MULTIHOP_ENABLED` | `false` | Sub-queries chạy tuần tự + chain snippet hop trước |
| `ADAPTIVE_ROUTING_ENABLED` | `false` | Cho phép fast_answer node (skip plan→decide loop) |
| `ADAPTIVE_FASTPATH_MIN_CONFIDENCE` | `0.85` | Ngưỡng confidence để vào fast-path |
| `AGENT_PROVIDER` / `AGENT_MODEL` / `AGENT_TEMPERATURE` / `AGENT_BASE_URL` | (fallback `EXTRACTION_*`) | Backend cho `AgentLLM` |
| `LLM_FALLBACK_MODEL` | `qwen2.5:7b-instruct` | Sticky fallback khi model 404 |
| `LLM_OLLAMA_NUM_CTX` | `32768` | num_ctx ép cho Ollama base URL (11434) |
| `LLM_TASK_MODEL_MAP` | `"{}"` | Per-task model routing (answer/decide/plan/classify/followup/starter…) |

## Recent additions (2026-06)

Tất cả default-OFF trừ khi ghi rõ. Branch `feat/ragas-langfuse-reranker`:

- **CRAG critique + corrective** (`CRAG_ENABLED`). `critique` node gọi
  `AgentService._critique` — kiểm tra relevance (`len(packed) >= CRAG_MIN_HITS`)
  + grounding (có citation, không phải câu "không tìm thấy…"), **không tốn LLM
  call**. Nếu fail → `corrective_retrieve` (step-back query rewrite → re-retrieve
  → re-answer), loop về critique tới `AGENT_CRITIQUE_MAX_RETRIES`. `timings_ms.critique`
  surface verdict latency cho UI.
- **Multi-hop chaining** (`AGENT_MULTIHOP_ENABLED`). Trong `bootstrap`, sub-queries
  chạy tuần tự; `_chain_query` seed mỗi hop bằng top snippet của hop trước
  ("Bối cảnh: …\n\nCâu hỏi: …"). Khi OFF → fan-out song song qua `asyncio.gather`.
- **Adaptive fast-path** (`ADAPTIVE_ROUTING_ENABLED`). `_route_intent` chuyển sang
  node `fast_answer` khi classifier báo `complexity == "simple"` + `single_domain`
  + `confidence >= ADAPTIVE_FASTPATH_MIN_CONFIDENCE` → một retrieve + một answer,
  bỏ qua plan→decide→tool loop. `reasoning_path: "fast"`.
- **UI-signal shim.** `_build_packed_citations` / `ContextAssembler._stage_citation_pack`
  carry `node_level` (RAPTOR layer) + `context_text` (Contextual Retrieval prefix).
  `_message_signals` derive `semantic_cache_hit` / `retrieval_mode` / `domain_route`
  từ tool_trace cho UI chips.

> **Lưu ý:** các flag `AGENT_SELF_CRITIQUE_*` trong README cũ **đã bị bỏ** — pha
> self-critique được thay bằng CRAG (`_critique` + critique node). Contextual
> Retrieval / RAPTOR / semantic cache nằm ở `ingestion/` + `services/`; module này
> chỉ *truyền* các signal của chúng ra UI, không tự bật chúng.

## Gotchas

- **`chat_stream` đi đường khác `chat`.** Streaming vẫn dùng loop thủ công trong
  `AgentService.chat_stream` (chit-chat → classify → bootstrap → decide loop →
  assemble → stream tokens). Nó **không** chạy CRAG critique / corrective /
  adaptive fast-path / multi-hop. Sửa logic ở graph nodes sẽ KHÔNG ảnh hưởng
  streaming — phải sửa cả hai.
- **`GraphAgentService` tái dùng một instance toàn cục** `_INNER = AgentService()`
  + một `_GRAPH` compiled. State pickle qua InMemorySaver nên `seen_calls` được
  serialize thành list (không phải set).
- **Verbose/summary follow-up rewrite.** `GraphAgentService.chat` viết lại câu
  hỏi ngắn kiểu "viết dài hơn được không?" bằng cách prepend câu hỏi user gần
  nhất (`effective_question`) — nếu không retrieval miss sạch. Câu trả về vẫn là
  `question` gốc.
- **`AgentLLM` provider auto-derive.** Khi `model_override` bắt đầu bằng
  `gemini-`/`gemma-`/`gpt-`/`o1`/`o3`/`deepseek`, provider được suy ra tự động để
  một entry trong `LLM_TASK_MODEL_MAP` nhắm backend khác `AGENT_PROVIDER`. Model
  404 → sticky fallback `LLM_FALLBACK_MODEL` (giữ tới hết process).
- **`json_response` ép chữ "json".** DeepSeek (và vài backend) từ chối
  `response_format=json_object` nếu prompt không chứa chữ "json" — `AgentLLM` tự
  chèn câu nhắc. `<think>…</think>` được strip qua `clean_thinking_content`.
- **Answer-shape recovery.** Finetune models hay trả sai top-level key
  (`{"summary": …}`, `{"search_results": […]}`, hoặc decide-shape). `_find_answer_field`
  walk cây để cứu; cuối cùng synthesize câu "chưa tìm được…" để UI không hiện
  bubble rỗng.
- **`AgentTools` là boundary leak.** Tự `new ElasticsearchRetriever()` thay vì lấy
  từ container; đọc `domain_filter` + document scope qua ContextVar
  (`retrieval.context`), không qua tham số.
- **Coverage diversity trong trim.** `ContextAssembler._stage_rank_trim` cap ≤ 3
  chunk / (document, page/section) bucket ở pass đầu rồi backfill — tránh dồn hết
  budget vào trang 1; còn force-inject ít nhất 1 StructMem/graph candidate nếu có.
