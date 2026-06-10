# structured — intent classification + SQL reasoning pipeline (ADR 0002)

## Mục đích / Purpose

Module này phục vụ 2 vai trò Reasoning Plane gắn liền nhau:

1. **Intent classification** (`query_classifier.py`) — phân loại mỗi câu hỏi thành
   `semantic` (nhánh RAG thường) hay `structured` (nhánh SQL reasoning), đồng thời
   ước lượng `complexity` (simple/complex) + `single_domain` để phục vụ adaptive
   fast-path routing.
2. **Structured reasoning pipeline** (`pipeline.py` + 4 stage modules) — cho các câu
   hỏi so sánh / đếm / tổng hợp / xếp hạng / lọc nhiều điều kiện, thay vì semantic
   search thuần, pipeline tự khám phá schema từ chunks, extract dữ liệu dạng bảng,
   sinh + chạy SQL trên **SQLite in-memory**, rồi synthesize câu trả lời có trích dẫn.

Triết lý: thay phần "reasoning số học/so sánh" mong manh của LLM bằng một SQL engine
deterministic; LLM chỉ lo schema inference, extraction, SQL compile và synthesis.

## Plane

**Reasoning Plane.** Module quyết định *cách trả lời* (rule/LLM classify, branching
structured vs semantic, sinh SQL, synthesize). Nó **không** tự instantiate IO — mọi
LLM call đi qua `LLMGateway`, mọi retrieval đi qua `KnowledgeService`, được inject vào
constructor từ `AgentService` / `GraphAgentService` (xem ARCHITECTURE.md hàng "Intent
classifier" và "SQL pipeline" — đều R). SQLite in-memory là cơ chế tính toán nội bộ,
không phải store bền vững nên không vi phạm ranh giới.

## Key files

| File | Class | Trách nhiệm |
|---|---|---|
| `query_classifier.py` | `QueryIntentClassifier`, `ClassifierOutput` | L1 rule (regex) + L2 LLM fallback; phân loại intent, query_type, complexity, single_domain |
| `pipeline.py` | `StructuredReasoningPipeline` | Orchestrator 5 bước; graceful fallback ở mỗi bước qua sentinel dict |
| `schema_discovery.py` | `SchemaDiscoveryModule`, `RelationalSchema`, `TableDef`, `JoinKey` | 1 LLM call → suy minimal relational schema từ top-N chunks |
| `extractor.py` | `StructuredExtractor`, `ExtractionOutput`, `ExtractionStats` | Extract rows từ chunks (async batch), validate CLEAR Level A + B |
| `sql_engine.py` | `SQLReasoningEngine`, `SQLEngineOutput`, `ProvenanceRecord` | Compile SQL (LLM, có retry), chạy trên SQLite in-memory, map provenance |
| `synthesizer.py` | `AnswerSynthesizer`, `SynthesizerOutput` | SQL result + provenance → câu trả lời NL có citation đã ground |
| `__init__.py` | — | Chỉ export `QueryIntentClassifier`, `ClassifierOutput` |

## Public interface

### `QueryIntentClassifier`
Export trực tiếp từ package. Khởi tạo với `LLMGateway | None` (nếu None thì L2 bị skip).

```python
out: ClassifierOutput = await classifier.classify(
    question, document_title=None, chat_history=None,
)
# out.intent:        "semantic" | "structured"
# out.query_type:    "comparison" | "aggregation" | "ranking" | "multi_filter" | "multi_hop" | None
# out.confidence:    float 0..1
# out.method:        "rule" | "llm" | "default"
# out.complexity:    "simple" | "complex"     # dùng cho adaptive fast-path
# out.single_domain: bool                     # dùng cho adaptive fast-path
```

### `StructuredReasoningPipeline`
Import từ `src.agentrag.structured.pipeline`. Constructor nhận
`(knowledge_service, llm_gateway, security_service)` rồi tự dựng 4 stage modules bên trong.

```python
result: dict = await pipeline.run(
    question, document_title, chat_history, query_type, classifier_confidence,
)
```

Trả về một trong hai shape:
- **Thành công** — dict tương thích `AgentService.chat()`:
  `{answer, citations, tool_trace, context, reasoning_path="structured", sql_query, timings_ms, ...}`.
- **Fallback** — sentinel dict `{"_structured_fallback": True, "_fallback_reason": str, "_trace": ...}`;
  caller phải detect và chuyển sang semantic path.

**Callers:** `agent/service.py` (`AgentService` — hand-rolled loop), `agent/graph_service.py`
(`GraphAgentService` — node `classify` → `structured_run`), và `mcp/{app,server}.py`
(gọi `pipeline.run` trực tiếp, **bỏ qua classifier**, hard-code `classifier_confidence=0.95`,
caller tự chỉ định `query_type`). Stage modules (Schema/Extractor/SQL/Synthesizer) là
nội bộ — không export ra ngoài package, chỉ pipeline dùng.

## Data flow

```
question
  │
  ├─ QueryIntentClassifier.classify()
  │     L1 _classify_l1(): scan _PATTERN_MAP theo ưu tiên
  │        comparison → aggregation → ranking → multi_filter → multi_hop
  │        match → intent=structured, confidence=0.95, method="rule"
  │     L1 không match + có llm_gateway + method∈{llm,rule+llm} → L2 _classify_l2() (1 LLM call)
  │     không match + không LLM → default semantic, confidence=0.5
  │  (mọi nhánh đều set complexity + single_domain qua _estimate_complexity)
  │
  └─ intent=="structured" → StructuredReasoningPipeline.run()
        1. Retrieval     KnowledgeService.bootstrap_search() → SecurityService.filter_tool_results()
        2. Schema        SchemaDiscoveryModule.discover()  (top STRUCTURED_MAX_CHUNKS_FOR_SCHEMA, mỗi chunk truncate 600 ký tự)
        3. Extraction    StructuredExtractor.extract()     (asyncio.gather mọi chunk×table; CLEAR A rồi CLEAR B)
        4. SQL           SQLReasoningEngine.execute()       (compile SQL → run SQLite :memory: → retry tối đa STRUCTURED_SQL_MAX_RETRIES)
        5. Synthesis     AnswerSynthesizer.synthesize()     (SQL result + provenance → answer + grounded citations)
```

Fallback (trả sentinel `_structured_fallback=True`) tại: retrieve lỗi · schema rỗng
(`schema.is_empty`) · extraction rỗng · SQL fail sau hết retry · **SQL chạy OK nhưng
result set rỗng** (coi như schema/cột bị hallucinate). Riêng synthesis fail dùng
"softer fallback" `_raw_sql_result()` — trả thẳng bảng SQL thô chứ KHÔNG quay về semantic.

**Upstream:** `AgentService` / `GraphAgentService` / MCP server.
**Downstream services:** `KnowledgeService.bootstrap_search` (retrieval),
`SecurityService.filter_tool_results` (lọc theo `document_title`),
`LLMGateway.json_response(system_prompt, user_prompt, task=...)` — gọi với `task` riêng
cho từng stage: `classify`, `schema_discovery`, `extract`, `sql_compile`, `synthesize`.
`synthesizer.py` còn import `MARKDOWN_FORMAT_RULES` từ `agent/service.py`.

## Config

`settings.*` từ `src/agentrag/config.py`:

| Key | Default | Tác dụng |
|---|---|---|
| `STRUCTURED_REASONING_ENABLED` | `True` | Gate toàn bộ nhánh structured. Off → classify bị skip, luôn semantic |
| `STRUCTURED_CLASSIFIER_METHOD` | `"rule+llm"` | `"rule"` (chỉ L1) / `"llm"` / `"rule+llm"` (L1 trước, L2 fallback) |
| `STRUCTURED_MAX_CHUNKS_FOR_SCHEMA` | `10` | Số chunk tối đa cho schema discovery |
| `STRUCTURED_MAX_CHUNKS_FOR_EXTRACT` | `20` | Số chunk tối đa cho extraction |
| `STRUCTURED_SQL_MAX_RETRIES` | `2` | Số lần re-compile SQL khi SQLite báo lỗi |
| `AGENT_PLAN_TRIGGER_MIN_CHARS` | `60` | Ngưỡng độ dài câu để `_estimate_complexity` coi là "long" |
| `ADAPTIVE_ROUTING_ENABLED` | `False` | (consumed ở `graph_service.py`) bật fast-path dựa trên `complexity`/`single_domain` |
| `ADAPTIVE_FASTPATH_MIN_CONFIDENCE` | `0.85` | (consumed ở `graph_service.py`) ngưỡng confidence để fast-path |

> **Lưu ý:** `STRUCTURED_CONFIDENCE_THRESHOLD` (`0.7`) **vẫn tồn tại trong config nhưng
> KHÔNG được đọc ở đâu** trong module/caller — định tuyến chỉ key trên `intent ==
> "structured"`, không so sánh confidence. Phiên bản README cũ mô tả "confidence <
> threshold → fallback" là **stale**.

## Recent additions (2026-06)

Adaptive fast-path routing. `ClassifierOutput` đã thêm 2 field `complexity:
"simple"|"complex"` và `single_domain: bool`, được tính ở **mọi** nhánh classify
(`_estimate_complexity`: dựa trên độ dài câu ≥ `AGENT_PLAN_TRIGGER_MIN_CHARS`,
`_COMPLEX_MARKERS` regex như "so sánh/tại sao/phân tích…", và multi-clause như " và ",
";", nhiều "?"). L1 mặc định coi mọi structured query_type là `complex` trừ
single-fact `aggregation`.

Hai field này **chỉ được tiêu thụ ngoài module** ở `agent/graph_service.py::_route_intent`:
khi `ADAPTIVE_ROUTING_ENABLED` và `complexity=="simple"` và `single_domain` và
`confidence >= ADAPTIVE_FASTPATH_MIN_CONFIDENCE` → route sang node `fast_answer`
(`reasoning_path="fast"`, bỏ qua plan→decide→tool loop). Bản thân module này chỉ
*sản xuất* tín hiệu; nó không tự routing. Cũng mới: query_type `multi_filter` và
`multi_hop` cùng các regex/few-shot tương ứng.

## Gotchas

- **Class tên KHÔNG khớp file cũ:** `SchemaDiscoveryModule` (không phải …Service),
  `StructuredExtractor` (không phải `DataExtractor`), `SQLReasoningEngine` (không phải
  `SQLEngine`). README cũ ghi sai — đã sửa.
- **Mọi giá trị SQLite là TEXT.** `_hydrate_db` tạo cột `... TEXT` và insert
  `str(value)`. Vì vậy `_SQL_SYSTEM` ép LLM dùng `CAST(col AS REAL)` cho mọi phép số
  học/so sánh. Quên CAST → so sánh chuỗi sai ("9" > "10").
- **SELECT-only được enforce 2 lớp:** prompt cấm DML, và `_run_sql` raise
  `sqlite3.Error` nếu câu không bắt đầu bằng `SELECT`. DB là `:memory:` mới mỗi lần chạy,
  đóng ngay sau (không persist).
- **Extraction là O(chunks × tables) LLM call**, chạy song song bằng `asyncio.gather(...,
  return_exceptions=True)`; exception của một (chunk,table) bị bỏ qua, không làm hỏng cả batch.
- **CLEAR validation 2 tầng:** Level A (`_validate_row_level_a`) drop row PK null/rỗng +
  coerce numeric-string → int/float. Level B (`_validate_cross_row_level_b`) dedup theo PK,
  và khi conflict (cùng PK khác giá trị) **giữ row từ chunk có `_source_position` nhỏ hơn**.
  Các cột `_source_*` (`_source_chunk_hash/_source_doc/_source_section/_source_position`) là
  metadata nội bộ, bị strip trước khi INSERT và bị lọc khỏi mọi bảng markdown.
- **Citation được ground 2 lần:** SQL `_map_provenance` join result→source qua PK; rồi
  `synthesize` chỉ giữ citation LLM trả mà `content_hash` nằm trong provenance
  (`allowed_hashes`). LLM bịa citation sẽ bị loại.
- **Ngôn ngữ trả lời tự suy từ câu hỏi:** `synthesizer._VI_RE` (regex ký tự có dấu
  tiếng Việt) quyết định Việt/Anh — không dựa vào setting hay tham số.
- **Schema prompt bias về cột giá:** `_SYSTEM_PROMPT` ép luôn include mọi cột chứa
  price/cost/iap/usd/vnd/gem/coin/gold/diamond…, và classifier có regex riêng cho mẫu
  "10 X giá bao nhiêu". Đây là tuning theo dữ liệu game/IAP, không phải y khoa — cân nhắc khi mở rộng.
- **MCP bỏ qua classifier hoàn toàn:** `mcp/{app,server}.py` gọi thẳng `pipeline.run`
  với `classifier_confidence=0.95` hard-code; caller phải tự truyền `query_type`.
