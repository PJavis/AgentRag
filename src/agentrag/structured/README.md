# Module: `structured` — SQL Reasoning Pipeline

**Vị trí:** `src/agentrag/structured/`

Pipeline suy luận có cấu trúc cho các câu hỏi so sánh, tổng hợp, xếp hạng. Thay vì semantic search, pipeline tự khám phá schema từ chunks, extract dữ liệu dạng bảng, rồi chạy SQL trên SQLite in-memory để trả lời.

---

## Files

| File | Class | Mô tả |
|---|---|---|
| `query_classifier.py` | `QueryIntentClassifier` | Phân loại intent: `structured` hay `semantic` |
| `schema_discovery.py` | `SchemaDiscoveryService` | Khám phá schema từ chunks liên quan |
| `extractor.py` | `DataExtractor` | Extract rows từ chunks theo schema đã khám phá |
| `sql_engine.py` | `SQLEngine` | Compile + execute SQL trên SQLite in-memory |
| `synthesizer.py` | `AnswerSynthesizer` | Chuyển kết quả SQL → câu trả lời tự nhiên |
| `pipeline.py` | `StructuredReasoningPipeline` | Orchestrator 5 bước |

---

## Luồng xử lý

```
question
  │
  1. QueryIntentClassifier.classify()
     ├── method: "rule"     → regex pattern matching
     ├── method: "llm"      → LLM phân loại
     └── method: "rule+llm" → rule trước, LLM xác nhận nếu rule uncertain
     │
     confidence < STRUCTURED_CONFIDENCE_THRESHOLD → fallback semantic
     │
  2. SchemaDiscoveryService.discover()
     ├── retrieval top-K chunks liên quan (STRUCTURED_MAX_CHUNKS_FOR_SCHEMA)
     └── LLM → schema: {entity, attributes: [{name, type, description}]}
     │
  3. DataExtractor.extract()
     ├── retrieval thêm chunks (STRUCTURED_MAX_CHUNKS_FOR_EXTRACT)
     └── LLM → rows[] theo schema, với validation CLEAR A+B
     │
  4. SQLEngine.compile_and_run()
     ├── tạo SQLite table từ schema
     ├── insert rows
     ├── LLM sinh SQL query
     └── execute → result set (retry tối đa STRUCTURED_SQL_MAX_RETRIES)
     │
  5. AnswerSynthesizer.synthesize()
     └── LLM → câu trả lời tự nhiên từ result set + context
```

---

## `QueryIntentClassifier`

Phân loại câu hỏi thành `structured` hoặc `semantic`.

```python
output = await classifier.classify(question, document_title, chat_history)
# output.intent: "structured" | "semantic"
# output.query_type: "comparison" | "aggregation" | "ranking" | None
# output.confidence: float 0..1
```

Trigger keywords cho rule-based: so sánh, hơn, kém, cao nhất, thấp nhất, tổng, trung bình, bao nhiêu, danh sách, xếp hạng...

---

## Fallback

Nếu `result["_structured_fallback"] = True` → `AgentService` tiếp tục chạy semantic path thay vì trả về kết quả structured.

Các trường hợp fallback:
- Confidence < threshold
- Schema discovery thất bại (không tìm đủ chunks)
- SQL execution lỗi sau hết retries
- Result set rỗng

---

## Tương tác

| Module | Vai trò |
|---|---|
| `agent.AgentService` | Gọi classifier + pipeline, nhận kết quả hoặc fallback |
| `services.LLMGateway` | Gọi LLM cho classify, schema, extract, SQL, synthesize |
| `services.KnowledgeService` | Retrieval chunks cho schema discovery + extraction |

---

## Config liên quan

| Key | Default | Mô tả |
|---|---|---|
| `STRUCTURED_REASONING_ENABLED` | `true` | Bật/tắt toàn bộ structured path |
| `STRUCTURED_CLASSIFIER_METHOD` | `rule+llm` | Phương pháp classify: `rule`, `llm`, `rule+llm` |
| `STRUCTURED_CONFIDENCE_THRESHOLD` | `0.7` | Ngưỡng confidence để đi structured path |
| `STRUCTURED_MAX_CHUNKS_FOR_SCHEMA` | `10` | Số chunks tối đa cho schema discovery |
| `STRUCTURED_MAX_CHUNKS_FOR_EXTRACT` | `20` | Số chunks tối đa cho data extraction |
| `STRUCTURED_SQL_MAX_RETRIES` | `2` | Số lần retry khi SQL lỗi |
