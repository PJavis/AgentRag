# ADR 0002 — Structured SQL Reasoning Pipeline (DocSage-inspired)

## Context

PAM hiện tại xử lý mọi câu hỏi qua một pipeline duy nhất:

```
Câu hỏi → hybrid_kg retrieval → Context Assembly → LLM generate answer
```

Pipeline này hoạt động tốt cho các câu hỏi **ngữ nghĩa đơn giản** ("Tính năng X là gì?",
"Mô tả Y"). Nhưng yếu với các câu hỏi **có cấu trúc logic**:

- **So sánh**: "So sánh hiệu suất của A, B và C theo tiêu chí X"
- **Thống kê / Tổng hợp**: "Có bao nhiêu entity thỏa điều kiện Y?"
- **Multi-hop Relationship**: "Tìm tất cả Z có quan hệ với A và thuộc loại B"

Vấn đề cốt lõi: LLM bị **attention diffusion** khi phải giữ nhiều entity trong context
dài và tự lý luận — dễ bỏ sót, nhầm lẫn, hoặc hallucinate.

## Quyết định

Thêm một **nhánh reasoning có cấu trúc** song song với nhánh semantic hiện tại.
Khi câu hỏi được phân loại là "structured", hệ thống sẽ:

1. Trích xuất văn bản → bảng quan hệ (relational tables)
2. Biên dịch câu hỏi → SQL
3. Thực thi SQL trên in-memory SQLite
4. Tổng hợp câu trả lời từ kết quả SQL + provenance chain

Nhánh semantic cũ **không thay đổi**, đảm bảo backward-compatibility hoàn toàn.

---

## Kiến trúc tổng thể

```
POST /chat
    │
    ▼
AgentService.chat()
    │
    ├─── QueryIntentClassifier
    │         │
    │    intent = "semantic"          intent = "structured"
    │         │                              │
    │         ▼                              ▼
    │   [Nhánh cũ]              StructuredReasoningPipeline
    │   bootstrap_search                     │
    │   agent loop                   ┌───────┴────────┐
    │   context assembly             │                │
    │   LLM answer                   ▼                ▼
    │                         KnowledgeService   SchemaDiscoveryModule
    │                         (bootstrap_search)      │
    │                                │                ▼
    │                                └──────► StructuredExtractor
    │                                                  │
    │                                                  ▼
    │                                         SQLReasoningEngine
    │                                                  │
    │                                                  ▼
    │                                         AnswerSynthesizer
    │
    └────────────────────────────────────────────────────────►
                          Unified Response (same schema)
```

---

## Vị trí module mới

```
src/pam/
└── structured/
    ├── __init__.py
    ├── pipeline.py          # StructuredReasoningPipeline (orchestrator)
    ├── query_classifier.py  # QueryIntentClassifier
    ├── schema_discovery.py  # SchemaDiscoveryModule
    ├── extractor.py         # StructuredExtractor (+ CLEAR validation)
    ├── sql_engine.py        # SQLReasoningEngine
    └── synthesizer.py       # AnswerSynthesizer
```

---

## Chi tiết từng thành phần

---

### 1. QueryIntentClassifier

**File:** `src/pam/structured/query_classifier.py`

**Mục đích:**
Phân loại câu hỏi vào 2 nhánh xử lý: `semantic` (nhánh cũ) hoặc `structured`
(nhánh mới). Đây là **gate** quyết định luồng xử lý. Cần nhanh và chính xác.

**Cách hoạt động:**
Kết hợp 2 lớp:
- **L1 — Rule-based (fast):** Regex/keyword detect các pattern cấu trúc rõ ràng.
  Nếu match → quyết định ngay, không cần LLM call.
- **L2 — LLM-based (fallback):** Với câu hỏi mơ hồ không match rule, gọi LLM
  classify với few-shot examples. Dùng model nhẹ (haiku / flash).

**Query types được classify là "structured":**
| Type | Ví dụ |
|---|---|
| `comparison` | "So sánh A và B", "A vs B theo tiêu chí X" |
| `aggregation` | "Có bao nhiêu X", "Tổng Y của Z", "Trung bình..." |
| `ranking` | "Top 3 X tốt nhất", "X nào lớn nhất/nhỏ nhất" |
| `multi_filter` | "Tìm tất cả X có Y và Z" |
| `multi_hop` | "A → B → C qua quan hệ nào?" |

**Input:**
```python
@dataclass
class ClassifierInput:
    question: str          # Câu hỏi của user
    document_title: str | None  # Scope document nếu có
    chat_history: list[dict] | None  # Lịch sử hội thoại
```

**Output:**
```python
@dataclass
class ClassifierOutput:
    intent: Literal["semantic", "structured"]
    query_type: str | None   # "comparison" | "aggregation" | "ranking" | ...
    confidence: float        # 0.0 - 1.0
    reasoning: str           # Lý do phân loại (dùng cho debug)
    method: Literal["rule", "llm"]  # L1 hay L2
```

**Tích hợp với PAM:**
Được gọi bởi `AgentService.chat()` ngay sau `security.validate_chat_request()`,
trước khi bất kỳ retrieval nào xảy ra.

---

### 2. StructuredReasoningPipeline

**File:** `src/pam/structured/pipeline.py`

**Mục đích:**
Orchestrator của nhánh structured. Điều phối tuần tự 4 bước:
Retrieval → Schema Discovery → Extraction → SQL Reasoning → Synthesis.

Trả về response cùng schema với `AgentService.chat()` để API không thay đổi.

**Cách hoạt động:**
```
run(question, document_title, chat_history)
    │
    ├─ Step 1: Gọi KnowledgeService.bootstrap_search() để lấy candidate chunks
    │          (tái sử dụng hoàn toàn infrastructure hiện có)
    │
    ├─ Step 2: Gọi SchemaDiscoveryModule.discover(question, chunks)
    │          → Nhận schema: danh sách tables, columns, relationships
    │
    ├─ Step 3: Gọi StructuredExtractor.extract(chunks, schema)
    │          → Nhận DB: dict[table_name → list[row_dict]]
    │
    ├─ Step 4: Gọi SQLReasoningEngine.execute(question, schema, db)
    │          → Nhận sql_result: ResultSet + SQL query string
    │
    └─ Step 5: Gọi AnswerSynthesizer.synthesize(question, sql_result, chunks)
               → Nhận answer + citations + provenance
```

**Input:**
```python
@dataclass
class PipelineInput:
    question: str
    document_title: str | None
    chat_history: list[dict] | None
    query_type: str          # từ QueryIntentClassifier
```

**Output:** Cùng schema với `AgentService.chat()`:
```python
{
    "question": str,
    "document_title": str | None,
    "answer": str,
    "citations": list[CitationDict],
    "tool_trace": list[ToolTraceEntry],  # ghi lại mỗi bước
    "context": list[PackedContextItem],
    "reasoning_path": "structured",      # thêm field mới để trace
    "sql_query": str,                    # SQL đã execute (cho transparency)
    "timings_ms": TimingsDict,
}
```

---

### 3. SchemaDiscoveryModule

**File:** `src/pam/structured/schema_discovery.py`

**Mục đích:**
Suy ra **minimal relational schema** đặc thù cho câu hỏi, từ các candidate chunks
đã retrieve được. Schema không cứng định trước mà được suy ra động (query-aware).

Lấy cảm hứng từ **ASK algorithm** của DocSage nhưng đơn giản hóa: PAM không
thực hiện multi-round dialogue với document set; thay vào đó, schema được suy ra
một lần từ chunks đã có.

**Cách hoạt động:**

```
Step 1 — Sample Analysis:
    Lấy top-N chunks (mặc định N=10)
    Gọi LLM với prompt:
      "Given question Q and these text samples, what relational tables
       and columns would you need to answer this? Output minimal schema."

Step 2 — Schema Validation:
    Kiểm tra schema có đủ để answer question không:
    - Có entity chính không?
    - Có attribute cần so sánh không?
    - Có join key giữa tables không (nếu multi-table)?

Step 3 — Schema Finalization:
    Trả về Schema object đã validated
```

**Input:**
```python
@dataclass
class SchemaDiscoveryInput:
    question: str
    query_type: str           # "comparison" | "aggregation" | ...
    candidate_chunks: list[ChunkDict]   # từ bootstrap search
    document_title: str | None
```

**Output:**
```python
@dataclass
class RelationalSchema:
    tables: list[TableDef]
    # Ví dụ:
    # TableDef(
    #   name="product",
    #   columns=["name", "price", "category", "rating"],
    #   primary_key="name",
    #   description="Product entities mentioned in docs"
    # )
    join_keys: list[JoinKey]   # foreign key relationships giữa tables
    query_focus: list[str]     # columns LLM xác định là quan trọng cho question này
    source_chunks_used: list[str]  # content_hash của chunks đã dùng để suy schema
```

**Tích hợp với PAM:**
Nhận chunks từ `KnowledgeService` (đã có sẵn). Gọi `LLMGateway.json_response()`
để infer schema — tái sử dụng LLM gateway hiện có.

---

### 4. StructuredExtractor

**File:** `src/pam/structured/extractor.py`

**Mục đích:**
Chuyển đổi text chunks → rows trong các bảng theo schema đã định. Đây là bước
**data transformation** cốt lõi: từ unstructured text → structured relational data.

Kết hợp với **CLEAR-inspired validation** để đảm bảo chất lượng extraction.

**Cách hoạt động:**

```
Step 1 — Per-chunk Extraction:
    Với mỗi chunk, gọi LLM:
      "Extract rows for table schema S from this text.
       Return JSON array of row objects. Return [] if not applicable."

Step 2 — CLEAR Validation (Level A — Single-row checks):
    Với mỗi row được extract:
    - Type check: giá trị numeric có thực sự là số không?
    - Null check: primary key có bị null không?
    - Range check: các giá trị có trong khoảng hợp lý không?
      (Ví dụ: rating 1-5, phần trăm 0-100)

Step 3 — CLEAR Validation (Level B — Cross-row consistency):
    Sau khi collect all rows vào in-memory dict:
    - Functional dependency check: cùng primary key → cùng attribute value?
      (Nếu không: conflict → keep row có confidence cao hơn)
    - Duplicate detection: merge rows giống nhau từ các chunks khác nhau

Step 4 — Provenance Tagging:
    Gắn source chunk metadata vào mỗi row:
    {row_data..., "_source_chunk_hash": "abc123", "_source_doc": "DocTitle"}
```

**Input:**
```python
@dataclass
class ExtractionInput:
    chunks: list[ChunkDict]       # candidate chunks
    schema: RelationalSchema      # từ SchemaDiscoveryModule
    question: str
```

**Output:**
```python
@dataclass
class ExtractionOutput:
    database: dict[str, list[dict]]
    # Ví dụ:
    # {
    #   "product": [
    #     {"name": "X", "price": 100, "rating": 4.5, "_source_chunk_hash": "abc"},
    #     {"name": "Y", "price": 200, "rating": 3.8, "_source_chunk_hash": "def"},
    #   ]
    # }
    extraction_stats: ExtractionStats
    # {table_name: {total_rows, valid_rows, conflict_rows, empty_chunks}}
    conflicts: list[ConflictRecord]  # Log các conflict đã detect + resolution
```

**Tích hợp với PAM:**
Gọi `LLMGateway.json_response()` cho mỗi chunk (có thể async batch để giảm latency).
Validation logic là pure Python, không cần external dependency.

---

### 5. SQLReasoningEngine

**File:** `src/pam/structured/sql_engine.py`

**Mục đích:**
Biên dịch câu hỏi tự nhiên → SQL, thực thi trên in-memory SQLite database
chứa data đã extract. Đây là bước **thay thế LLM reasoning** bằng SQL engine
deterministic — cốt lõi của toàn bộ kiến trúc này.

**Tại sao SQLite?**
- Không cần infra mới (built-in Python)
- In-memory, lifecycle gắn với request
- Full SQL: JOIN, GROUP BY, ORDER BY, subqueries
- Provenance: có thể trace từng row về source chunk

**Cách hoạt động:**

```
Step 1 — DB Hydration:
    Tạo in-memory SQLite connection
    CREATE TABLE cho mỗi bảng trong schema
    INSERT tất cả rows từ ExtractionOutput

Step 2 — SQL Compilation:
    Gọi LLM với prompt:
      "Given schema S and question Q, write a SQLite SQL query.
       Available tables: {schema}. Return only the SQL."
    LLM nhìn thấy schema rõ ràng → generate SQL chính xác

Step 3 — SQL Execution:
    Thực thi SQL trên SQLite
    Nếu lỗi syntax: retry với error message (tối đa 2 lần)
    Nếu empty result: fallback về nhánh semantic

Step 4 — Result Mapping:
    Với mỗi row trong result → trace về source chunk hash
    Gắn provenance: row → chunk → document
```

**Input:**
```python
@dataclass
class SQLEngineInput:
    question: str
    schema: RelationalSchema
    database: dict[str, list[dict]]   # từ StructuredExtractor
    query_type: str
```

**Output:**
```python
@dataclass
class SQLEngineOutput:
    sql_query: str              # SQL đã execute (cho transparency)
    result_rows: list[dict]     # Kết quả thô từ SQL
    provenance: list[ProvenanceRecord]
    # ProvenanceRecord: {row_index, source_chunk_hash, source_doc, source_section}
    execution_ok: bool
    fallback_reason: str | None  # Nếu execution_ok=False, lý do fallback
    retry_count: int
```

**Tích hợp với PAM:**
- Dùng `sqlite3` (stdlib) → không cần dependency mới
- Gọi `LLMGateway.json_response()` cho SQL compilation
- Nếu `execution_ok=False` → `StructuredReasoningPipeline` fallback về
  nhánh semantic của `AgentService` (graceful degradation)

---

### 6. AnswerSynthesizer

**File:** `src/pam/structured/synthesizer.py`

**Mục đích:**
Tổng hợp câu trả lời tự nhiên từ SQL result set + provenance chain, đảm bảo
mỗi claim đều traceable về source document.

**Cách hoạt động:**

```
Step 1 — Result Formatting:
    Format result_rows thành readable text
    Ví dụ: table → markdown table; single value → direct statement

Step 2 — Answer Generation:
    Gọi LLM với:
      - question
      - formatted SQL results
      - provenance (list of source chunks + excerpts)
    → Generate câu trả lời tự nhiên

Step 3 — Citation Grounding:
    Map citations về `_source_chunk_hash` trong provenance
    Chỉ cite chunks có trong provenance chain (không hallucinate citation)
    Dùng lại `AgentService._ground_citations()` logic hiện có
```

**Input:**
```python
@dataclass
class SynthesizerInput:
    question: str
    sql_result: SQLEngineOutput
    candidate_chunks: list[ChunkDict]   # để lấy excerpt cho citations
    query_type: str
    chat_history: list[dict] | None
```

**Output:**
```python
@dataclass
class SynthesizerOutput:
    answer: str
    citations: list[CitationDict]
    sql_result_summary: str    # human-readable summary của SQL result
```

---

## Luồng dữ liệu đầy đủ

```
User: "So sánh rating của sản phẩm A, B, C"
    │
    ▼
QueryIntentClassifier
    intent="structured", query_type="comparison", confidence=0.92
    │
    ▼
StructuredReasoningPipeline.run()
    │
    ├─► KnowledgeService.bootstrap_search(question, document_title)
    │       → 10 chunks (hybrid_kg search, infrastructure hiện có)
    │       ChunkDict: {content, section_path, document_title, content_hash, score}
    │
    ├─► SchemaDiscoveryModule.discover(question, chunks)
    │       → RelationalSchema:
    │           tables=[TableDef(name="product", columns=["name","rating","price"])]
    │           query_focus=["name", "rating"]
    │
    ├─► StructuredExtractor.extract(chunks, schema)
    │       → ExtractionOutput:
    │           database={
    │             "product": [
    │               {"name":"A","rating":4.5,"_source_chunk_hash":"h1"},
    │               {"name":"B","rating":3.8,"_source_chunk_hash":"h2"},
    │               {"name":"C","rating":4.1,"_source_chunk_hash":"h3"},
    │             ]
    │           }
    │
    ├─► SQLReasoningEngine.execute(question, schema, database)
    │       SQL: "SELECT name, rating FROM product ORDER BY rating DESC"
    │       Result: [{"name":"A","rating":4.5}, {"name":"C","rating":4.1}, {"name":"B","rating":3.8}]
    │       Provenance: [{row=0, chunk_hash="h1", doc="DocX"}, ...]
    │
    └─► AnswerSynthesizer.synthesize(question, sql_result, chunks)
            Answer: "Xếp hạng theo rating: A (4.5) > C (4.1) > B (3.8)..."
            Citations: [
              {document_title:"DocX", section_path:"products", content_hash:"h1"},
              ...
            ]
    │
    ▼
AgentService trả về unified response:
{
  "answer": "Xếp hạng theo rating: A (4.5) > C (4.1) > B (3.8)...",
  "citations": [...],
  "reasoning_path": "structured",
  "sql_query": "SELECT name, rating FROM product ORDER BY rating DESC",
  "timings_ms": {total, classify, retrieve, schema, extract, sql, synthesize}
}
```

---

## Fallback Strategy

Tại mỗi bước, nếu thất bại → graceful fallback về nhánh semantic:

```
QueryIntentClassifier fails
    → default intent = "semantic" (nhánh cũ)

SchemaDiscoveryModule fails / empty schema
    → fallback = "semantic"

StructuredExtractor fails / 0 rows extracted
    → fallback = "semantic"

SQLReasoningEngine fails / empty result
    → fallback = "semantic"

AnswerSynthesizer fails
    → return raw SQL result as plain text
```

Fallback được log trong `tool_trace` với `reasoning_path: "structured→fallback_semantic"`.

---

## Thay đổi cần thiết trong code hiện tại

### AgentService (`src/pam/agent/service.py`)

Thêm classify step trước bootstrap:
```python
# THÊM: classify query intent
classifier_output = await self.query_classifier.classify(question, document_title)

if classifier_output.intent == "structured":
    # Nhánh mới
    return await self.structured_pipeline.run(
        question=question,
        document_title=document_title,
        chat_history=chat_history,
        query_type=classifier_output.query_type,
    )

# Nhánh cũ — không thay đổi
bootstrap_input, bootstrap_output = await self.knowledge.bootstrap_search(...)
...
```

### Config (`src/pam/config.py`)

Thêm các config mới:
```python
# Structured reasoning
STRUCTURED_REASONING_ENABLED: bool = True
STRUCTURED_CLASSIFIER_METHOD: str = "rule+llm"   # "rule" | "llm" | "rule+llm"
STRUCTURED_MAX_CHUNKS_FOR_SCHEMA: int = 10
STRUCTURED_MAX_CHUNKS_FOR_EXTRACT: int = 20
STRUCTURED_SQL_MAX_RETRIES: int = 2
STRUCTURED_CONFIDENCE_THRESHOLD: float = 0.7
```

### Response schema (`main.py`)

Thêm 2 optional fields vào `/chat` response:
```python
"reasoning_path": "semantic" | "structured" | "structured→fallback_semantic"
"sql_query": str | None   # None nếu reasoning_path = "semantic"
```

---

## Dependencies mới

| Package | Lý do | Có sẵn chưa? |
|---|---|---|
| `sqlite3` | In-memory SQL engine | Có (Python stdlib) |
| Không có thêm | Mọi LLM call qua LLMGateway | Tái sử dụng |

**Không cần thêm dependency nào.** Toàn bộ dùng lại infrastructure hiện có của PAM.

---

## Tradeoffs

| Ưu điểm | Nhược điểm |
|---|---|
| Accuracy cao hơn cho câu hỏi có cấu trúc | Latency cao hơn (nhiều LLM calls hơn) |
| Deterministic reasoning (SQL) | Extraction có thể thiếu nếu chunks nghèo |
| Provenance rõ ràng, traceable | Schema discovery có thể sai với domain mới |
| Không thay đổi nhánh semantic cũ | Cần nhiều LLM tokens hơn |
| Fallback graceful | SQL chỉ đúng khi extraction đúng |

---

## Roadmap triển khai

### Phase 1 — Foundation (ưu tiên)
- [ ] `QueryIntentClassifier` với L1 rule-based (không cần LLM)
- [ ] `StructuredReasoningPipeline` skeleton
- [ ] `SchemaDiscoveryModule` (LLM-based, single-pass)
- [ ] `StructuredExtractor` (basic extraction, no CLEAR yet)
- [ ] `SQLReasoningEngine` (sqlite3, basic SQL)
- [ ] `AnswerSynthesizer` (reuse `AgentService._answer()` logic)
- [ ] Integration vào `AgentService`
- [ ] Config flags + fallback

### Phase 2 — Quality
- [ ] L2 LLM-based classifier cho edge cases
- [ ] CLEAR Level A validation (type/null/range checks)
- [ ] CLEAR Level B validation (cross-row consistency)
- [ ] SQL retry với error feedback
- [ ] Async batch extraction (parallel LLM calls per chunk)

### Phase 3 — Advanced
- [ ] Schema caching (per query_type + document scope)
- [ ] Multi-table JOIN support
- [ ] Evaluation trên test cases

---

## Liên quan

- ADR 0001: Target Architecture Rollout (Phase B — Intent-aware retrieval)
- DocSage paper: arXiv:2603.11798v1
- PAM structured module: `src/pam/structured/`
