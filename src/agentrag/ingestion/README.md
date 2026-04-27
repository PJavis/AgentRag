# Module: `ingestion` — Ingestion Pipeline

**Vị trí:** `src/pam/ingestion/`

Entry point duy nhất để đưa dữ liệu vào hệ thống. Đọc tài liệu từ disk, parse, chunk, embed rồi lưu vào PostgreSQL + Elasticsearch. Sau đó trigger StructMem extraction (sync hoặc async).

---

## Files & Sub-modules

### `pipeline.py` — Orchestrator

`ingest_folder(folder_path, graph_ingest_mode)` — hàm chính, xử lý toàn bộ workflow.

### `connectors/`

| File | Class | Mô tả |
|---|---|---|
| `folder.py` | `FolderConnector` | Duyệt thư mục, trả về danh sách file paths theo extension |
| `markdown.py` | `MarkdownConnector` | Đọc file `.md` → raw text |

### `parsers/`

| File | Class | Mô tả |
|---|---|---|
| `markitdown_parser.py` | `MarkItDownParser` | Parse PDF/DOCX/PPTX/HTML → Markdown text (dùng `microsoft/markitdown`) |
| `excel_parser.py` | `ExcelParser` | Parse XLSX/CSV — 2 mode: `markdown` (table → text) hoặc `sql` (→ SQLite) |

### `chunkers/`

| File | Class | Mô tả |
|---|---|---|
| `hybrid_chunker.py` | `HybridChunker` | Tạo 2 lớp chunks: search (512 tok) và graph (1536 tok) |

### `embedders/`

| File | Class | Mô tả |
|---|---|---|
| `base.py` | `BaseEmbedder` | Abstract interface |
| `factory.py` | `EmbedderFactory` | Tạo embedder theo `EMBEDDING_PROVIDER` |
| `openai_embedder.py` | `OpenAIEmbedder` | OpenAI text-embedding API |
| `gemini_embedder.py` | `GeminiEmbedder` | Google Gemini embedding API |
| `hf_inference_embedder.py` | `HFInferenceEmbedder` | HuggingFace Inference API |

### `stores/`

| File | Class | Mô tả |
|---|---|---|
| `postgres_store.py` | `PostgresStore` | Upsert documents + segments vào PostgreSQL |
| `elasticsearch_store.py` | `ElasticsearchStore` | Upsert chunks + entries + synthesis vào ES |

---

## Luồng xử lý

```
ingest_folder(folder_path)
  │
  ├──▶ FolderConnector.scan()                → danh sách files
  │
  └──▶ for each file:
          ├── .md                → MarkdownConnector.read()
          ├── .pdf/.docx/.pptx  → MarkItDownParser.parse()
          ├── .xlsx/.csv        → ExcelParser.parse()
          │
          ├──▶ HybridChunker.chunk()
          │       search_chunks (512 tok, overlap 64)
          │       graph_chunks  (1536 tok, overlap 128)
          │
          ├──▶ Embedder.embed_batch(search_chunks)
          │
          ├──▶ PostgresStore.upsert_document() + upsert_segments()
          ├──▶ ElasticsearchStore.index_chunks()
          │
          └──▶ [STRUCTMEM_ENABLED]
                 mode=sync  → StructMemService.sync_chunks() trực tiếp
                 mode=async → graph_jobs_queue.put(GraphJob(...))
```

---

## Định dạng hỗ trợ

| Extension | Parser |
|---|---|
| `.md` | MarkdownConnector (raw text) |
| `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt` | MarkItDownParser |
| `.html`, `.htm` | MarkItDownParser |
| `.xlsx`, `.xls` | ExcelParser |
| `.csv` | ExcelParser |
| `.txt` | MarkdownConnector (raw text) |

---

## Tương tác

| Module | Vai trò |
|---|---|
| `graph.graph_jobs` | Nhận `GraphJob` qua queue (async mode) |
| `graph.StructMemService` | Chạy extraction trực tiếp (sync mode) |
| `database.AsyncSessionLocal` | Lưu document + segment metadata |
| `main.py` | Expose `/ingest/folder` và `/ingest/upload` |

---

## Config liên quan

| Key | Default | Mô tả |
|---|---|---|
| `SEARCH_CHUNK_MAX_TOKENS` | `512` | Token/chunk cho search layer |
| `SEARCH_CHUNK_OVERLAP_TOKENS` | `64` | Overlap giữa search chunks |
| `GRAPH_CHUNK_MAX_TOKENS` | `1536` | Token/chunk cho graph/StructMem layer |
| `GRAPH_CHUNK_OVERLAP_TOKENS` | `128` | Overlap giữa graph chunks |
| `SEARCH_CHUNK_BY_PARAGRAPH` | `true` | Chia theo đoạn văn thay vì cắt cứng |
| `EMBEDDING_PROVIDER` | `hf_inference` | Provider embedding |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large-instruct` | Model embedding |
| `EMBEDDING_BATCH_SIZE` | `32` | Số chunks embed mỗi batch |
| `EXCEL_INGEST_MODE` | `markdown` | `markdown` hoặc `sql` cho Excel files |
| `GRAPH_INGEST_MODE` | `async` | `sync` hoặc `async` |
| `STRUCTMEM_ENABLED` | `true` | Bật StructMem extraction sau ingest |
