# Module: `ingestion` — Ingestion Pipeline

**Vị trí:** `src/agentrag/ingestion/`

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
| `pdf_parser.py` | `PDFParser` | **Page-aware** PDF parser dùng PyMuPDF (`fitz`) — trả `parsed_content` với page markers + `page_data` per-page + `extract_images()` |
| `markitdown_parser.py` | `MarkItDownParser` | Parse DOCX/PPTX/HTML → Markdown (legacy fallback, không có page info) |
| `image_parser.py` | `ImageParser` | Mô tả ảnh standalone (.jpg/.png/.webp/...) qua Vision LLM, sinh text description để embed |
| `excel_parser.py` | `ExcelParser` | Parse XLSX/CSV — 2 mode: `markdown` (table → text) hoặc `sql` (→ SQLite) |

**PDF routing**: `PDF_PARSER_BACKEND=pymupdf` (mặc định) → `PDFParser`; `markitdown` → `MarkItDownParser` (mất thông tin page).

**Page markers** (`\x00P{N}\x00`) là ký tự không in được, embed vào full text. Chunker tự strip + assign `page_start`/`page_end` cho mỗi chunk → enable NotebookLM-style citations với page number cụ thể.

**Image extraction từ PDF**: `PDFParser.extract_images()` lưu ảnh vào `IMAGE_STORAGE_DIR/{slug(title)}/p{page}_{idx}.{ext}` rồi pass cho `ImageParser` mô tả → tạo image segments có `segment_type="image"`, `image_url`, `page`.

### `chunkers/`

| File | Class | Mô tả |
|---|---|---|
| `hybrid_chunker.py` | `HybridChunker` | Tạo 2 lớp chunks: search (512 tok) và graph (1536 tok). Tự strip page markers từ PDFParser và assign `page_start`/`page_end` per-chunk |

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
          ├── .md / .txt        → MarkdownConnector.read()
          ├── .pdf              → PDFParser.parse()  ← page-aware (mặc định)
          │                        → extract_images() nếu VISION_PROVIDER set
          ├── .docx/.pptx/.html → MarkItDownParser.parse()
          ├── .jpg/.png/...     → ImageParser.parse() (Vision LLM)
          ├── .xlsx/.csv        → ExcelParser.parse()
          │
          ├──▶ HybridChunker.chunk()
          │       search_chunks (512 tok, overlap 64)
          │       graph_chunks  (1536 tok, overlap 128)
          │       — tự strip page markers + gán page_start/page_end
          │
          ├──▶ Embedder.embed_batch(search_chunks)
          │
          ├──▶ PostgresStore.upsert_document() + upsert_segments()
          ├──▶ ElasticsearchStore.index_chunks()
          │       (text segments + image segments có segment_type="image")
          │
          └──▶ [STRUCTMEM_ENABLED]
                 mode=sync  → StructMemService.sync_chunks() trực tiếp
                 mode=async → arq_pool.enqueue_job("graph_ingest", ...)
```

---

## Định dạng hỗ trợ

| Extension | Parser | Notes |
|---|---|---|
| `.md`, `.txt` | MarkdownConnector | Raw text |
| `.pdf` | PDFParser (PyMuPDF) | Page-aware. Tự extract ảnh nếu `VISION_PROVIDER` set |
| `.docx`, `.doc`, `.pptx`, `.ppt`, `.html`, `.htm` | MarkItDownParser | Không có page info |
| `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif` | ImageParser | Cần `VISION_PROVIDER` + `VISION_MODEL` |
| `.xlsx`, `.xls`, `.csv` | ExcelParser | 2 mode: markdown / SQL |

---

## Tương tác

| Module | Vai trò |
|---|---|
| `graph.graph_jobs` | Nhận `GraphJob` qua queue (async mode) |
| `graph.StructMemService` | Chạy extraction trực tiếp (sync mode) |
| `services.LLMGateway` | `vision_response()` cho ImageParser (PDF images + standalone) |
| `database.AsyncSessionLocal` | Lưu document + segment metadata |
| `main.py` | Expose `/ingest/folder` và `/ingest/upload`. `/images/*` static mount cho ảnh extract |
| `adapter.routers.sources` | Wrapper REST khớp open-notebook contract |

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
| `PDF_PARSER_BACKEND` | `pymupdf` | `pymupdf` (page-aware) hoặc `markitdown` (legacy) |
| `VISION_PROVIDER` | `None` | `openai` / `gemini` / `ollama`. `None` = bỏ qua image parsing |
| `VISION_MODEL` | `None` | VD: `gpt-4o`, `gemini-1.5-flash`, `llava:13b` |
| `VISION_BASE_URL` | `None` | Override endpoint nếu cần (Ollama / custom) |
| `IMAGE_STORAGE_DIR` | `data/images` | Thư mục lưu ảnh extract từ PDF + standalone uploads |
| `IMAGE_MIN_SIZE_BYTES` | `5000` | Skip ảnh nhỏ hơn ngưỡng (icons, decorative bullets) |
| `VISION_INGEST_MODE` | `async` | `sync` = describe inline (blocks pipeline); `async` = queue ARQ vision_extract job |
| `VISION_TIMEOUT_SECONDS` | `180` | Timeout vision LLM call (llava cold-start) |
| `VISION_MAX_CONCURRENCY` | `4` | Parallel describes in vision_extract job |
| `VISION_MAX_RPM` | `10` | RPM cap (0 = disabled) — Gemini free tier = 10 |
| `VISION_PER_IMAGE_RETRIES` | `3` | Transient error retry per image |
| `VISION_FLUSH_BATCH_SIZE` | `10` | Commit segments to PG+ES every N images |
| `ORIGINALS_DIR` | `data/originals` | Persist original uploaded bytes for `'Open original'` UI button. Empty = discard after ingest |
