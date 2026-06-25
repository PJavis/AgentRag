# ingestion — đưa tài liệu vào PG + ES (parse → chunk → embed → index)

## Mục đích / Purpose
Entry point duy nhất để nạp dữ liệu vào hệ thống. `ingest_folder()` quét một thư
mục, parse từng file theo định dạng, chunk thành 2 lớp (search + graph), embed lớp
search, rồi lưu **PostgreSQL `segments`** (durable spine) + **Elasticsearch
`agentrag_segments`** (retrieval projection). Sau đó kích hoạt StructMem/KG
extraction (sync trực tiếp hoặc async qua ARQ). Đây là nơi các tính năng RAG mới
(Contextual Retrieval, RAPTOR, S5 section tagging, visual/CLIP embedding) cắm vào
*trước khi* embedding/indexing.

## Plane
**Execution Plane** — Infrastructure IO worker. Module này parse/chunk/embed/index;
không có vòng lặp reasoning, không có prompt-branching quyết định. Ngoại lệ đáng
chú ý: `contextualizer.py` và `raptor.py` *gọi LLM* (`LLMGateway.text_response`) để
sinh context/summary, nhưng đó là enrichment phụ trợ cố định trên đường ingest, không
phải decision logic. `section_tagger.py` dùng `ontology/resolver.py` (lookup, không
LLM).

## Key files
| File | Responsibility |
|---|---|
| `pipeline.py` | `ingest_folder()` — orchestrator của toàn bộ workflow per-document |
| `contextualizer.py` | WS1 Contextual Retrieval — sinh 1 câu "situating context" / chunk (file-cached) |
| `raptor.py` | WS2 RAPTOR — cluster embeddings + summarize đệ quy → summary nodes |
| `section_tagger.py` | S5 — gán `system_tag` / `specialty_tag` / `canonical_terms` cho chunk |
| `connectors/folder.py` | `FolderConnector` — quét đệ quy, map extension → `source_type`, tính `content_hash` |
| `connectors/markdown.py` | `MarkdownConnector` — legacy, chỉ quét `.md` |
| `parsers/pdf_parser.py` | `PDFParser` — page-aware (PyMuPDF) + Tesseract/vision/MinerU OCR escalation; page markers; `extract_images()` |
| `parsers/markitdown_parser.py` | `MarkItDownParser` — DOCX/PPTX/HTML → Markdown (không có page info) |
| `parsers/excel_parser.py` | `ExcelParser` — XLSX/XLS/CSV → markdown table hoặc CSV-for-SQL |
| `parsers/image_parser.py` | `ImageParser` — mô tả ảnh qua Vision LLM (standalone + PDF-extracted) |
| `parsers/audio_parser.py` | `AudioParser` — Whisper transcription (faster-whisper / OpenAI) với timestamp markers |
| `parsers/mineru_parser.py`, `pptx_via_mineru.py` | MinerU CLI shim (opt-in: layout + OCR + formula + table) |
| `chunkers/hybrid_chunker.py` | `HybridChunker` — split theo heading/paragraph/token; strip page markers → `page_start/page_end` |
| `embedders/base.py` | `BaseEmbeddingProvider` — interface + LRU cache + batch loop |
| `embedders/factory.py` | `build_embedding_provider(settings)` — chọn provider theo `EMBEDDING_PROVIDER` |
| `embedders/{openai,gemini,hf_inference}_embedder.py` | Concrete text embedders |
| `embedders/visual_embedder.py` | `VisualEmbedder` — CLIP text↔image singleton (P0.2 visual retrieval) |
| `stores/postgres_store.py` | `PostgresStore` — upsert Document + Segment, content_hash dedupe, skeleton-fill |
| `stores/elasticsearch_store.py` | `ElasticsearchStore` — index + search segments/entities/relationships/StructMem-memory |

## Public interface
- **`ingest_folder(folder_path, graph_ingest_mode=None, user_id=None) -> dict`**
  (`pipeline.py`) — hàm chính. Gọi trực tiếp (không qua ServiceContainer). Upstream:
  `adapter/routers/sources.py` (background task sau upload) và `eval/freshness.py`.
  Trả về report `{status, ingested, total, graph_ingest_mode, chunking, timings_ms_totals, documents[]}`.
- **`ElasticsearchStore`** (`stores/elasticsearch_store.py`) — KHÔNG chỉ dùng khi
  ingest. Đây là search backend cho retrieval: `sparse_search` / `dense_search` /
  `hybrid_search` (RRF fuse), `visual_search` (kNN trên `image_embedding`),
  `search_entries` / `search_synthesis` (StructMem), `fetch_all_segments` (whole-doc
  summary). Readers: `retrieval/elasticsearch_retriever.py`,
  `generation/summary_service.py`, `generation/mindmap_service.py`. Có shared
  process-wide client; `close_shared_es_client()` được gọi từ FastAPI lifespan shutdown.
- **`PostgresStore.save_document_and_segments(session, doc_data, chunks, project_id=None)`**
  → `(doc_id, status)` với `status ∈ {ingested, skipped}` (và đường `retry`/`failed`
  do pipeline xử lý). PG là source of truth cho full-text reconstruction + tool API.
- Parsers/chunkers/embedders được `pipeline.py` import & dựng trực tiếp (không qua
  Protocol). Embedders tuân `BaseEmbeddingProvider.embed(texts) -> list[list[float]]`.

## Data flow
```
ingest_folder(folder_path, graph_ingest_mode)
  mode = graph_ingest_mode or settings.STRUCTMEM_INGEST_MODE
  FolderConnector.list_documents()        → [{source_id, title, file_path, content_hash, source_type}]
  for each doc:
    ── parse theo source_type ──
      image  → ImageParser.parse()            (skip nếu VISION_PROVIDER=None)
      pdf    → PDFParser.parse() (+ extract_images() nếu vision bật)
      word   → MarkItDownParser.parse()
      excel/csv → ExcelParser.parse(mode=EXCEL_INGEST_MODE)
      audio  → AudioParser.parse()            (Whisper)
      markdown → Path.read_text()
    ── HybridChunker (search: SEARCH_CHUNK_*) → drop chunk < 80 chars ──
    ── HybridChunker (graph: STRUCTMEM_CHUNK_*) → chunks_graph (cho StructMem) ──
    ── [TAGGING_ENABLED]            SectionTagger.tag_chunk  → system/specialty/canonical
    ── [CONTEXTUAL_RETRIEVAL_ENABLED] Contextualizer         → chunk["context_text"]
    ── embed: _embed_input_for_chunk(c) → embedder.embed()   → chunk["embedding"]
    ── PostgresStore.save_document_and_segments() → (doc_id, status)
       skipped → next doc
    ── ElasticsearchStore.index_segments(chunks_search)
    ── [RAPTOR_ENABLED] RaptorBuilder.build() → index summary nodes
    ── mode=sync : StructMemService.sync_chunks() + index_structmem_views() → graph_status=done
       mode=async: graph_status="searchable" → enqueue ARQ "graph_ingest"
                   (+ cache parsed text vào STRUCTMEM_CACHE_DIR/parsed/)
    ── pending PDF vision images (async) → enqueue ARQ "vision_extract"
```
Downstream phụ thuộc: `graph/structmem_service.py`, `graph/structmem_sync.py`,
`worker/pool.py` (ARQ enqueue `graph_ingest` / `vision_extract`),
`services/llm_gateway.py`, `database/models.py` (`Document`, `Segment`, `Project`),
`ontology/resolver.py`, `common/progress.py` (publish "searchable" event).

## Embed-input vs cited content
`_embed_input_for_chunk(chunk)` quyết định text đem đi embed + BM25: nếu WS1 sinh
`context_text` thì embed `f"{context_text}\n\n{content}"`, ngược lại dùng `content`
thô. **`content` gốc luôn là cái được cite** — context chỉ giúp findability, không lộ
ra citation. ES lưu cả `content` và `context_text`; `sparse_search` boost
`context_text^1.5`.

## Config
Đọc từ `src/agentrag/config.py` (`settings.*`):

| Key | Default | Vai trò |
|---|---|---|
| `STRUCTMEM_INGEST_MODE` | `async` | `sync`=chờ StructMem xong; `async`=enqueue ARQ (mark "searchable" ngay) |
| `SEARCH_CHUNK_MAX_TOKENS` / `_OVERLAP_TOKENS` | `512` / `64` | Lớp search chunks |
| `SEARCH_CHUNK_BY_PARAGRAPH` | `True` | Split theo đoạn thay vì cắt cứng token |
| `STRUCTMEM_CHUNK_MAX_TOKENS` / `_OVERLAP_TOKENS` | `1536` / `128` | Lớp graph/StructMem chunks |
| `CHUNK_TOKENIZER_MODEL` | `text-embedding-3-large` | tiktoken model (fallback `cl100k_base` / `SimpleTokenizer`) |
| `EMBEDDING_PROVIDER` | `hf_inference` | `openai`/`gemini`/`hf_inference`/`ollama` |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large-instruct` | Model embedding |
| `EMBEDDING_BATCH_SIZE` | `32` | Batch size (cũng cap concurrency của Contextualizer) |
| `EXCEL_INGEST_MODE` | `markdown` | `markdown` hoặc `sql` |
| `TAGGING_ENABLED` | `True` | Bật SectionTagger (S5 domain tags) |
| `STRUCTMEM_INDEX` | `agentrag_memory_doc` | Index hợp nhất entry+synthesis (discriminator `kind`) |
| `STRUCTMEM_CACHE_DIR` | `.cache/agentrag/extract` | Cache parsed text cho ARQ worker (`/parsed/`) |
| `ELASTICSEARCH_INDEX_NAME` | `agentrag_segments` | Index segment retrieval |
| **PDF / OCR** | | |
| `PDF_PARSER_BACKEND` | `hybrid` | `hybrid`=PyMuPDF→Tesseract→vision; `mineru`=whole/thin-page MinerU. (Lưu ý gotcha) |
| `PDF_OCR_VISION_FALLBACK` | `True` | Cho phép gửi page ảnh qua VISION_PROVIDER khi text-layer mỏng |
| `PDF_MINERU_MIN_THIN_FRACTION` | `0.4` | Ngưỡng % trang mỏng để kích MinerU |
| `MINERU_BACKEND` / `MINERU_LANG` | `vlm-auto-engine` / `latin` | MinerU CLI config |
| `INGEST_USE_MINERU_FOR_PPTX` | `False` | PPTX → libreoffice→PDF→MinerU |
| **Vision images** | | |
| `VISION_PROVIDER` / `VISION_MODEL` | `None` | `None`=bỏ qua image parsing hoàn toàn |
| `VISION_INGEST_MODE` | `async` | `sync`=describe inline (block); `async`=ARQ `vision_extract` |
| `IMAGE_STORAGE_DIR` | `data/images` | Lưu ảnh extract; `IMAGE_MIN_SIZE_BYTES=5000` skip icon |
| `ORIGINALS_DIR` | `data/originals` | Giữ bytes gốc cho nút "Open original" |
| **Visual / CLIP** | | |
| `VISUAL_EMBEDDING_ENABLED` | `True` | Bật CLIP `image_embedding` + `visual_search` |
| `VISUAL_EMBEDDING_MODEL` / `_DIMS` / `_DEVICE` | `clip-ViT-B-32-multilingual-v1` / `512` / `auto` | CLIP model |
| **Audio** | | |
| `AUDIO_TRANSCRIBE_PROVIDER` | `faster_whisper` | `faster_whisper` (local) hoặc `openai` |
| `AUDIO_WHISPER_MODEL` / `_LANGUAGE` | `small` / `None` | `None`=auto-detect; `vi` ép tiếng Việt |
| **Contextual Retrieval (WS1)** | | |
| `CONTEXTUAL_RETRIEVAL_ENABLED` | `False` | Bật Contextualizer trước embed |
| `CONTEXTUAL_RETRIEVAL_TASK` | `contextualize` | LLM task key (model-map) |
| `CONTEXTUAL_MAX_DOC_CHARS` | `48000` | Clip doc đưa vào system prompt (cache prefix) |
| `CONTEXTUAL_CACHE_DIR` | `.cache/agentrag/context` | File cache `(sig, doc_hash, chunk_hash)` |
| **RAPTOR (WS2)** | | |
| `RAPTOR_ENABLED` | `False` | Bật RaptorBuilder sau khi index leaf segments |
| `RAPTOR_MAX_LEVELS` / `_MIN_LEAVES` / `_CLUSTER_SIZE` | `3` / `8` / `5` | Tham số cây |
| `RAPTOR_SUMMARY_TASK` | `raptor_summary` | LLM task key |

## Recent additions (2026-06)
Tất cả mặc định **OFF** trừ khi ghi chú khác — flag-gated, không đổi hành vi baseline.

- **WS1 Contextual Retrieval** (`contextualizer.py`, flag `CONTEXTUAL_RETRIEVAL_ENABLED`):
  chèn 1 câu situating-context / chunk vào `chunk["context_text"]` **trước embed**.
  Cả document đi vào *system prompt* (prefix ổn định → provider context-cache reuse),
  chỉ passage thay đổi. Kết quả file-cached keyed `(provider_sig, doc_hash, chunk_hash)`
  → backfill idempotent. ES lưu `context_text` (mapping mới) và boost nó trong
  `sparse_search`. Citation vẫn dùng `content` gốc.
- **WS2 RAPTOR** (`raptor.py`, flag `RAPTOR_ENABLED`): sau khi index leaf segments,
  `RaptorBuilder.build()` cluster embeddings (UMAP→GaussianMixture, fallback
  contiguous), summarize mỗi cluster bằng LLM, embed summary, đệ quy đến root. Summary
  nodes (`segment_type="raptor_summary"`, mang `node_level`, `child_ids`, domain tags
  union từ con) được append vào **cùng index `agentrag_segments`** qua `index_segments`.
- **S5 SectionTagger** (`section_tagger.py`, flag `TAGGING_ENABLED`, default **ON**):
  resolve `section_path` qua `ontology/resolver.py`, fallback `find_in_text()` trên
  content → `system_tag` / `specialty_tag` / `canonical_terms`. ES dùng làm `terms`
  filter cho domain routing (`_tag_filter_clauses`).
- **UI-signal shim**: ES segment mapping + `_normalize_hits` giờ expose `node_level`
  và `context_text` để citation downstream gắn được node-level (leaf vs RAPTOR summary)
  và context text vào UI.
- **Visual retrieval (P0.2)**: `embedders/visual_embedder.py` (CLIP) + cột
  `image_embedding` trên segment index + `ElasticsearchStore.visual_search()`. Gate
  `VISUAL_EMBEDDING_ENABLED`.

## Gotchas
- **Hai định nghĩa `PDF_PARSER_BACKEND` trong `config.py`.** Cái đầu là
  `Literal["pymupdf","markitdown"] = "pymupdf"`, cái sau (`str = "hybrid"`) **ghi đè**
  trong Pydantic → giá trị thực tế là `"hybrid"`. Mọi PDF luôn route qua `PDFParser`
  (pipeline `_PYMUPDF_SOURCE_TYPES = {"pdf"}` cố định); `PDFParser` đọc
  `settings.PDF_PARSER_BACKEND` cho escalation nội bộ `hybrid` vs `mineru`. Nhánh
  `markitdown` cho PDF **không còn được pipeline dùng** — `_MARKITDOWN_SOURCE_TYPES`
  chỉ chứa `"word"`.
- **Page markers `\x00P{N}\x00`** là ký tự null không in được, embed trong full text.
  `HybridChunker._resolve_page_numbers` strip chúng + gán `page_start/page_end` rồi
  **recompute `content_hash`**. Đừng dedupe/so hash trước khi qua chunker.
- **Order bắt buộc**: tag → contextualize → embed. Embed phải chạy SAU contextualize
  (vì embed input gồm `context_text`) và RAPTOR phải chạy SAU khi leaf chunks đã có
  `embedding` (RAPTOR cluster trên `chunk["embedding"]`).
- **Async ≠ searchable-only ngay.** Ở `mode=async`, text segments đã vào PG+ES (chat
  trả lời được) nhưng `graph_status="searchable"`; StructMem/KG extract chạy ở async
  tail (ARQ → `enriching` → `done`). RAPTOR + contextualize vẫn chạy **inline** ngay
  cả async (chúng nằm trên đường chính của `ingest_folder`, không phải trong worker).
- **Dedupe & skeleton**: `PostgresStore` so `content_hash` cấp document. Hash trùng +
  đã extract → `"skipped"` (bỏ qua cả ES re-index). Skeleton (`graph_status="pending"`,
  chưa có segment, tạo trước bởi `/sources` upload) hoặc doc `"failed"` → được
  populate/reset thay vì tạo mới. Hash khác → **xóa toàn bộ** doc cũ cùng `source_id`
  rồi re-ingest.
- **ES index auto-recreate khi đổi embedding dims.** `_recreate_index_if_dims_changed`
  **DROP** index nếu `EMBEDDING_MODEL` đổi dims (vd 1024→768). Đổi embedding model =
  mất index segments, phải re-ingest. CLIP `image_embedding` dims cố định
  `VISUAL_EMBEDDING_DIMS` (512), độc lập text dims.
- **StructMem index hợp nhất**: `entries_index_name` / `synthesis_index_name` chỉ là
  alias trỏ về `STRUCTMEM_INDEX` (`agentrag_memory_doc`); phân biệt entry/synthesis qua
  field `kind` lúc query, không phải index riêng.
- **Chunk < 80 chars bị drop** ở lớp search (loại heading-only như "## API"). Threshold
  này chỉ áp cho `chunks_search`, không áp `chunks_graph`.
- **`connectors/markdown.py` (`MarkdownConnector`) là legacy** — pipeline dùng
  `FolderConnector` (đa định dạng). Giữ lại cho back-compat.
