# Báo cáo dự án AgentRag

> Nền tảng RAG (Retrieval-Augmented Generation) cho học liệu y khoa Việt Nam.
> Cập nhật: 2026-05-22 · Branch hiện tại: `structmem` · Phiên bản adapter: `0.7.0`

---

## 1. Đối tượng người dùng

AgentRag được thiết kế để phục vụ ba nhóm chính trong hệ sinh thái y khoa và đào tạo y khoa Việt Nam.

### 1.1 Sinh viên y khoa & học viên sau đại học
- Người học cần một trợ lý đọc-hiểu tài liệu y khoa số lượng lớn (giáo trình, atlas giải phẫu, guideline, slide bài giảng) bằng tiếng Việt.
- Cần khả năng hỏi đáp dựa trên tài liệu đã ingest, có trích dẫn số trang chính xác (kiểu NotebookLM) để truy nguồn nhanh.
- Cần tạo **mindmap Mermaid** và **bản tóm tắt cấu trúc y khoa 9 mục** (Định nghĩa → Dịch tễ → Nguyên nhân → Sinh lý bệnh → Triệu chứng → Cận lâm sàng → Điều trị → Biến chứng → Tiên lượng) để ôn tập.

### 1.2 Bác sĩ lâm sàng và giảng viên
- Truy hồi nhanh thông tin theo **hệ cơ quan × chuyên khoa** (15 hệ × 14 chuyên khoa) từ kho tài liệu nội bộ của khoa/bệnh viện.
- Dùng `clinical summary` để chuẩn bị tài liệu phòng khám, hội chẩn.
- Inline `[Lĩnh vực]` filter trên giao diện chat cho phép giới hạn truy vấn theo domain cụ thể (ví dụ: chỉ tim mạch + nội khoa).

### 1.3 Nhà nghiên cứu & kỹ sư AI nội bộ
- Triển khai trên hạ tầng riêng (on-prem) với Ollama local hoặc cloud API (OpenAI/Gemini) — không bị phụ thuộc nhà cung cấp. 
- Có **MCP server** (`/mcp`) để tích hợp AgentRag như một tool vào Claude Desktop, Claude Code hoặc bất kỳ MCP client nào.
- Có **admin reasoning panel** (`/admin`) cho phép kiểm tra `tool_trace`, `timings_ms`, citations từng turn — phục vụ debug và đánh giá.
- Có **cost dashboard** (`/cost`) theo dõi LLM spend per-task / per-model với p50/p95 latency.

---

## 2. Mục đích

### 2.1 Mục tiêu cốt lõi
Xây dựng một nền tảng RAG **specialized** cho lĩnh vực y khoa Việt Nam, vượt lên trên các giải pháp RAG generic ở những điểm sau:

1. **Tiếng Việt chuẩn y khoa** — toàn bộ ontology, prompt, output, citation, tag đều bằng tiếng Việt; tránh việc dịch ngược qua tiếng Anh làm mất nghĩa thuật ngữ chuyên môn.
2. **Domain-aware retrieval** — không truy hồi flat trên toàn corpus mà partition theo `hệ cơ quan × chuyên khoa` để cải thiện precision và giảm context noise.
3. **Hai luồng suy luận song song** — semantic agent loop cho câu hỏi mở (`Tim hoạt động như thế nào?`) và structured SQL reasoning cho câu hỏi so sánh/thống kê/xếp hạng (`So sánh A và B`).
4. **Bộ nhớ phân cấp** — `StructMem` (doc memory) và `Chat StructMem` (conversation memory) với dual-perspective extraction (factual + relational) và cross-chunk/cross-turn consolidation, thay thế kiến trúc Graphiti + Neo4j cũ với chi phí thấp hơn (~$0.97 vs ~$1.28 / 100 chunks).
5. **Page-aware citation** — mọi trích dẫn từ PDF đều có `page_start` / `page_end` chính xác, phục vụ trang truy nguồn kiểu NotebookLM.
6. **Vision LLM cho ảnh y tế** — atlas giải phẫu, X-quang, sơ đồ trong PDF đều được mô tả bằng vision model (GPT-4o / Gemini / llava local).
7. **Transparent reasoning** — mỗi câu trả lời đều có thể mở `Trace` để xem đầy đủ pipeline `plan → decide → tool → assemble → answer → critique` với I/O từng tool.
8. **Self-hostable** — chạy hoàn toàn local với Ollama + Postgres + Elasticsearch + Valkey, không bắt buộc dùng cloud LLM.

### 2.2 Khác biệt so với các giải pháp khác

| Giải pháp | Vấn đề | AgentRag |
|---|---|---|
| ChatGPT / Gemini chat | Hallucinate; không trích dẫn được tài liệu nội bộ | Strict RAG, citation có hash + page |
| NotebookLM | Đóng kín; không tự host được; tiếng Việt yếu | Self-host, ontology tiếng Việt thuần |
| LangChain RAG generic | Không có domain partition; không có structured path | Domain router + 2 luồng song song |
| Graphiti + Neo4j | 4 sequential LLM call / chunk, infra phức tạp | StructMem 2 parallel call, ES-only |

---

## 3. Đã hoàn thành

> Khoảng thời gian phát triển: **2026-03-16 → nay**. Hơn 60 milestone commit. Dưới đây tóm tắt theo trục tính năng.

### 3.1 Hạ tầng nền tảng (3/2026 – 4/2026)
- **Storage layer 4 tầng**: PostgreSQL (source of truth + pgvector), Elasticsearch (hybrid search), Valkey (cache + ARQ queue + cost ledger stream), Filesystem (ảnh).
- **Docker Compose** với 6 tier presets (CPU-only đến GPU 24GB đến cloud API) — `make use-preset TIER=3a`.
- **Alembic migration** đầy đủ; `make install` là one-shot setup.
- **ARQ background workers + auto-scaler** dựa trên queue depth (1 → 4 workers).

### 3.2 Pipeline ingestion (3/2026 – 5/2026)
- **Parser đa định dạng**: PDF (PyMuPDF text-layer + tier escalation), DOCX/PPTX/HTML (MarkItDown), Excel/CSV (markdown | sql mode), Image (vision LLM), Markdown.
- **Page-aware chunking**: marker `\x00P{N}\x00` chèn vào text → `HybridChunker` gán `page_start`/`page_end` từng chunk.
- **Hai backend PDF**:
  - `hybrid` — PyMuPDF → Tesseract OCR → Vision LLM fallback (theo từng page khi text thin).
  - `mineru` — single-pass layout + OCR + formula→LaTeX + table→HTML (default `vlm-auto-engine` cho tiếng Việt, giữ dấu chuẩn xác).
- **PPTX → libreoffice → PDF → MinerU** (opt-in, giữ slide layout + công thức).
- **Vision LLM cho ảnh y tế** (OpenAI / Gemini / Ollama llava), prompt tune cho ngữ cảnh y khoa (identify image type, anatomical structures, labels).
- **Vision async mode** với RPM cap (free Gemini 10 RPM), per-image retry, flush-batch — text retrieval sẵn sàng ngay, ảnh lấp dần.

### 3.3 Hệ truy hồi (3/2026 – 5/2026)
- **Hybrid search**: BM25 + kNN dense + RRF fusion + StructMem (KG) entries — mode `hybrid_kg` mặc định.
- **Reranking** 2 backend: `llm_chat` (LLM as reranker) và `local_cross_encoder`.
- **Embedding cache** TTL 600s cho query path; ES result cache 60s.
- **Cap image-segment fraction** trong top_k (tránh ảnh chiếm chỗ text).
- **Filename hint resolution** — câu hỏi nhắc tên file → resolve về `document_title` tự động.

### 3.4 Bộ nhớ phân cấp — StructMem (4/2026)
- **Doc StructMem** thay thế Graphiti + Neo4j: per chunk chạy `factual_call` + `relational_call` song song → index vào `agentrag_memory_doc` (kind=entry).
- **Cross-chunk consolidation**: trigger tự động khi đạt threshold → embed → cosine search top-K seeds → LLM synthesis → index synthesis (kind=synthesis) → multi-hop reasoning.
- **Chat StructMem** (4/2026) — semantic conversation memory thay sliding-window; consolidate cross-turn theo cùng pattern.
- **Inject `conversation_memory`** vào `_decide()` + `_answer()` prompts.

### 3.5 Lý luận có cấu trúc — Structured SQL (4/2026)
- **5-bước pipeline**: Classify → Schema discovery → Extract (CLEAR A+B) → SQL compile → Synthesize.
- **3 query type**: `comparison`, `aggregation`, `ranking`.
- Fallback về semantic path nếu bất kỳ bước nào thất bại.

### 3.6 Agent harness (5/2026)
- **GraphAgentService** — LangGraph `StateGraph` với 13 nodes (validate → memory → chitchat_check → classify → structured/semantic → plan → bootstrap → decide ⇄ tool_exec → assemble → answer → ground), `InMemorySaver` checkpoint, `thread_id = conversation_id` cho resume.
- **Chit-chat fast-path**: greeting/thanks tokens skip retrieval → cheap routing model.
- **Plan-then-execute**: planner decomposes multi-hop → parallel retrieval → single answer pass.
- **Token-aware context budget** (`AGENT_MAX_CONTEXT_TOKENS=6000`) + **lost-in-the-middle reorder** (best chunks ở đầu + cuối context).
- **Long-answer unlock**: `AGENT_MAX_OUTPUT_TOKENS=131072`.

### 3.7 S-series milestones (5/2026)

| Mã | Tên | Trạng thái |
|---|---|---|
| **S1** | LLM cost & token dashboard (`/cost` page, per-task/per-model summary, p50/p95 latency, recent calls feed) | ✅ complete |
| **S2** | Per-turn LangGraph-style reasoning trace UI (Trace button mở dialog với pipeline graph + tool I/O + sub-queries + SQL) | ✅ complete |
| **S3** | Embedding cache + p50/p95 latency surface | ✅ complete |
| **S4** | Reasoning Plane / Execution Plane split + `ServiceContainer` DI + Protocol-based service contracts | ✅ complete |
| **S5** | Medical KB domain partition — 15 hệ × 14 chuyên khoa, shared ontology, `pg_trgm` fuzzy, `DomainRouter` SLM, `FederatedRetriever`, UI override dropdown trên ChatPanel | ✅ complete |
| **S6** | Per-user Activity panel + admin global feed | ✅ complete |
| **S7/D1** | Follow-up chips (Ask & Search StreamingResponse render `FollowupChips`) | ✅ complete |
| **S8** | Long-answer unlock (max output tokens) | ✅ complete |

### 3.8 LLM Routing & Generation (5/2026)
- **Per-task model map** — `classify/decide/domain_router/followup` → `llama3.2:3b`; `answer` → `qwen-agentrag` (finetuned) hoặc `qwen2.5:7b-instruct`; `mindmap/summary` → model trung.
- **Auto-fallback** khi model tag chính thiếu trên Ollama.
- **Large-context auto-route** — prompt > threshold → switch sang model context lớn (Gemini 2.5 pro 1M, qwen2.5:32b 128k).
- **Mindmap service** — Mermaid output, in-process TTL 24h cache.
- **Summary service** — 3 style (`study_note`, `clinical`, `quick_review`), iterate 9 medical sections song song.
- **Highlights** — 3-5 điểm quan trọng nhất kèm câu trả lời, `**bold**` term.

### 3.9 Giao diện người dùng (5/2026)
- **Vendor frontend** từ open-notebook (Next.js), tinh chỉnh `LoginForm` (signup + Google).
- **NotebookLM-style polish**: hover citation, inline images, follow-up chips.
- **Source chat vs Notebook chat** — source chat isolate theo `document_title`, tránh leak từ doc khác qua graph memory.
- **Domain filter dropdown** (S5) trên ChatPanel.
- **Trace dialog** (S2) trên mỗi AI bubble.
- **Cost dashboard** (S1) tại `/cost`.
- **Activity panel** (S6) cho user + global feed cho admin.
- **Thumbs-up/down** persist vào `adapter_chat_feedback` cho preference-pair dataset.
- **Original file download** — PDF inline view + XLSX/DOCX download.
- **i18n leak fix** + optimistic user bubble + graceful chat error.

### 3.10 Auth & bảo mật (5/2026)
- **JWT + Email/password signup** (bcrypt hash, TTL 7 ngày).
- **Google OAuth flow** (`/on/api/auth/google/*`).
- **Legacy bearer** (`OPEN_NOTEBOOK_PASSWORD`) vẫn tương thích backward.
- **Rate limit** per-user (120/min chat+search, 20/min upload) qua Valkey INCR + EXPIRE.
- **Upload dedupe** theo bytes hash — skip re-ingest.
- **`UPLOAD_MAX_BYTES=104857600`** (100 MB).
- **SecurityPolicy** filter theo `document_title` + `section_path` prefix/pattern.
- **Self-heal user row** khi JWT survive `make reset-data`.
- **ADMIN_EMAILS** auto-promotion.

### 3.11 Tooling & DX
- **CLI** Typer + Rich, persistent state `~/.agentrag/state.json`, inline commands `/new` `/switch` `/list` `/clear`.
- **MCP server** (FastMCP) — tools `search` và `structured_query` cho Claude Desktop / Claude Code.
- **Makefile** unified: `install`, `dev`, `up-bg`, `logs`, `stop`, `reset` (3 mức: soft / data / nuke), `seed-ontology`, `backfill-tags`, `ollama-pull`, `vision-pull`, `convert-llm`, `reseed-models`, `health`, `test-fast`, `bench-ingest`.
- **Benchmark scripts**: `benchmark_ingest.py`, `benchmark_retrieval.py`, `benchmark_agent.py`.
- **Finetune strategy** doc cho `qwen-agentrag` + `agentrag-embed-v1`.
- **System library install script** (2026-05-19).

### 3.12 Refactor & cleanup (5/2026)
- **R1-R7 cleanup** (2026-05-20) — agent/storage/cache cleanup, 24 files, +745 / -711 LOC.
- **Redis label → Valkey** trên toàn dự án (giao thức RESP tương thích).
- **README + ARCHITECTURE.md** refresh đầy đủ cho S1–S5 + plane split.

---

## 4. Hướng phát triển trong tương lai

### 4.1 Ngắn hạn (Q3/2026)

1. **Self-critique node** — port lại critique pass từ hand-rolled loop cũ thành node trong LangGraph StateGraph (hiện ARCHITECTURE.md ghi nhận "not currently ported").
2. **Preference learning loop** — dùng `adapter_chat_feedback` (thumbs-up/down) đã thu thập để tạo preference-pair dataset, finetune model qua DPO/ORPO.
3. **Evaluation harness chuẩn hoá** — benchmark dataset y khoa Việt Nam (multi-hop QA, structured comparison, image-grounded), CI tracking accuracy regression.
4. **Mở rộng ontology** — hiện 59 canonical term (custom_terms.yaml) + ICD10 VN; cần seed thêm chuyên khoa Nhi, Nội tiết, Da liễu, Tâm thần.
5. **Streaming SQL reasoning** — hiện structured path không stream; cần stream SQL compile + result rendering để giảm perceived latency.
6. **Citation hover preview** — preview đoạn trích trực tiếp khi hover citation (hiện chỉ link).

### 4.2 Trung hạn (Q4/2026)

1. **Multi-modal mở rộng** — video lecture (Whisper transcribe + diarization) và audio podcast y khoa.
2. **Finetuned medical embedding** (`agentrag-embed-v1`) — train trên corpus y khoa VN để cải thiện retrieval (hiện đang dùng `nomic-embed-text` hoặc tương đương).
3. **Knowledge graph visualization** — hiển thị graph các synthesis entries của một document (giải phẫu → bệnh lý → điều trị) bằng D3/Cytoscape.
4. **Adaptive consolidation threshold** — hiện hardcode `STRUCTMEM_CONSOLIDATION_THRESHOLD=20`; cần điều chỉnh động theo entropy của entries.
5. **Multi-tenant isolation** — workspace-level data isolation (mỗi khoa/bệnh viện một workspace), share ontology cấp tổ chức.
6. **Mobile-friendly UI** — chat panel responsive, citation drawer cho màn nhỏ.
7. **Voice input/output** — TTS tiếng Việt cho câu trả lời, STT cho câu hỏi (sinh viên đọc câu hỏi rảnh tay khi học atlas).

### 4.3 Dài hạn (2027+)

1. **Federated retrieval cross-institution** — nhiều bệnh viện share retrieval qua privacy-preserving protocol (không share raw chunk, share embedding + entries).
2. **Clinical decision support tích hợp** — kết nối với guideline + drug interaction DB (e.g. Vidal VN, Bộ Y tế clinical pathway).
3. **Patient case-based learning** — student mode: ingest case file → agent đóng vai bệnh nhân hoặc giáo viên Socratic.
4. **Agentic experiment runner** — agent tự lập kế hoạch literature review (search → screen → extract → tổng hợp) cho nghiên cứu sinh.
5. **On-device deployment** (laptop / mini-PC) cho bệnh viện tuyến huyện với 8GB VRAM, optimize qua quantization + speculative decoding.
6. **Compliance & audit log** — chuẩn HIPAA-like cho dữ liệu bệnh nhân (nếu ingest case file).

### 4.4 Rủi ro & ưu tiên xử lý

| Rủi ro | Khả năng | Giảm thiểu |
|---|---|---|
| Hallucinate trong câu trả lời y khoa (nguy hiểm cho lâm sàng) | Trung bình | Strict grounding (đã có `ground` node), critique node (TBD), highlight chỉ dùng kèm citation |
| LLM cost vượt budget | Trung bình | Cost dashboard (S1) đã có; cần alert + budget cap automate |
| Domain router phân loại sai → trả về kết quả sai chuyên khoa | Trung bình | Confidence threshold ≥ 0.7, fallback top-K khi mơ hồ, UI override |
| MinerU model download chậm / không stable trên hạ tầng yếu | Thấp-Trung | Fallback `hybrid` backend, cache models |
| Ontology không bao phủ hết thuật ngữ chuyên ngành hẹp | Cao | Section-primary tagging + content fallback đã có; cần workflow đóng góp ontology |

---

## 5. Tham chiếu nội bộ

- Kiến trúc: [`ARCHITECTURE.md`](./ARCHITECTURE.md) — chi tiết Reasoning/Execution Plane split.
- README chính: [`README.md`](./README.md) — full 1540 dòng cấu hình, API, CLI.
- Module READMEs: `src/agentrag/{agent,services,ontology,structured,retrieval,ingestion,graph,chat,generation,adapter,worker,cli,mcp,common}/README.md`.
- Finetune strategy: [`docs/FINETUNE_STRATEGY.md`](./docs/FINETUNE_STRATEGY.md).
- Ontology seed: `data/ontology/custom_terms.yaml` (59 canonical term) + ICD-10 VN CSV.

---

*Báo cáo lập ngày 2026-05-22 bởi nhóm phát triển AgentRag.*
