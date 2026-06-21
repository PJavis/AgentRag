# Sơ đồ kiến trúc VITAL (Mermaid)

> Dán trực tiếp vào báo cáo `.md` (GitHub/VSCode render tự động), hoặc export PNG/SVG để chèn vào
> docx/slide. Cách render & export ở cuối file.

---

## Hình 1 — Kiến trúc tổng thể

```mermaid
flowchart TB
  U["👩‍⚕️ Người dùng"] --> FE["Frontend — Next.js<br/>chat · hover trích dẫn · Trace · /cost"]
  FE -->|"HTTP / streaming"| BE["Backend API — FastAPI<br/>xác thực JWT · giới hạn tần suất · lọc bảo mật"]

  subgraph RP["REASONING PLANE — quyết định LÀM GÌ"]
    AG["Agent — LangGraph 13 node<br/>plan → retrieve → answer → critique → ground"]
  end

  subgraph EP["EXECUTION PLANE — thực thi IO"]
    RET["Retrieval<br/>BM25 + kNN + RRF + rerank<br/>DomainRouter 15×14"]
    ING["Ingestion<br/>parse · chunk(+trang) · embed"]
    GEN["Generation<br/>mindmap · tóm tắt 9 mục"]
    SM["StructMem<br/>thực thể + quan hệ"]
    VIS["Vision LLM<br/>mô tả ảnh y tế"]
    LLM["LLM Gateway<br/>Ollama · DeepSeek · Gemini"]
  end

  BE --> AG & ING & GEN
  AG --> RET & SM & LLM
  RET --> SM
  ING --> SM & VIS

  subgraph ST["LƯU TRỮ"]
    PG[("PostgreSQL<br/>+ pgvector")]
    ES[("Elasticsearch<br/>hybrid + StructMem")]
    VK[("Valkey<br/>cache · queue · cost")]
    FS[("Filesystem<br/>ảnh")]
  end

  ING --> PG & ES & FS
  RET --> ES
  SM --> ES
  GEN --> ES
  BE --> VK
  AG -. "cost / trace" .-> OBS["Observability<br/>/cost · Trace"]
  AG -->|"câu trả lời + trích dẫn [n]"| BE
```

---

## Hình 2 — Luồng nạp tài liệu (offline, một lần khi upload)

```mermaid
flowchart LR
  F["File<br/>PDF/DOCX/PPTX/<br/>Excel/HTML/ảnh"] --> P["Bóc tách<br/>PyMuPDF · OCR · MinerU · MarkItDown"]
  P --> C["Cắt đoạn<br/>(gắn page_start/end)"]
  C --> E["Nhúng vector<br/>bge-m3"]
  E --> PG[("PostgreSQL")]
  E --> ES[("Elasticsearch")]
  C -. "worker nền" .-> SM["StructMem<br/>thực thể + quan hệ"] --> ES
  P -. "worker nền" .-> V["Vision LLM<br/>mô tả ảnh"] --> FS[("Filesystem")]
```

---

## Hình 3 — Luồng trả lời câu hỏi (Agent 13 node)

```mermaid
flowchart TB
  Q["Câu hỏi"] --> V["validate<br/>(an ninh)"] --> M["memory<br/>(bộ nhớ hội thoại)"] --> CC{"chitchat?"}
  CC -->|"có"| CA["chitchat_answer"] --> EN(["Trả lời + trích dẫn"])
  CC -->|"không"| SP["semantic_plan<br/>(tách câu hỏi con)"] --> B["bootstrap<br/>(truy hồi hybrid_kg + rerank)"]
  B --> D{"decide<br/>đủ ngữ cảnh chưa?"}
  D -->|"cần thêm"| T["tool_exec<br/>(tìm tiếp)"] --> D
  D -->|"đủ"| AS["assemble<br/>(ghép ngữ cảnh)"] --> AN["answer<br/>(viết câu trả lời + [n])"]
  AN --> CR{"critique<br/>(CRAG — tắt mặc định)"}
  CR -->|"chưa chắc"| COR["corrective_retrieve<br/>(tìm lại)"] --> CR
  CR -->|"đạt"| G["ground<br/>(gắn trích dẫn / từ chối nếu thiếu căn cứ)"] --> EN
```

---

## Cách render & export

**Xem nhanh / sửa:**
- Web: dán vào https://mermaid.live → sửa trực quan, bấm **Export PNG/SVG**.
- VSCode: cài extension *"Markdown Preview Mermaid Support"* → mở preview (Ctrl+Shift+V).
- GitHub: tự render khi xem file `.md` trong repo.

**Export ra ảnh để chèn docx/slide (dòng lệnh):**
```bash
# cài 1 lần
npm i -g @mermaid-js/mermaid-cli
# tách từng hình ra file .mmd rồi:
mmdc -i hinh1.mmd -o hinh1.svg     # SVG: nét sắc, phóng to không vỡ (khuyên dùng)
mmdc -i hinh1.mmd -o hinh1.png -s 3   # PNG độ phân giải cao (scale 3x)
```

**Nếu muốn hình "đẹp tay" hơn** (báo cáo in / bảo vệ): vẽ lại bằng **Excalidraw**
(excalidraw.com — phong cách phác thảo) hoặc **draw.io** (app.diagrams.net) — GUI kéo–thả, export
PNG/SVG. Đánh đổi: không sửa bằng text, khó version-control như Mermaid.
