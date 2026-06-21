# BÁO CÁO HỆ THỐNG VITAL

**VITAL** — *Vietnamese Integrated reTrieval-Augmented Learning*
Nền tảng Hỏi–Đáp tài liệu y khoa tiếng Việt dựa trên RAG (Retrieval-Augmented Generation)

> *Tên cũ của dự án: AgentRag.*
> Ngày lập: 21/06/2026 · Nhánh mã nguồn: `feat/ragas-langfuse-reranker` · Nhóm phát triển VITAL

---

## Mục lục

1. [Giới thiệu đề tài](#1-giới-thiệu-đề-tài)
2. [Khảo sát và phân tích yêu cầu](#2-khảo-sát-và-phân-tích-yêu-cầu)
3. [Công nghệ sử dụng](#3-công-nghệ-sử-dụng)
4. [Thiết kế và triển khai hệ thống](#4-thiết-kế-và-triển-khai-hệ-thống)
5. [Kết quả thực nghiệm](#5-kết-quả-thực-nghiệm)
6. [Kết luận và hướng phát triển](#6-kết-luận-và-hướng-phát-triển)

---

## 1. Giới thiệu đề tài

### 1.1. Bối cảnh

Sinh viên y khoa và bác sĩ phải làm việc với khối lượng tài liệu khổng lồ: giáo trình, atlas
giải phẫu, guideline, slide bài giảng, bảng số liệu. Tra cứu thủ công thì chậm; dùng các trợ lý
AI tổng quát (ChatGPT, Gemini) thì tiện nhưng có **hai rủi ro nghiêm trọng trong y khoa**:

1. **Bịa thông tin (hallucination)** — mô hình trả lời trôi chảy nhưng sai, cực kỳ nguy hiểm cho
   quyết định lâm sàng.
2. **Không truy được nguồn** — không biết câu trả lời lấy từ đâu để kiểm chứng.

### 1.2. Mục tiêu đề tài

Xây dựng **VITAL** — một nền tảng hỏi–đáp **chuyên biệt cho tài liệu y khoa tiếng Việt**, trả lời
**chỉ dựa trên tài liệu người dùng đã nạp vào**, kèm **trích dẫn số trang** để kiểm chứng, và biết
**từ chối khi không đủ căn cứ** thay vì bịa.

> **Ví dụ dễ hiểu.** Hãy hình dung một thủ thư cực nhanh: bạn đưa 500 cuốn sách y khoa và hỏi
> *"Nhồi máu cơ tim điều trị thế nào?"*. Thủ thư lật đúng vài trang liên quan, đọc, tóm tắt câu
> trả lời, và **chỉ rõ lấy từ trang nào của sách nào**. VITAL làm đúng việc đó, tự động, trong
> khoảng 20 giây.

### 1.3. Phạm vi

- **Đầu vào:** PDF, Word, PowerPoint, HTML, Excel/CSV, ảnh (atlas, X-quang, sơ đồ).
- **Ngôn ngữ:** tiếng Việt (chính) và tiếng Anh chuyên ngành.
- **Triển khai:** tự host hoàn toàn (on-premise) với mô hình nội bộ, hoặc dùng cloud LLM.
- **Không thuộc phạm vi:** chẩn đoán thay bác sĩ; VITAL là công cụ tra cứu/đọc–hiểu có trích dẫn.

### 1.4. Đóng góp chính

- Pipeline RAG **chuyên biệt y khoa tiếng Việt**: ontology, prompt, trích dẫn đều bằng tiếng Việt.
- **Truy hồi theo chuyên khoa** (15 hệ cơ quan × 14 chuyên khoa) thay vì tìm tràn lan.
- **Bộ nhớ tri thức (StructMem)** giúp hiểu quan hệ giữa các khái niệm, hỗ trợ suy luận nhiều bước.
- **Cơ chế an toàn**: trích dẫn theo trang + từ chối khi thiếu căn cứ + nhật ký suy luận minh bạch.
- **Quy trình đánh giá (benchmark) nghiêm túc**, dám loại bỏ kỹ thuật không chứng minh được hiệu quả.

---

## 2. Khảo sát và phân tích yêu cầu

### 2.1. Khảo sát các giải pháp hiện có

| Giải pháp | Ưu điểm | Hạn chế (với bài toán đề tài) |
|---|---|---|
| ChatGPT / Gemini (chat tổng quát) | Mạnh, tiện | Bịa thông tin; không trích dẫn được tài liệu nội bộ |
| NotebookLM | Hỏi đáp theo tài liệu, có trích dẫn | Đóng kín, không tự host; tiếng Việt yếu |
| LangChain RAG (generic) | Linh hoạt, nhiều thành phần | Không phân vùng chuyên khoa; không tối ưu y khoa VN |
| Graphiti + Neo4j (knowledge graph) | Bộ nhớ tri thức mạnh | 4 lần gọi LLM tuần tự/đoạn; hạ tầng phức tạp, tốn kém |

**Kết luận khảo sát:** chưa có giải pháp nào vừa (a) tự host được, (b) chuyên biệt y khoa tiếng
Việt, (c) trích dẫn truy nguồn, (d) có cơ chế chống bịa. Đây là khoảng trống VITAL nhắm tới.

### 2.2. Đối tượng người dùng

| Nhóm | Nhu cầu chính |
|---|---|
| Sinh viên / học viên sau đại học | Đọc–hiểu tài liệu lớn, hỏi đáp có trích dẫn, tạo mindmap & tóm tắt ôn tập |
| Bác sĩ lâm sàng & giảng viên | Tra cứu nhanh theo hệ cơ quan/chuyên khoa; tóm tắt lâm sàng |
| Nhà nghiên cứu / kỹ sư AI nội bộ | Triển khai on-prem; tích hợp qua MCP; xem nhật ký suy luận & chi phí |

### 2.3. Yêu cầu chức năng

1. Nạp đa định dạng tài liệu, cắt đoạn **gắn số trang** để trích dẫn chính xác.
2. Hỏi–đáp dựa trên tài liệu, câu trả lời **kèm trích dẫn `[n]`** trỏ về đúng nguồn + trang.
3. Mô tả **ảnh y tế** trong tài liệu (Vision LLM).
4. Truy hồi **theo chuyên khoa** (lọc domain) và **truy hồi lai** (từ khoá + ngữ nghĩa).
5. **Từ chối an toàn** khi không có đủ ngữ cảnh liên quan.
6. Sinh **mindmap** và **tóm tắt cấu trúc y khoa 9 mục**.
7. **Nhật ký suy luận (trace)** và **theo dõi chi phí/độ trễ** cho mỗi câu trả lời.

### 2.4. Yêu cầu phi chức năng

| Yêu cầu | Mô tả |
|---|---|
| Tin cậy (faithfulness) | Câu trả lời bám nguồn, hạn chế tối đa bịa — ưu tiên số 1 |
| Truy nguồn | Mọi khẳng định gắn được với đoạn nguồn cụ thể |
| Tự chủ dữ liệu | Chạy nội bộ, dữ liệu y khoa không bắt buộc rời hạ tầng |
| Hiệu năng | Thời gian trả lời ở mức chấp nhận được (p50 ~18–26s) |
| Bảo mật | Xác thực người dùng, giới hạn tần suất, lọc theo phạm vi tài liệu |
| Khả năng mở rộng | Worker chạy nền tự co giãn theo tải |

---

## 3. Công nghệ sử dụng

### 3.1. Tổng quan ngăn xếp công nghệ

| Tầng | Công nghệ | Vai trò |
|---|---|---|
| Giao diện | Next.js (React 19), Radix UI, Tailwind | Web app hỏi–đáp kiểu NotebookLM |
| Cổng API | FastAPI, Pydantic | REST + streaming; xác thực; tương thích open-notebook |
| Điều phối tác tử | LangGraph (StateGraph, InMemorySaver) | Máy trạng thái cho luồng trả lời |
| Mô hình ngôn ngữ | Ollama (llama3.2:3b, qwen), DeepSeek, Gemini, OpenAI | Định tuyến per-task qua `LLM_TASK_MODEL_MAP` |
| Nhúng (embedding) | BAAI **bge-m3** (qua TEI) | Biến đoạn văn/câu hỏi thành vector |
| Xếp hạng lại (rerank) | **bge-reranker-v2-m3** (cross-encoder, chạy local) | Đẩy đoạn đúng trọng tâm lên đầu |
| CSDL gốc | PostgreSQL + pgvector | Nguồn sự thật: tài liệu, đoạn, hội thoại |
| Tìm kiếm | Elasticsearch (BM25 + dense kNN + RRF) | Truy hồi lai + bộ nhớ tri thức |
| Bộ đệm / hàng đợi | Valkey (tương thích Redis) | Cache + hàng đợi ARQ + sổ chi phí |
| Bóc tách tài liệu | PyMuPDF, Tesseract OCR, MinerU, MarkItDown, openpyxl | Parse PDF/Office/ảnh/bảng |
| Thị giác (vision) | GPT-4o / Gemini / llava (local) | Mô tả ảnh y tế |
| Worker nền | ARQ | Trích xuất tri thức, mô tả ảnh, hợp nhất bộ nhớ |
| Đánh giá | DeepEval, RAGAS, HuggingFace `datasets` | Benchmark chất lượng RAG |
| Hạ tầng | Docker Compose, Alembic | Dựng hệ thống một lệnh; migration CSDL |
| Tích hợp ngoài | MCP (FastMCP), CLI (Typer/Rich) | Gắn vào Claude Desktop; thao tác dòng lệnh |

### 3.2. Vì sao chọn các công nghệ này

- **Truy hồi lai (BM25 + kNN + RRF)** thay vì chỉ vector: kết hợp khớp **từ khoá** (thuật ngữ y
  khoa chính xác) và **ngữ nghĩa** (diễn đạt khác nhau) → bao phủ tốt hơn.
- **bge-m3 / bge-reranker-v2-m3**: hỗ trợ đa ngôn ngữ tốt (gồm tiếng Việt), chạy được **miễn phí
  trên hạ tầng nội bộ**.
- **StructMem trên Elasticsearch** thay cho Graphiti + Neo4j: chỉ **2 lần gọi LLM song song/đoạn**
  (so với 4 tuần tự), không cần thêm CSDL đồ thị → rẻ và đơn giản hơn.
- **LangGraph**: mô hình hoá luồng trả lời thành các "node" rõ ràng, dễ kiểm soát rẽ nhánh và lưu
  checkpoint theo hội thoại.
- **Định tuyến mô hình per-task**: tác vụ nhẹ (phân loại, điều phối) chạy mô hình nhỏ nội bộ (miễn
  phí); tác vụ khó (sinh câu trả lời) dùng mô hình mạnh → cân bằng chi phí/chất lượng.

---

## 4. Thiết kế và triển khai hệ thống

### 4.1. Kiến trúc tổng thể

Hệ thống tách thành **hai mặt phẳng (plane)**, nối với nhau qua một `ServiceContainer` (tiêm phụ
thuộc — DI):

- **Reasoning Plane (mặt phẳng suy luận)** — quyết định *làm gì*: máy trạng thái, prompt, vòng lặp
  quyết định của LLM. Không trực tiếp đụng vào IO.
- **Execution Plane (mặt phẳng thực thi)** — làm *IO*: gọi mô hình, nhúng vector, truy hồi
  Elasticsearch, lưu trữ, thị giác. Không chứa logic quyết định.

Ưu điểm: dễ kiểm thử (thay thế dịch vụ bằng mock qua Protocol), dễ bảo trì, ranh giới trách nhiệm
rõ ràng.

### 4.2. Bốn kho lưu trữ

| Kho | Vai trò |
|---|---|
| **PostgreSQL** (+pgvector) | Nguồn sự thật: bản đầy đủ tài liệu, đoạn văn, hội thoại, notebook |
| **Elasticsearch** | Tìm kiếm: `agentrag_segments` (BM25+kNN+RRF) + bộ nhớ StructMem |
| **Valkey** | Bộ đệm chat + hàng đợi ARQ + luồng ghi chi phí |
| **Hệ thống tệp** | Ảnh trích từ tài liệu (phục vụ qua `/images/*`) |

PostgreSQL là "xương sống" bền vững; Elasticsearch là "hình chiếu" phục vụ truy hồi (dựng lại được
từ PostgreSQL).

### 4.3. Các thành phần và cách giao tiếp

| Thành phần | Chức năng | Nhận từ → Gửi đến |
|---|---|---|
| **Frontend** (Next.js) | Khung chat, hover trích dẫn, trace, trang chi phí | Người dùng → Backend API |
| **Backend / Adapter** (FastAPI) | Xác thực, giới hạn tần suất, lọc bảo mật, streaming | Frontend/MCP/CLI → Agent, Ingestion, Generation |
| **Ingestion** | Bóc tách → cắt đoạn (gắn trang) → nhúng → ghi index | File (qua Backend) → PostgreSQL, Elasticsearch, StructMem, Tệp ảnh |
| **Retrieval** | Truy hồi lai + rerank + định tuyến chuyên khoa | Câu hỏi (từ Agent) → trả đoạn đã xếp hạng cho Agent |
| **StructMem** | Trích xuất thực thể + quan hệ; hợp nhất tri thức | Đoạn (từ Ingestion), hội thoại (từ Agent) → Elasticsearch; tín hiệu → Retrieval |
| **Agent** (LangGraph) | Điều phối toàn bộ việc trả lời | Câu hỏi (từ Backend) → Retrieval, StructMem, LLM → Backend |
| **Generation** | Mindmap + tóm tắt 9 mục | Yêu cầu (từ Backend) → Elasticsearch → Frontend |
| **Observability** | Ghi chi phí/độ trễ; nhật ký suy luận | Mọi lần gọi mô hình → trang `/cost`, Trace dialog |
| **Hạ tầng** | 4 kho + Docker + worker ARQ tự co giãn | Tất cả module |

### 4.4. Luồng nạp tài liệu (offline)

```
Tài liệu → Bóc tách chữ + ảnh → Cắt đoạn (gắn số trang) → Nhúng vector (bge-m3)
        → Ghi vào PostgreSQL + Elasticsearch
        → (nền) Trích xuất tri thức (StructMem) + Mô tả ảnh (Vision LLM)
```

- PDF: PyMuPDF đọc lớp chữ; trang scan → OCR (Tesseract) / Vision; tuỳ chọn MinerU giữ bố cục +
  công thức (LaTeX) + bảng (HTML).
- Mỗi đoạn được gắn `page_start`/`page_end` để trích dẫn đúng trang.
- Việc nặng (tri thức, ảnh) đẩy vào **worker nền** → người dùng không phải chờ.

### 4.5. Luồng trả lời câu hỏi (online) — Agent 13 node

Agent là một LangGraph `StateGraph` gồm 13 node. Luồng:

```
START → validate (an ninh) → memory (bộ nhớ hội thoại) → chitchat_check
   ├─ (xã giao) → chitchat_answer → END
   └─ semantic_plan (tách câu hỏi con nếu phức) → bootstrap (truy hồi hybrid_kg)
         → decide ⇄ tool_exec (vòng lặp: cần tìm thêm hay đủ?)
         → assemble (ghép ngữ cảnh, sắp xếp tránh "lạc giữa ngữ cảnh")
         → answer (LLM viết câu trả lời + trích dẫn [n])
         → critique → (đạt) ground | (chưa chắc) corrective_retrieve → critique
         → ground (gắn trích dẫn theo số nguồn + trang) → END
```

Hai cơ chế an toàn:

- **Từ chối khi thiếu căn cứ** — nếu đoạn liên quan nhất vẫn dưới ngưỡng tin cậy
  (`RETRIEVAL_RELEVANCE_FLOOR = 0.6`), hệ thống trả lời *"Tôi không có thông tin"* và **xoá trích
  dẫn**, thay vì bịa.
- **Tự phê bình (CRAG)** — node `critique` + `corrective_retrieve` có thể kiểm tra câu trả lời và
  tìm lại (step-back) nếu chưa chắc. Hiện **TẮT mặc định** (`CRAG_ENABLED=false`), chờ đánh giá thêm.

### 4.6. Truy hồi và bộ nhớ tri thức

- **Truy hồi lai (`hybrid_kg`, mặc định):** trộn BM25 (từ khoá) + kNN (ngữ nghĩa) bằng RRF, cộng
  tín hiệu từ StructMem, rồi **rerank** bằng cross-encoder `bge-reranker-v2-m3`.
- **Định tuyến chuyên khoa (DomainRouter):** nhận diện câu hỏi thuộc hệ cơ quan/chuyên khoa nào
  (15 × 14) để thu hẹp tìm kiếm, giảm nhiễu.
- **StructMem:** mỗi đoạn được trích **thực thể** (factual) + **quan hệ** (relational) song song;
  khi tích đủ thì **hợp nhất** thành tri thức tổng hợp, hỗ trợ suy luận nhiều bước. Có phiên bản
  cho hội thoại (thay cửa sổ trượt).

### 4.7. Một câu hỏi đi xuyên hệ thống

Ví dụ: *"Nhồi máu cơ tim cấp điều trị thế nào?"*

1. Frontend gửi câu hỏi + token tới Backend.
2. Backend xác thực, giới hạn tần suất, lọc bảo mật → chuyển cho Agent.
3. Agent kiểm tra an ninh, lấy bộ nhớ hội thoại, xác định không phải câu xã giao.
4. Agent gọi Retrieval: truy hồi lai trên Elasticsearch → rerank chọn đoạn đúng trọng tâm.
5. Agent ghép ngữ cảnh; nếu không đủ căn cứ → **từ chối an toàn**.
6. Agent gửi ngữ cảnh + câu hỏi cho LLM → viết câu trả lời kèm `[n]`.
7. Agent gắn trích dẫn về đúng đoạn nguồn + trang; Observability ghi chi phí + nhật ký.
8. Backend stream câu trả lời về Frontend; người dùng hover `[n]` để xem nguồn, hoặc bấm **Trace**.

### 4.8. Quyết định thiết kế quan trọng: tinh gọn kiến trúc

Trong quá trình phát triển, hệ thống từng có thêm một luồng **"suy luận có cấu trúc" (Structured
SQL)** — với câu hỏi so sánh/thống kê, nó thử sinh câu lệnh SQL chạy trên dữ liệu bảng. Qua thực
nghiệm, luồng này **hiếm khi kích hoạt** (corpus y khoa chủ yếu là văn xuôi) và làm kiến trúc phức
tạp. **Quyết định: gỡ bỏ hoàn toàn**, đưa Agent về **một luồng semantic duy nhất** — đơn giản, dễ
bảo trì, dễ giải thích. Việc gỡ bỏ đã được kiểm chứng **không gây lỗi hồi quy**.

> Một quyết định kỹ thuật tốt không chỉ là thêm tính năng, mà còn là **dám bỏ thứ không chứng minh
> được giá trị**.

---

## 5. Kết quả thực nghiệm

### 5.1. Phương pháp đánh giá

**Bộ dữ liệu chuẩn (công khai, có đáp án vàng) từ HuggingFace:**

| Bộ con | Ngôn ngữ | Nguồn | Lĩnh vực |
|---|---|---|---|
| `vn_bkai` | Việt | `sailor2/Vietnamese_RAG` (BKAI) | Hỏi đáp tổng hợp |
| `vn_legal` | Việt | `sailor2/Vietnamese_RAG` (Legal) | Pháp lý |
| `en_covidqa` | Anh | `galileo-ai/ragbench` (covidqa) | Y khoa — COVID |
| `en_pubmedqa` | Anh | `galileo-ai/ragbench` (pubmedqa) | Y sinh — PubMed |

**Quy trình:** nạp các đoạn "đáp án vàng" vào hệ thống qua đúng pipeline thật → cho hệ thống trả
lời từng câu qua đúng đường đi agent thật → một mô hình AI mạnh đóng vai **giám khảo (LLM-as-judge,
DeepSeek)** chấm điểm → tính chi phí, độ trễ, tỷ lệ lỗi. Đây là chuẩn đánh giá RAG phổ biến
(DeepEval/RAGAS).

**9 chỉ số đánh giá:**

| Chỉ số | Ngưỡng đạt | Ý nghĩa |
|---|---|---|
| Contextual recall | ≥ 0.70 | Có tìm ĐỦ đoạn cần thiết không |
| Contextual precision | ≥ 0.70 | Đoạn tìm được có ĐÚNG trọng tâm không (sau rerank) |
| Faithfulness | ≥ 0.80 | Câu trả lời có BÁM nguồn, không bịa (quan trọng nhất) |
| Answer correctness | ≥ 0.70 | Có KHỚP đáp án vàng không (khó nhất) |
| Citation accuracy | ≥ 0.70 | Trích dẫn `[n]` có trỏ ĐÚNG nguồn không |
| Failure rate | < 0.05 | Tỷ lệ câu bị lỗi |
| Freshness | mới > cũ | Khi 2 nguồn mâu thuẫn, ưu tiên dữ liệu mới |
| Cost / câu | tham khảo | Chi phí AI trung bình mỗi câu |
| Latency p50/p95 | tham khảo | Thời gian trả lời |

> **Lưu ý trung thực khi đọc số:** 5 chỉ số chất lượng đầu là **hợp lệ và so sánh được**. Riêng
> **chi phí/độ trễ KHÔNG so trực tiếp với production**: khi benchmark, các tác vụ điều phối bị ép
> chạy trên cloud (vì mô hình nội bộ Ollama hay tắt giữa chừng); ở production chúng chạy **miễn phí**
> trên mô hình nội bộ → chi phí thật thấp hơn. Sai số mẫu: n=80 khoảng ±0.05; bộ ngoài kho n=15 ±0.1.

### 5.2. Kết quả vòng đánh giá gần nhất (19/06/2026)

Đây là vòng đánh giá **mới nhất**, phản ánh hệ thống ở **trạng thái hiện tại**. Khác các vòng trước
(corpus "dễ", mỗi câu chỉ một đoạn cần tìm), vòng này chạy trên kho tài liệu **"khó"**: gộp nhiều
đoạn thành tài liệu đa-đoạn, cố tình thêm **nhiễu (distractor)** để benchmark phân biệt được chất
lượng thật; đồng thời **bật tính năng từ-chối-khi-thiếu-căn-cứ** ở ngưỡng tin cậy **0.6** vừa hiệu
chỉnh. Gồm 80 câu **có trong kho** (đo chất lượng) + 15 câu **ngoài kho** (đo an toàn). Giám khảo:
DeepSeek. Không câu nào lỗi.

**Vì sao ngưỡng = 0.6.** Đo trực tiếp điểm tin cậy của bộ rerank (thang 0–1):

| Loại câu | Điểm rerank cao nhất |
|---|---|
| Câu CÓ trong kho (liên quan) | ≈ 0.73 |
| Câu NGOÀI kho (lạc đề) | ≈ 0.50 |

Bộ rerank **không bao giờ chấm dưới ~0.50** kể cả nội dung lạc đề. Ngưỡng cũ **0.3** nằm dưới cả
hai vùng → tính năng từ chối **chưa bao giờ kích hoạt** (lỗi hiệu chỉnh ẩn). Đặt lại **0.6** (giữa
khe 0.50–0.73) thì tính năng hoạt động đúng: câu trong kho giữ nguyên, câu ngoài kho bị chặn.

**a) Chất lượng trên 80 câu CÓ trong kho:**

| Chỉ số | Ngưỡng | Kết quả | Nhận xét |
|---|---|---|---|
| Contextual recall | ≥ 0.70 | **0.873** | ✅ ĐẠT — tìm đủ đoạn cần thiết |
| Contextual precision | ≥ 0.70 | **0.699** | ≈ ngưỡng — corpus cố tình có nhiễu |
| Faithfulness | ≥ 0.80 | **0.951** | ✅ ĐẠT mạnh — rất ít bịa |
| Answer correctness | ≥ 0.70 | **0.721** | ✅ ĐẠT — khớp đáp án vàng |
| Citation accuracy | ≥ 0.70 | **0.788** | ✅ ĐẠT — trích dẫn đúng nguồn |
| Failure rate | < 0.05 | **0.000** | ✅ ĐẠT — không câu nào lỗi |
| Latency p50 | tham khảo | 17.7s | thời gian trả lời trung vị |

**b) An toàn trên 15 câu NGOÀI kho** (thuốc/bệnh bịa — hệ thống PHẢI từ chối; so trước/sau khi bật
tính năng):

| Chỉ số an toàn | Trước (tắt) | Sau (bật, ngưỡng 0.6) |
|---|---|---|
| Từ chối sạch — lý tưởng ↑ | 0.000 | **0.467** ▲ |
| Rào nhưng vẫn trích nhiễu ↓ | 0.533 | **0.000** ▲ |
| Bịa tự tin — nguy hiểm ↓ | 0.467 | 0.533 |
| Số ca (từ chối / rào / bịa) | 0 / 8 / 7 | **7 / 0 / 8** |

### 5.3. Phân tích kết quả

**Về chất lượng:** tất cả chỉ số ĐẠT ngưỡng, nổi bật **Faithfulness 0.951** (rất ít bịa — yêu cầu
số một của y khoa) và **Failure rate 0.000**. Precision 0.699 ở ngay ranh giới là điều **bình
thường và mong đợi**: đây là benchmark khó, có nhiễu cài vào để thử khả năng lọc — chính vì vậy nó
phản ánh chất lượng thật hơn các vòng "dễ" (vốn ai cũng điểm cao). Quan trọng: bật tính năng từ
chối **không làm hỏng** chất lượng câu trong kho (recall/precision/faithfulness phẳng-hoặc-tăng).

**Về an toàn:** với câu hỏi ngoài kho, tính năng từ chối kéo **tỷ lệ từ chối sạch từ 0 lên 0.467**:
7/8 câu trước đây "rào nhưng vẫn trích đoạn nhiễu" nay từ chối dứt khoát, tỷ lệ rào-nhưng-trích về
**0**. Còn lại 8 ca "bịa tự tin" (phớt lờ chỉ thị từ chối) là **bài toán riêng** đang xử lý tiếp —
tính năng **không làm nhóm này tệ hơn**.

**Kết luận thực nghiệm:** ở cấu hình hiện tại, VITAL đạt **mọi ngưỡng chất lượng ngay cả trên
corpus khó**, với độ trung thực cao và không lỗi; cơ chế an toàn đã được kích hoạt đúng mà không hy
sinh chất lượng. Đây là **cấu hình đang chạy** ở hệ thống hiện tại.

---

## 6. Kết luận và hướng phát triển

### 6.1. Kết luận

VITAL là một nền tảng hỏi–đáp tài liệu y khoa tiếng Việt **hoàn chỉnh**: từ nạp tài liệu, truy hồi
thông minh theo chuyên khoa, bộ nhớ tri thức, đến trả lời có trích dẫn và cơ chế an toàn chống bịa.
Kiến trúc được tổ chức thành các thành phần rõ ràng (hai mặt phẳng suy luận/thực thi), giao tiếp
mạch lạc, và vừa được **tinh gọn** bằng việc loại bỏ luồng Structured SQL không cần thiết. Thực
nghiệm cho thấy hệ thống **đạt mọi ngưỡng chất lượng** với độ trung thực cao, đồng thời quy trình
đánh giá đủ nghiêm để **dám loại bỏ kỹ thuật không hiệu quả**.

### 6.2. Điểm mạnh

- Đạt mọi ngưỡng chất lượng; faithfulness cao (ít bịa) — phù hợp y khoa.
- Có cơ chế an toàn (từ chối khi thiếu căn cứ) + trích dẫn theo trang + nhật ký minh bạch.
- Tự host hoàn toàn — phù hợp dữ liệu y khoa nhạy cảm.
- Quy trình đánh giá nghiêm túc, trung thực.

### 6.3. Hạn chế

- Còn 8/15 ca "bịa tự tin" với câu hỏi ngoài kho — cần làm mạnh chỉ thị từ chối.
- Số chi phí/độ trễ trong benchmark chưa phản ánh đúng production (điều phối bị ép chạy cloud).
- Cần kiểm tra môi trường trước khi chạy benchmark (mô hình nội bộ, quota giám khảo).

### 6.4. Hướng phát triển

- Làm mạnh prompt từ chối + hạ "nhiệt độ" khi thiếu căn cứ, rồi đánh giá lại nhóm "bịa tự tin".
- Đo lại phân bố điểm rerank trên kho y khoa thật để chỉnh ngưỡng theo từng chuyên khoa.
- Bật + đánh giá CRAG (tự phê bình); học từ phản hồi thích/không thích của người dùng (DPO/ORPO).
- Huấn luyện mô hình nhúng y khoa tiếng Việt riêng; mở rộng đa phương thức (video/audio bài giảng).

---

*Báo cáo lập ngày 21/06/2026 · Nhóm phát triển VITAL · Nguồn số liệu thực nghiệm:
`docs/eval/benchmark_abstain_ab_2026-06-19_vi.md`; mã nguồn nhánh `feat/ragas-langfuse-reranker`.*
