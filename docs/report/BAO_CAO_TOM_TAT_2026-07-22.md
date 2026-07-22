# Báo cáo tóm tắt — AgentRag/VITAL

> Nền tảng hỏi đáp học liệu y khoa tiếng Việt theo kiến trúc RAG + agent.
> Bản rút gọn, đọc nhanh (~10 trang). Bản đầy đủ: `docs/report/ch1–ch6` và `BaoCao_AgentRag_VITAL.docx`.
> Cập nhật: 2026-07-22.

---

## 1. Vấn đề

Sinh viên, bác sĩ và giảng viên y khoa hằng ngày phải tra cứu khối lượng tài liệu rất lớn: giáo trình
giải phẫu, phác đồ điều trị, hướng dẫn của Bộ Y tế, bài giảng, bảng số liệu, và nhiều tài liệu **scan
từ bản in**. Nhu cầu tra cứu rất cụ thể ("liều khởi đầu của thuốc X là bao nhiêu", "tiêu chuẩn chẩn
đoán hội chứng Y gồm gì") và câu trả lời phải **nhanh, đúng, và kiểm chứng được** — biết thông tin nằm
ở tài liệu nào, trang nào.

Dùng thẳng một mô hình ngôn ngữ lớn (LLM) như ChatGPT không đáp ứng được, vì ba hạn chế cố hữu:

1. **Ảo giác** — LLM sinh ra thông tin nghe hợp lý nhưng sai. Trong y khoa, một liều thuốc bịa ra có
   thể gây hậu quả trực tiếp trên người bệnh.
2. **Không có tài liệu nội bộ** — tri thức của LLM bị đóng băng tại thời điểm huấn luyện và là tri thức
   công cộng; nó không biết kho tài liệu riêng của một bộ môn hay bệnh viện.
3. **Không trích dẫn nguồn** — LLM không chỉ ra được thông tin lấy từ đâu, nên người dùng không thể kiểm
   chứng.

Thêm vào đó, tiếng Việt y khoa có đặc thù riêng: thuật ngữ trộn ba lớp (Hán–Việt "tăng huyết áp", dân
dã "cao huyết áp", tiếng Anh "hypertension"), và nhiều tài liệu là bản scan có dấu, chất lượng không đều,
xen hình vẽ và bảng biểu.

---

## 2. Mục tiêu

Xây dựng **AgentRag/VITAL** — một nền tảng hỏi đáp học liệu y khoa tiếng Việt hoàn chỉnh, với năm mục
tiêu:

1. **Trích dẫn kiểm chứng được đến từng trang** — mỗi ý trong câu trả lời gắn chỉ dấu `[n]` trỏ về đoạn
   tài liệu cụ thể, kèm tên tài liệu và số trang (kiểu NotebookLM).
2. **Chống ảo giác có cơ chế** — hệ thống **chủ động từ chối** (abstain) khi kho tài liệu không có căn
   cứ đủ mạnh, thay vì gượng ép bịa ra câu trả lời.
3. **Mọi tính chất đo lường được** — đi kèm bộ đánh giá tự động, mọi thay đổi phải qua thử nghiệm A/B
   trước khi bật trên hệ thống thật.
4. **Tối ưu cho tiếng Việt y khoa** — OCR có dấu, embedding đa ngữ mạnh về tiếng Việt, ontology thuật
   ngữ y khoa — tiếng Việt là ngôn ngữ hạng nhất, không phải bản dịch gá lắp.
5. **Vận hành trong điều kiện thực tế** — chạy tự chủ (self-hosted) trên GPU phổ thông với mô hình mở,
   đồng thời định tuyến sang cloud khi cần chất lượng cao hơn.

**Phạm vi:** công cụ **tra cứu học liệu** cho học tập và tham khảo chuyên môn — *không phải* hệ tư vấn
hay chẩn đoán cho bệnh nhân. Mô hình triển khai single-tenant (một tổ chức một bản).

---

## 3. Giải pháp: RAG + agent

**RAG (Retrieval-Augmented Generation)** tách tri thức ra khỏi mô hình: thay vì bắt LLM trả lời từ trí
nhớ, hệ thống **truy hồi** các đoạn tài liệu liên quan từ kho thực, đưa vào ngữ cảnh, và yêu cầu LLM chỉ
tổng hợp câu trả lời dựa trên các đoạn đó kèm trích dẫn.

| Tiêu chí | LLM thuần | RAG |
|---|---|---|
| Nguồn tri thức | Tham số, đóng băng | Kho tài liệu thực, cập nhật bằng nạp thêm |
| Tài liệu nội bộ | Không tiếp cận | Là nguồn trả lời chính |
| Ảo giác | Không kiểm soát | Giảm mạnh; có thể từ chối khi thiếu căn cứ |
| Trích dẫn | Không thể | Tới từng tài liệu, từng trang |

RAG đơn giản (truy hồi một lần, trả lời một lần) chưa đủ cho câu hỏi y khoa thực tế — nhiều câu đa bước
(so sánh hai phác đồ ở hai chương khác nhau) cần tách truy vấn con và truy hồi nhiều lượt. Vì vậy đồ án
chọn **agentic RAG**: đặt một **agent** ở trung tâm — vòng lặp trong đó LLM lập kế hoạch, gọi công cụ
truy hồi nhiều lần, tự phê bình kết quả, kiểm tra độ bám căn cứ trước khi trả lời, và **từ chối** khi mọi
nỗ lực đều không đủ căn cứ.

---

## 4. Kiến trúc tổng quan

Hệ thống tổ chức theo **hai mặt phẳng**, nối qua một container tiêm phụ thuộc (`ServiceContainer`):

- **Reasoning Plane (mặt phẳng suy luận)** — quyết định *làm gì*: vòng lặp agent → sinh câu trả lời +
  gắn căn cứ.
- **Execution Plane (mặt phẳng thực thi)** — lo *vào-ra*: gateway LLM đa nhà cung cấp, embedding, truy
  hồi (ES hybrid), lưu trữ, vision. Không rẽ nhánh theo nội dung câu hỏi.

**Bốn tầng lưu trữ:**

| Store | Vai trò |
|---|---|
| **PostgreSQL** (+ pgvector) | source of truth — tài liệu, đoạn văn, hội thoại, notebook |
| **Elasticsearch** | tìm kiếm lai + chỉ mục bộ nhớ tri thức StructMem |
| **Valkey** (tương thích Redis) | cache hội thoại + hàng đợi tác vụ nền ARQ + sổ chi phí |
| **Filesystem** | ảnh trích xuất từ tài liệu |

**Luồng trả lời (`POST /chat`) rút gọn:**

```
Câu hỏi
  → validate → nạp bộ nhớ → kiểm tra chitchat → phân loại ý định
  → lập kế hoạch (tách truy vấn con nếu đa bước)
  → truy hồi: BM25 + kNN dense + RRF fusion + StructMem KG
  → rerank bằng cross-encoder → lọc theo sàn điểm liên quan
  → lắp ráp ngữ cảnh (ngân sách token + sắp lại chống "lost-in-the-middle")
  → sinh câu trả lời + trích dẫn [n]  |  HOẶC  từ chối nếu ngữ cảnh rỗng
  → gắn căn cứ (grounding)
```

Toàn bộ vòng lặp trên là **đồ thị LangGraph 13 nút**, có checkpoint theo `conversation_id` để tiếp tục
hội thoại.

---

## 5. Các tính năng chính

### 5.1. Nạp tài liệu đa định dạng, gắn số trang
- Hỗ trợ PDF (kể cả **bản scan**), DOCX, PPTX, HTML, Markdown, Excel/CSV, ảnh rời.
- Với PDF: chiến lược **leo thang theo từng trang** — ưu tiên lớp văn bản có sẵn → rơi xuống **OCR
  Tesseract** (`vie+eng`, giữ dấu) → chỉ khi vẫn quá mỏng mới gọi **Vision LLM**.
- **Mọi đoạn (chunk) mang theo `page_start`/`page_end`** — nền tảng cho trích dẫn về trang.

### 5.2. Truy hồi lai + phân hoạch theo miền
- **Tìm kiếm lai:** BM25 (từ khóa) + kNN (ngữ nghĩa) + hợp nhất RRF + bộ nhớ tri thức StructMem, để
  bắc cầu giữa các lớp thuật ngữ tiếng Việt.
- **Rerank cross-encoder** xếp hạng lại các ứng viên; điểm rerank cũng là tín hiệu để quyết định từ chối.
- **Phân hoạch miền: 15 hệ cơ quan × 14 chuyên khoa** — mỗi đoạn được gắn nhãn hệ/chuyên khoa dựa trên
  ontology dùng chung + đối sánh mờ `pg_trgm`. Người dùng có thể giới hạn truy vấn theo domain để tăng
  độ chính xác.

### 5.3. Trích dẫn về trang (page-aware citation)
- Câu trả lời gắn chỉ dấu `[n]`; di chuột lên trích dẫn hiện ngay đoạn văn gốc + số trang, bấm vào mở
  đúng trang PDF — phong cách NotebookLM.

### 5.4. Chống ảo giác (abstain)
- Prompt yêu cầu "chỉ trả lời theo ngữ cảnh" là **không đủ**; hệ thống có **cửa chặn cứng** ở tầng truy
  hồi: **sàn điểm liên quan** (floor) đặt trên điểm cross-encoder.
- Khi sàn điểm làm rỗng ngữ cảnh, hệ thống **từ chối tất định — không gọi mô hình sinh nữa**, triệt tiêu
  hoàn toàn khả năng ảo giác từ tham số.
- Từ chối an toàn hai chiều: từ chối đúng với câu ngoài kho, nhưng không từ chối oan khi căn cứ có thật.

### 5.5. Minh bạch — Trace & Cost
- Mỗi câu trả lời có nút **Trace**: xem toàn bộ pipeline agent (`plan → decide → tool → assemble →
  answer → ground`), truy vấn con, I/O từng công cụ.
- **Cost dashboard** (`/cost`): chi phí LLM theo từng tác vụ/mô hình, độ trễ p50/p95, các lời gọi gần đây.

### 5.6. Bộ nhớ tri thức (StructMem)
- Thay thế kiến trúc Graphiti + Neo4j cũ bằng cách trích **thực thể + quan hệ** song song từ mỗi đoạn,
  rồi **hợp nhất xuyên đoạn** để suy luận đa bước — chi phí thấp hơn (~$0.97 so với ~$1.28 / 100 chunks),
  hạ tầng đơn giản hơn (chỉ dùng Elasticsearch).
- Có cả **bộ nhớ hội thoại** ngữ nghĩa thay cho cửa sổ trượt truyền thống.

### 5.7. Sinh nội dung học tập
- **Mindmap Mermaid** và **tóm tắt y khoa 9 mục có cấu trúc** (Định nghĩa → Dịch tễ → Nguyên nhân → Sinh
  lý bệnh → Triệu chứng → Cận lâm sàng → Điều trị → Biến chứng → Tiên lượng).

### 5.8. Đa phương thức (vision)
- Ảnh y tế trong tài liệu được **mô tả bằng Vision LLM** (caption) và **đưa vào chỉ mục truy hồi**.
- Chế độ để mô hình trả lời "đọc" trực tiếp điểm ảnh (answer-time vision) đã được xây và đánh giá A/B, nhưng
  **mặc định TẮT** vì trên corpus + bộ eval hiện tại nó không cải thiện độ đúng (xem §7).

### 5.9. Tự chủ & tích hợp
- Chạy hoàn toàn local với Ollama + Postgres + Elasticsearch + Valkey; định tuyến **từng tác vụ LLM** sang
  cloud (DeepSeek, Gemini, OpenAI) khi cần.
- **MCP server** để dùng AgentRag như một công cụ trong Claude Desktop / Claude Code.
- Xác thực JWT + Google OAuth; rate-limit per-user; dedupe file theo hash.

---

## 6. Bộ đánh giá

Đồ án chủ trương "không đo thì không cải tiến". Bộ đánh giá ngoại tuyến chấm tự động các phương diện:
**faithfulness** (độ trung thực với ngữ cảnh), **context precision/recall** (độ chính xác/độ phủ ngữ
cảnh), **answer correctness** (độ đúng câu trả lời), cùng độ trễ, chi phí, tỷ lệ lỗi.

- Dựa trên hai khung công khai **RAGAS** và **DeepEval** (LLM làm giám khảo — LLM-as-judge).
- Mở rộng bằng **giám khảo tổ hợp tự xây** (nugget-recall + rubric có tham chiếu) và **đầu dò oracle** để
  kiểm định độ tin cậy của *chính thước đo*.
- Xây **bộ đánh giá trên corpus sản phẩm thật** (sinh câu hỏi + gold có căn cứ trên nội dung đã nạp), có
  **dấu vân tay corpus** (`corpus_fp`) chống chạy nhầm bộ eval trên corpus sai.

---

## 7. Kết quả & phát hiện chính

**Số liệu trên corpus y tế thật (bản đo sạch, n≈50):**

| Chỉ số | Kết quả | Ý nghĩa |
|---|---|---|
| Faithfulness | **0.93 – 0.97** | Phần khó nhất — bám căn cứ — chạy tốt |
| Answer correctness (thước đo cũ RAGAS) | **~0.74** (trần) | Chững lại bất chấp cải tiến |
| Correctness (giám khảo tổ hợp mới) | **~0.89** | Con số thật đáng tin |
| Khoảng cách oracle − hệ thống | **+0.046** (< 0.05) | Hệ thống đã sát mức trần khả thi |
| Abstain trên câu ngoài kho | refusal 0 → **0.267** | Cơ chế từ chối thực sự hoạt động |

**Phát hiện quan trọng nhất — trần chất lượng là *thước đo*, không phải hệ thống.**
Độ đúng chững quanh 0.74 dù em thử **hai đòn bẩy mạnh độc lập**: mô hình trả lời mạnh gấp đôi
(`flash → pro`, chỉ +0.006 — nhiễu) và nhiều kiến trúc truy hồi nâng cao (Contextual Retrieval, RAPTOR,
CRAG…, không vượt nhiễu). Khi cả hai đều không nhấc được một chỉ số, **chính chỉ số đó mới là nút thắt.**

Em chứng minh bằng **đầu dò oracle**: đưa cho giám khảo một câu trả lời chắc chắn đúng — thước đo cũ vẫn
chỉ chấm ~0.74, còn giám khảo tổ hợp mới chấm **~0.98**. Nguyên nhân: RAGAS `answer_correctness` (dạng
claim-F1) **phạt các câu đúng-nhưng-diễn-đạt-khác** so với gold ngắn gọn của bộ dữ liệu công khai — đó là
giới hạn của thước đo, không phải lỗi thiết kế. Sau khi thay thước đo và **hai giám khảo độc lập khác nhà
cung cấp đồng thuận (pearson ~0.73–0.97)**, con số correctness thật là **~0.89**.

Đây vừa là kết quả kỹ thuật vừa là **đóng góp phương pháp luận**: không tin mù quáng vào điểm số — phải
kiểm định chính thước đo trước khi kết luận.

**Các kết quả phụ:**
- **Fine-tune embedding y khoa VN (`agentrag-embed-v1`)**: cải thiện truy hồi rõ rệt (**+0.20 recall@10**);
  nhưng fine-tune mô hình *trả lời* không thắng → giữ mô hình trả lời gốc.
- **Rerank-before-trim**: +0.082, đã bật.
- **Vision answer-time**: A/B cho **−0.024 (nhiễu)** → giữ TẮT; nút thắt là truy hồi chưa xếp đúng *ảnh*
  cho câu hỏi thị giác, không phải mô hình trả lời. Caption ảnh vẫn nằm trong chỉ mục và truy hồi được.

---

## 8. Hạn chế

1. **Single-tenant** — chưa có phân quyền chi tiết theo quyền sở hữu notebook/tài liệu giữa nhiều bên
   thuê; đợt kiểm toán bảo mật đã ghi nhận rủi ro IDOR/BOLA như một giới hạn chấp nhận được trong bối cảnh
   đơn thuê bao.
2. **Bộ đánh giá còn phần sinh tự động** — gold do LLM sinh (có lọc), chưa có gold do người gán; câu hỏi
   một số còn dạng đếm thị giác dễ mơ hồ.
3. **Vision chưa tạo lift** — trên corpus hiện tại, đọc ảnh trực tiếp không cải thiện độ đúng; cần truy hồi
   ảnh tốt hơn trước khi bật.
4. **Node self-critique** trong LangGraph vẫn ở dạng chờ port lại từ vòng lặp cũ.

---

## 9. Hướng phát triển

**Ngắn hạn:** node self-critique trong LangGraph; **học từ phản hồi người dùng** (dữ liệu thumbs up/down
đã thu → dataset preference-pair → DPO/ORPO); mở rộng ontology các chuyên khoa còn thiếu; hover-preview
trích dẫn.

**Trung hạn:** đa phương thức mở rộng sang **video bài giảng** (Whisper transcribe) và audio; trực quan
hóa đồ thị tri thức của tài liệu; cô lập dữ liệu **multi-tenant** theo workspace; giao diện responsive cho
di động.

**Dài hạn:** truy hồi liên-cơ-sở (federated) bảo toàn riêng tư; tích hợp hỗ trợ quyết định lâm sàng
(guideline + tương tác thuốc); triển khai on-device cho tuyến huyện; nhật ký audit chuẩn HIPAA-like.

---

## 10. Kết luận

AgentRag/VITAL là một hệ RAG y khoa tiếng Việt **hoàn chỉnh và có thể vận hành**: trả lời **có trích dẫn
tới từng trang**, **biết từ chối khi thiếu căn cứ**, xử lý được cả tài liệu scan và ảnh, chạy **tự chủ
trên hạ tầng riêng**, và — điểm khác biệt — đi kèm một **bộ đánh giá được kiểm định độ tin cậy**. Kết quả
faithfulness 0.93–0.97 cho thấy phần khó nhất đã chạy tốt; và phát hiện "trần chất lượng nằm ở thước đo,
không ở hệ thống" là một đóng góp phương pháp luận có giá trị vượt ra ngoài phạm vi đồ án.

---

*Tài liệu liên quan: kiến trúc chi tiết `ARCHITECTURE.md`; báo cáo đầy đủ `docs/report/ch1–ch6` +
`BaoCao_AgentRag_VITAL.docx`; script demo `docs/report/SCRIPT_DEMO_2026-07-22.md`.*
