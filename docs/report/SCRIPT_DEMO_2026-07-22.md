# Script video demo — AgentRag/VITAL (bảo vệ đồ án)

> **Thời lượng mục tiêu:** 5–7 phút · **Ngôn ngữ:** tiếng Việt · **Mục đích:** trình bày bảo vệ đồ án.
> Cột trái là **lời thoại** (đọc/thu âm), cột phải là **thao tác trên màn hình** (quay lại đúng lúc).
> Chuẩn bị trước: đã `make dev`, có sẵn 1 notebook đã nạp vài giáo trình y khoa (PDF text + 1 PDF scan + 1 ảnh atlas),
> mở sẵn tab `http://localhost:3000` và `http://localhost:3000/cost`.

---

## Timeline tổng quan

| Thời gian | Phân đoạn | Nội dung |
|---|---|---|
| 0:00 – 0:40 | 1. Mở đầu & vấn đề | Vì sao LLM thuần không dùng được cho y khoa |
| 0:40 – 1:30 | 2. Giải pháp & kiến trúc | AgentRag là gì, 2 mặt phẳng, luồng xử lý |
| 1:30 – 2:15 | 3. Demo: nạp tài liệu | Upload PDF/scan/ảnh → parse → chunk gắn số trang |
| 2:15 – 3:30 | 4. Demo: hỏi đáp có trích dẫn | Câu hỏi → câu trả lời + citation trang → hover xem nguồn |
| 3:30 – 4:15 | 5. Demo: minh bạch & an toàn | Trace suy luận + abstain khi không có căn cứ |
| 4:15 – 5:00 | 6. Demo: tính năng chuyên biệt | Lọc chuyên khoa, mindmap/tóm tắt, cost dashboard |
| 5:00 – 6:00 | 7. Kết quả đánh giá | Số liệu eval + phát hiện "trần là thước đo" |
| 6:00 – 6:40 | 8. Kết luận & hướng phát triển | Đóng góp chính, giới hạn, tương lai |

---

## Phân đoạn 1 — Mở đầu & vấn đề (0:00 – 0:40)

**Lời thoại:**
> "Xin chào thầy cô. Em xin trình bày đồ án **AgentRag/VITAL** — nền tảng hỏi đáp học liệu y khoa tiếng Việt.
> Bài toán xuất phát từ một thực tế: sinh viên và bác sĩ phải tra cứu hàng trăm trang giáo trình, phác đồ, atlas.
> Nếu dùng thẳng một mô hình ngôn ngữ lớn như ChatGPT, ta gặp ba vấn đề chí mạng trong y khoa:
> **một là ảo giác** — mô hình bịa ra thông tin nghe rất hợp lý nhưng sai;
> **hai là không đọc được tài liệu nội bộ** của bộ môn hay bệnh viện;
> **ba là không trích dẫn được nguồn** để người dùng kiểm chứng.
> Trong y khoa, một liều thuốc bịa ra có thể gây hậu quả trực tiếp. Đồ án giải quyết đúng ba vấn đề này."

**Màn hình:**
- Slide tiêu đề (tên đồ án + tên sinh viên).
- Slide 3 gạch đầu dòng: *Ảo giác · Không có tài liệu nội bộ · Không trích dẫn nguồn*.

---

## Phân đoạn 2 — Giải pháp & kiến trúc (0:40 – 1:30)

**Lời thoại:**
> "Giải pháp là kiến trúc **RAG kết hợp agent**. RAG tách tri thức ra khỏi mô hình: hệ thống truy hồi
> đúng các đoạn tài liệu liên quan, rồi bắt mô hình chỉ trả lời dựa trên các đoạn đó, kèm trích dẫn.
> Điểm khác với RAG thông thường là em đặt một **agent** ở trung tâm — một vòng lặp trên LangGraph gồm 13 nút:
> nó lập kế hoạch, truy hồi nhiều lượt, tự phê bình, gắn căn cứ, và **từ chối trả lời khi không đủ căn cứ**.
> Hệ thống chia làm hai mặt phẳng: **Reasoning Plane** lo việc suy luận — quyết định làm gì,
> và **Execution Plane** lo vào-ra — gọi mô hình, truy hồi, lưu trữ. Toàn bộ chạy được tự chủ trên máy chủ riêng."

**Màn hình:**
- Sơ đồ kiến trúc 2 mặt phẳng (lấy từ `ARCHITECTURE.md`).
- Sơ đồ luồng ngắn: `Câu hỏi → classify → plan → retrieve (BM25+kNN+RRF) → rerank → answer + citation → ground`.

---

## Phân đoạn 3 — Demo nạp tài liệu (1:30 – 2:15)

**Lời thoại:**
> "Đầu tiên là nạp tài liệu. Em kéo-thả một giáo trình PDF, một bản **scan không có lớp chữ**, và một ảnh atlas giải phẫu.
> Với PDF thường, hệ thống bóc lớp văn bản; với bản scan, nó leo thang xuống **OCR tiếng Việt có dấu**;
> với ảnh y tế, nó dùng **Vision LLM** để mô tả thành văn bản truy hồi được.
> Quan trọng: mỗi đoạn được cắt ra đều **gắn số trang bắt đầu và kết thúc** — đây là nền tảng cho trích dẫn về sau."

**Màn hình:**
- Kéo-thả 3 file vào notebook, hiện tiến trình nạp.
- (Tua nhanh phần chờ.) Mở panel Sources cho thấy các tài liệu đã sẵn sàng.

---

## Phân đoạn 4 — Demo hỏi đáp có trích dẫn (2:15 – 3:30)

**Lời thoại:**
> "Giờ em hỏi một câu thực tế — ví dụ *'Tiêu chuẩn chẩn đoán tăng huyết áp gồm những gì?'*.
> Câu trả lời trả về kèm các chỉ dấu trích dẫn `[1] [2]`. Em di chuột lên một trích dẫn —
> nó hiện ngay **đoạn văn gốc và số trang** trong tài liệu, đúng phong cách NotebookLM.
> Người dùng luôn kiểm chứng được, không phải tin suông.
> Đáng chú ý: em hỏi bằng cụm 'tăng huyết áp' nhưng tài liệu có thể viết 'cao huyết áp' hay 'hypertension' —
> nhờ **tìm kiếm lai** kết hợp từ khóa và ngữ nghĩa, cùng lớp ontology thuật ngữ, hệ thống vẫn bắc cầu được."

**Màn hình:**
- Gõ câu hỏi trong ChatPanel, câu trả lời stream ra kèm citation.
- Hover một citation → hiện preview đoạn trích + số trang.
- (Tùy chọn) Bấm vào citation để mở PDF đúng trang.

---

## Phân đoạn 5 — Minh bạch & an toàn (3:30 – 4:15)

**Lời thoại:**
> "Hai điểm em muốn nhấn mạnh về độ tin cậy.
> **Thứ nhất — minh bạch.** Mỗi câu trả lời có nút **Trace**. Bấm vào, ta thấy toàn bộ đường đi của agent:
> nó đã lập kế hoạch gì, gọi công cụ truy hồi mấy lần, truy vấn con là gì, xếp hạng ra sao.
> **Thứ hai — an toàn.** Em cố tình hỏi một câu **ngoài kho tài liệu**.
> Thay vì bịa, hệ thống **từ chối trả lời** — vì điểm liên quan không vượt ngưỡng, ngữ cảnh bị làm rỗng,
> và khi ngữ cảnh rỗng nó **không gọi mô hình sinh** nữa, triệt tiêu hoàn toàn khả năng ảo giác."

**Màn hình:**
- Bấm **Trace** trên một câu trả lời → dialog hiện node graph + I/O từng công cụ.
- Gõ câu hỏi lạc đề (vd hỏi về chủ đề không có trong tài liệu) → hệ thống trả lời từ chối lịch sự.

---

## Phân đoạn 6 — Tính năng chuyên biệt (4:15 – 5:00)

**Lời thoại:**
> "Vài tính năng phục vụ đúng ngữ cảnh y khoa.
> Kho tri thức được phân hoạch theo **15 hệ cơ quan × 14 chuyên khoa**; em có thể giới hạn truy vấn theo domain —
> ví dụ chỉ tim mạch — để tăng độ chính xác.
> Hệ thống sinh **mindmap Mermaid** và **bản tóm tắt y khoa 9 mục** để ôn tập.
> Và cho người vận hành, có **bảng chi phí** theo dõi từng lời gọi mô hình, độ trễ p50/p95 — luôn biết hệ thống tốn bao nhiêu."

**Màn hình:**
- Chọn domain filter trên ChatPanel, hỏi lại.
- Bấm tạo mindmap / summary cho một tài liệu.
- Chuyển tab `/cost` cho thấy dashboard chi phí + độ trễ.

---

## Phân đoạn 7 — Kết quả đánh giá (5:00 – 6:00)

**Lời thoại:**
> "Về đánh giá — đây là phần em tâm đắc nhất.
> Trên corpus y tế thật, độ **trung thực với ngữ cảnh (faithfulness) đạt 0.93–0.97** — phần khó nhất đã chạy tốt.
> Nhưng độ **đúng của câu trả lời (correctness) chững lại quanh 0.74** dù em thử cả mô hình trả lời mạnh gấp đôi
> lẫn nhiều kiến trúc truy hồi nâng cao. Khi hai đòn bẩy mạnh độc lập đều không nhấc được một chỉ số,
> **vấn đề nằm ở thước đo, không phải ở hệ thống.**
> Em chứng minh điều đó bằng **đầu dò oracle**: đưa cho giám khảo một câu trả lời chắc chắn đúng —
> thước đo cũ vẫn chỉ chấm 0.74, còn giám khảo tổ hợp mới em tự xây chấm **~0.98**.
> Sau khi thay thước đo và hai giám khảo độc lập khác nhà cung cấp đồng thuận, con số thật là **~0.89**.
> Đây là đóng góp phương pháp luận: **không tin mù quáng vào điểm số — phải kiểm định chính thước đo.**"

**Màn hình:**
- Slide bảng số liệu: `faithfulness 0.93–0.97 | correctness cũ 0.74 → oracle 0.98 → thước đo mới ~0.89`.
- Slide "Phát hiện: trần chất lượng là *thước đo*, không phải hệ thống".

---

## Phân đoạn 8 — Kết luận & hướng phát triển (6:00 – 6:40)

**Lời thoại:**
> "Tóm lại, AgentRag/VITAL là một hệ RAG y khoa tiếng Việt hoàn chỉnh:
> trả lời **có trích dẫn tới từng trang**, **biết từ chối khi thiếu căn cứ**, xử lý được cả tài liệu scan và ảnh,
> chạy tự chủ trên hạ tầng riêng, và đi kèm **bộ đánh giá được kiểm định độ tin cậy**.
> Giới hạn hiện tại: mô hình single-tenant và bộ đánh giá còn phần sinh tự động.
> Hướng phát triển: mở rộng đa phương thức sang video bài giảng, học tiếp từ phản hồi người dùng, và tinh chỉnh mô hình y khoa chuyên sâu.
> Em xin cảm ơn thầy cô và sẵn sàng trả lời câu hỏi."

**Màn hình:**
- Slide "Đóng góp chính" (3–4 gạch đầu dòng).
- Slide "Hướng phát triển" + lời cảm ơn.

---

## Ghi chú quay dựng

- **Tổng khớp thời lượng:** cân đối để dừng ở ~6:40; nếu vượt, cắt bớt phân đoạn 6 (tính năng phụ) trước.
- **Tua nhanh** mọi đoạn chờ nạp tài liệu / mô hình suy nghĩ (giữ nhịp).
- **Zoom** vào citation hover và Trace dialog — hai điểm "ăn tiền" nhất, đừng để chữ quá nhỏ.
- Chuẩn bị **câu hỏi ngoài kho** đã test trước để chắc chắn hệ thống abstain đúng lúc quay.
- Nếu quay ở nhà, bật `VISION_PROVIDER` để ảnh có caption; nhưng để `VISION_ANSWER_MODEL` OFF (mặc định) — đúng cấu hình đã đánh giá.
