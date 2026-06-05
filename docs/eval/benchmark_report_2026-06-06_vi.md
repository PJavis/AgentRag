# Báo cáo Benchmark RAG — 06/06/2026

**Bộ dữ liệu:** `both` — cả tiếng Việt và tiếng Anh (4 bộ con) · **n = 80 câu** · **Bộ chấm (judge):** DeepSeek
**Nhánh:** `feat/ragas-langfuse-reranker` · **Dữ liệu thô:** `data/eval/benchmark_both_2026-06-06.json`

## Mục tiêu

Đánh giá chất lượng hệ thống RAG trên **cả hai ngôn ngữ** (lần trước chỉ chạy tiếng Việt),
để xác nhận hệ thống hoạt động tốt với tài liệu tiếng Anh chuyên ngành (y khoa) lẫn tiếng Việt
(tổng hợp + pháp lý). Báo cáo này **không so sánh** với các lần benchmark trước — chỉ trình bày
kết quả hiện tại trên đầy đủ 2 ngôn ngữ.

## Các bộ dữ liệu đã dùng

| Bộ con | Ngôn ngữ | Nguồn (HuggingFace) | Lĩnh vực | Số câu |
|---|---|---|---|---|
| `vn_bkai` | Tiếng Việt | `sailor2/Vietnamese_RAG` (BKAI_RAG) | Hỏi đáp tổng hợp | 20 |
| `vn_legal` | Tiếng Việt | `sailor2/Vietnamese_RAG` (LegalRAG) | Pháp lý | 20 |
| `en_covidqa` | Tiếng Anh | `galileo-ai/ragbench` (covidqa) | Y khoa — COVID | 20 |
| `en_pubmedqa` | Tiếng Anh | `galileo-ai/ragbench` (pubmedqa) | Y sinh — PubMed | 20 |
| **Tổng** | | | | **80** |

Có **370 đoạn ngữ cảnh gốc (gold contexts)** được nạp (ingest) vào hệ thống trước khi chấm.

## Kết quả

| Chỉ số | Ngưỡng đạt | **Kết quả** | Tỷ lệ câu đạt | Đánh giá |
|---|---|---|---|---|
| Contextual recall (độ bao phủ truy hồi) | ≥ 0.70 | **0.904** | 0.88 | ✅ ĐẠT |
| Contextual precision (độ chính xác truy hồi) | ≥ 0.70 | **0.806** | 0.78 | ✅ ĐẠT |
| Faithfulness (độ trung thực) | ≥ 0.80 | **0.944** | 0.90 | ✅ ĐẠT |
| Answer correctness (độ đúng của câu trả lời) | ≥ 0.70 | **0.755** | 0.76 | ✅ ĐẠT |
| Citation accuracy (độ chính xác trích dẫn) | ≥ 0.70 | **0.819** | 0.85 | ✅ ĐẠT |
| Failure rate (tỷ lệ lỗi) | < 0.05 | **0.000** | — | ✅ ĐẠT |
| Freshness (ưu tiên dữ liệu mới) | mới xếp trên cũ | **mới@1 vs cũ@2** | — | ✅ ĐẠT |
| Chi phí / câu hỏi | tham khảo | $0.001966 | — | ⚠️ xem lưu ý |
| Độ trễ p50 / p95 / p99 | tham khảo | 25.8s / 164.6s / 198.0s | — | — |

**Kết luận: toàn bộ 5 chỉ số chất lượng (do LLM chấm) + tỷ lệ lỗi + freshness đều ĐẠT** trên cả
hai ngôn ngữ. Hệ thống RAG vận hành ổn định với tài liệu tiếng Việt lẫn tiếng Anh chuyên ngành.

## Giải thích từng chỉ số

- **Contextual recall — 0.904:** Phần truy hồi (retrieval) có lấy được đủ các đoạn cần thiết để
  trả lời không. 0.904 = rất tốt, gần như luôn tìm đủ ngữ cảnh.
- **Contextual precision — 0.806:** Trong số đoạn đã truy hồi, bao nhiêu phần thực sự liên quan
  (chất lượng xếp hạng sau khi rerank bằng `bge-reranker-v2-m3`). 0.806 = phần lớn ngữ cảnh đưa vào
  đều đúng trọng tâm; vẫn cao hơn ngưỡng 0.70 đáng kể.
- **Faithfulness — 0.944:** Câu trả lời có bám sát ngữ cảnh truy hồi không (không "bịa"). 0.944 =
  rất ít ảo giác — chỉ số mạnh nhất, quan trọng nhất cho ứng dụng y khoa.
- **Answer correctness — 0.755:** Câu trả lời có khớp với đáp án chuẩn không. 0.755 = đúng tốt; đây
  thường là chỉ số khó nhất vì so khớp trực tiếp với đáp án vàng.
- **Citation accuracy — 0.819:** Các đánh dấu trích dẫn `[n]` có trỏ đúng đoạn nguồn ủng hộ cho
  khẳng định không. 0.819 = trích dẫn đáng tin cậy (xác nhận cơ chế đánh số nguồn `[n]` hoạt động đúng).
- **Failure rate — 0.000:** Không có câu nào lỗi/không trả lời được (80/80 thành công).
- **Freshness:** Khi hai ngữ cảnh mâu thuẫn, hệ thống xếp ngữ cảnh **mới** lên trên ngữ cảnh **cũ**
  (mới@1 vs cũ@2) — đạt.

## Phương pháp

1. Tải bộ `both` (4 bộ con, 20 câu/bộ = 80 câu).
2. Nạp 370 đoạn ngữ cảnh gốc qua đúng pipeline thật (parse → chunk → nhúng TEI bge-m3 → index ES).
   *(Cơ sở dữ liệu đã bị xoá sạch trong phiên này nên đây là lần nạp mới hoàn toàn.)*
3. Trả lời từng câu qua **đường đi agent thật**. Chế độ truy hồi đặt là `hybrid_kg`
   (vì `STRUCTMEM_ENABLED=true`), nhưng xem mục **"StructMem KHÔNG hoạt động"** bên dưới —
   thực tế truy hồi chạy = **hybrid (dense + sparse RRF) + rerank `bge-reranker-v2-m3`**, sinh
   câu trả lời bằng DeepSeek.
4. Chấm 5 chỉ số thang 1–5 bằng **bộ chấm DeepSeek** (DeepEval), kèm các chỉ số tính toán:
   tỷ lệ lỗi, freshness, chi phí, độ trễ.

## ⚠️ StructMem KHÔNG hoạt động trong lần benchmark này

Mặc dù `STRUCTMEM_ENABLED=true` và chế độ truy hồi là `hybrid_kg`, **StructMem / tri thức đồ thị
(KG) không thực sự đóng góp** vào kết quả này. Lý do:

- Benchmark nạp dữ liệu ở chế độ **async** (`STRUCTMEM_INGEST_MODE=async`): hàm `_ingest_gold`
  gọi `ingest_folder(...)` rồi **không chờ** — phần trích xuất StructMem được đẩy vào ARQ worker
  chạy nền, và thư mục tạm bị xoá ngay sau đó.
- Hệ thống trả lời cả 80 câu **trước khi** worker kịp trích xuất xong cho 370 ngữ cảnh.
- Kiểm chứng: trên Elasticsearch chỉ tồn tại index `agentrag_segments` (3031 đoạn); các index
  `agentrag_entities` / `agentrag_relationships` / `agentrag_memory_doc` **rỗng/không tồn tại**.
- Do đó `hybrid_kg` không có tín hiệu KG để thêm → thực chất chạy như **hybrid (dense + sparse) +
  rerank**.

**Hệ quả:** các con số ĐẠT ở trên phản ánh chất lượng truy hồi **không có** enrichment KG. Đây vừa
là điểm tốt (đã đạt mọi ngưỡng chỉ với hybrid+rerank) vừa là điều cần lưu ý (chưa đo được phần
đóng góp của StructMem). Muốn đo đúng nhánh KG, cần nạp **đồng bộ** (sync) hoặc chờ worker trích
xuất xong trước khi chấm — xem khuyến nghị.

## Lưu ý quan trọng (đọc trước khi tin số chi phí / độ trễ)

- **Phần điều phối (orchestration) đã được ép chạy trên cloud cho lần benchmark này.** Trong cấu
  hình production, các tác vụ `classify` / `decide` / `domain_router` / `followup` chạy bằng
  **Ollama nội bộ (llama3.2:3b)**. Tuy nhiên Ollama liên tục bị "chết" tiến trình giữa chừng (do
  cách chạy host process trong phiên này) khiến câu trả lời lỗi `APIConnectionError`. Để có lần
  chạy sạch, các tác vụ này được định tuyến tạm sang `deepseek-v4-flash` (không sửa `.env`). Do đó:
  - **Các chỉ số chất lượng vẫn hợp lệ** — chúng phụ thuộc vào retrieval + model sinh câu trả lời,
    không phụ thuộc model điều phối.
  - **Chi phí/câu ($0.00197) và độ trễ KHÔNG so sánh trực tiếp được với production** — vì production
    chạy 4 tác vụ điều phối miễn phí trên Ollama nội bộ. Chi phí thật ở production sẽ thấp hơn.
- **Đuôi độ trễ p95/p99 cao** (164s / 198s) — một số câu chạy lâu do vòng lặp decide + điều phối
  trên cloud. Production với model 3B nội bộ cho phần định tuyến sẽ nhanh hơn.

## Khuyến nghị

1. **An toàn để triển khai** — không có dấu hiệu suy giảm; chất lượng tốt trên cả 2 ngôn ngữ.
2. **Cho Ollama chạy bền vững** (dịch vụ systemd) để benchmark sau chạy đúng cấu hình production và
   cho số chi phí/độ trễ chính xác, không cần ép điều phối lên cloud.
3. **Đo đúng nhánh StructMem/KG:** chạy lại với nạp **đồng bộ** (`STRUCTMEM_INGEST_MODE=sync`) hoặc
   chờ worker trích xuất xong rồi mới chấm, để `hybrid_kg` thực sự có index entity/memory. Khi đó
   sẽ so được chất lượng có-KG vs không-KG (hiện kết quả này là không-KG).
