# Báo cáo Benchmark "khó" — grouped corpus + refusal (2026-06-15)

Lần đầu chạy benchmark **có thể fail**: gộp ngữ cảnh thành doc đa-chunk (`--group-size 4` → kích
hoạt RAPTOR + distractors) và thêm bộ câu **ngoài kho tài liệu** để đo an toàn (`--refusal-set`).
Khác hẳn benchmark cũ (chỉ gold, 1 passage/doc, mọi câu đều trả lời được → gần trần, RAPTOR bị bỏ qua).

- Suite `both`, n=20 → 80 câu gold + 15 câu out-of-corpus · judge DeepSeek · cấu hình: CR+RAPTOR BẬT
- Ingest đồng bộ (CR contextualize + RAPTOR summary tree + StructMem) · ~5h · 0 lỗi
- Dữ liệu thô: `data/eval/grouped_2026-06-15.json`

## Chất lượng (có distractors → khó hơn)

| Chỉ số | grouped (lần này) | n=40 "dễ" (tham chiếu) | Δ |
|---|---|---|---|
| Contextual recall | 0.852 | 0.874 | −0.022 |
| **Contextual precision** | **0.689** | 0.799 | **−0.110** |
| Faithfulness | 0.925 | 0.954 | −0.029 |
| Answer correctness | 0.711 | 0.740 | −0.029 |
| Citation accuracy | 0.805 | 0.833 | −0.028 |
| Failure rate | 0.000 | — | — |
| p50 | 24.8s | — | — |
| cost/câu | $0.00171 | — | — |

**Đọc:** thêm distractors kéo **precision tụt mạnh (0.799 → 0.689)** — đúng như mong đợi: retriever
giờ phải lọc nhiễu thật, benchmark **không còn gần-trần**. Các chỉ số khác giảm nhẹ. Đây là điều
TỐT về mặt đo lường: cuối cùng cũng có một benchmark **phân biệt được** chất lượng (cái cũ không).

## ⚠️ An toàn — refusal_rate = 0.00 (15 câu ngoài kho)

`refusal_rate = 0.0` → theo bộ phát hiện, **0/15** câu out-of-corpus được "từ chối sạch". Nhưng đọc
per_case cho thấy bức tranh tinh tế hơn:

- Model **KHÔNG bịa tự tin** — nó **rào trước** ("Ngữ cảnh được cung cấp không có thông tin về…",
  "context does not contain…"). Các thuốc bịa (Vextramol, Blorbocide) → trả lời "không có thông tin".
- NHƯNG nó **vẫn trích dẫn chunk distractor liên quan mơ hồ** rồi mới rào. Ví dụ "Thủ đô Pháp?" →
  nó bới được chunk có "Paris, Pháp" (địa chỉ trong một bản tin) → trích `[1][2]` → bộ phát hiện
  abstention (yêu cầu **bất định VÀ không có citation**) xếp là **KHÔNG abstain**.
- Thêm nữa, refusal set là **trivia chung** (thủ đô Pháp, Hamlet, đại dương) trùng một phần với kho
  tin tức/y khoa → "ngoài kho" không hoàn toàn ngoài (Paris CÓ xuất hiện) → refusal_rate bị thấp giả.

**Kết luận an toàn:** không phải "bịa nguy hiểm" (model rào, không khẳng định sai), nhưng **cũng
chưa từ chối sạch** — nó nên nói "không có thông tin" mà KHÔNG vớ lấy distractor để trích dẫn. Đây
là điểm cần cải thiện rõ ràng nhất.

## Khuyến nghị

1. **Siết logic abstain:** khi chỉ truy hồi được chunk distractor/tangential (điểm rerank thấp, không
   khớp thực thể câu hỏi) → từ chối thẳng, KHÔNG trích dẫn. Hiện model rào-nhưng-vẫn-trích → vừa
   không an toàn rõ ràng vừa làm refusal_rate=0.
2. **Refusal set tốt hơn:** dùng câu out-of-corpus thật sự không giao với kho (thuốc/bệnh bịa như
   Vextramol/Blorbocide hoạt động tốt; bỏ trivia trùng tin tức như "thủ đô Pháp").
3. **Grouped mode là benchmark đáng dùng từ giờ** — nó phân biệt được chất lượng (precision tụt khi
   có nhiễu). Chạy thêm baseline-grouped (CR+RAPTOR TẮT) cùng group-size để đo CR/RAPTOR có giúp lọc
   distractor không (đây mới là phép thử công bằng cho chúng, khác benchmark "dễ" trước).

## Lưu ý

- 1 cấu hình (CR+RAPTOR bật) — chưa có baseline-grouped để so trực tiếp; bảng trên so với n=40 "dễ"
  chỉ để thấy mức độ khó tăng, không phải so CR/RAPTOR.
- Điều phối ép cloud (ổn định); chất lượng hợp lệ, chi phí/độ trễ không so prod.
- Harness đã sửa (wipe ES+PG sạch) → ingest grouped thật sự build RAPTOR (mem ~4.4k, seg ~970).
