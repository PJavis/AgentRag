# Báo cáo A/B — Relevance-floor gate OFF vs ON (16/06/2026)

Kiểm chứng `RETRIEVAL_RELEVANCE_GATE_ENABLED` (floor 0.3): gate cắt các chunk có điểm rerank
(sigmoid bge cross-encoder) dưới ngưỡng TRƯỚC khi answer thấy. Giả thuyết: out-of-corpus → mọi
chunk dưới floor → context rỗng → **từ chối sạch** (refusal↑, hallucination↓), mà không hại
chất lượng in-corpus.

- Cùng kho grouped+CR+RAPTOR (skip-ingest → chỉ khác cái gate, so sánh công bằng)
- 80 câu in-corpus + 15 câu out-of-corpus · judge DeepSeek · 0 lỗi

## Kết quả

**Chất lượng in-corpus:**

| Chỉ số | OFF | ON | Δ |
|---|---|---|---|
| Contextual recall | 0.895 | 0.865 | **−0.030** |
| Contextual precision | 0.711 | 0.692 | −0.019 |
| Faithfulness | 0.920 | 0.942 | +0.022 |
| Answer correctness | 0.720 | 0.721 | +0.001 |
| Citation accuracy | 0.735 | 0.784 | +0.049 |
| p50 | 24.3s | 23.6s | −0.7s |

**An toàn out-of-corpus (15 câu thuốc/tech bịa):**

| Chỉ số | OFF | ON |
|---|---|---|
| refusal_rate (từ chối sạch — lý tưởng) | 0.000 | 0.000 |
| hedged_cited_rate (rào nhưng trích distractor) | 0.533 | 0.400 |
| **hallucination_rate (NGUY HIỂM)** | 0.467 | **0.600** |

## ⚠️ Kết luận: gate PHẢN TÁC DỤNG — KHÔNG bật ở prod

Giả thuyết sai. Bật gate ở floor 0.3:

1. **Tăng hallucination (0.467 → 0.600)** — điều tệ nhất. Gate cắt chunk distractor trên câu
   out-of-corpus → model **mất luôn ngữ cảnh mơ hồ vốn khiến nó rào** ("ngữ cảnh có nhắc Paris
   nhưng…") → thay vì im lặng, nó trả lời theo **kiến thức tham số → bịa tự tin**. Nói cách khác
   gate **biến hedged_cited (0.533→0.40, an toàn-nhưng-bẩn) thành hallucinated (an toàn ← KHÔNG,
   nguy hiểm)**, KHÔNG phải thành abstained. refusal_rate vẫn 0.0 — model **không bao giờ từ chối
   sạch** dù gate bật hay tắt.
2. **Giảm recall in-corpus (−0.030)** — gate cũng cắt nhầm chunk liên quan thật (floor 0.3 quá
   cao hoặc điểm rerank chưa hiệu chỉnh) → bớt ngữ cảnh đúng.
3. Chỉ faithfulness/citation nhỉnh lên (+0.02/+0.05) — hệ quả của việc đơn giản là ÍT chunk hơn
   (ít cơ hội trích sai), không phải lọc thông minh hơn.

→ **Đánh đổi xấu:** mất recall + tăng bịa, đổi lấy citation cao hơn. Để gate **OFF**.

## Nguyên nhân gốc + hướng đúng

Vấn đề thật: **đường abstain/từ chối không kích hoạt** trên out-of-corpus (refusal_rate=0 cả hai
chiều). Bỏ context (gate) KHÔNG ép model im lặng — nó quay sang bịa. Sửa đúng nằm ở **tầng
answer/prompt**, không phải tỉa retrieval:

1. Khi mọi chunk truy hồi dưới floor (không có bằng chứng đủ liên quan) → **chèn chỉ thị tường
   minh** vào prompt: "Không có ngữ cảnh liên quan → trả lời 'Tôi không có thông tin', KHÔNG suy
   từ kiến thức nền, KHÔNG trích nguồn." Đây mới là cái biến hành vi sang abstained, thay vì chỉ
   xoá context rồi để model tự bịa.
2. (Không liên quan) "unscored-tail" trong `apply_relevance_floor`: không phải nguyên nhân ở đây
   — vấn đề là chiến lược, không phải rò tail.

## Khuyến nghị

- **`RETRIEVAL_RELEVANCE_GATE_ENABLED=false`** (giữ mặc định OFF). Không ship.
- Làm tính năng **abstain-on-thin-context ở answer node** (chỉ thị prompt khi top rerank < floor),
  rồi A/B lại bằng đúng harness này (gate-style nhưng can thiệp ở prompt). Đó mới là phép thử cho
  refusal_rate↑ + hallucination↓.
- n=80/15, sai số ±0.05–0.1 (refusal n=15 nhỏ) — nhưng hướng (hallucination TĂNG) đủ rõ để bác gate.
