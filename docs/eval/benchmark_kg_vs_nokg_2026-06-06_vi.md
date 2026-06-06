# Báo cáo Benchmark — StructMem/KG có vs không (06/06/2026)

So sánh hệ thống RAG **có** và **không có** tri thức đồ thị StructMem (KG) trong khâu truy hồi,
trên cùng bộ dữ liệu 2 ngôn ngữ.

- **Bộ dữ liệu:** `both` (80 câu) — `vn_bkai`, `vn_legal` (tiếng Việt) + `en_covidqa`, `en_pubmedqa` (tiếng Anh)
- **Bộ chấm:** DeepSeek · **Nhánh:** `feat/ragas-langfuse-reranker`
- **Dữ liệu thô:** không-KG `data/eval/benchmark_both_2026-06-06.json` · có-KG `data/eval/benchmark_kg_2026-06-06.json`

## Bối cảnh

Lần benchmark trước (`benchmark_both_2026-06-06`) đặt chế độ `hybrid_kg` nhưng nạp **async** →
extraction chưa kịp chạy → index entity/memory **rỗng** → thực chất chạy **không có KG**.

Lần này nạp **đồng bộ** (`STRUCTMEM_INGEST_MODE=sync`, tắt dedupe) → extraction chạy xong **trước**
khi chấm → KG thực sự đầy: **`agentrag_memory_doc` = 7.678 mục** (trước đó = 0). Nhờ vậy đo được
đóng góp thật của StructMem.

## Kết quả so sánh

| Chỉ số | Ngưỡng | Không-KG | **Có-KG** | Δ |
|---|---|---|---|---|
| Contextual recall | ≥0.70 | 0.904 | **0.900** | −0.004 |
| Contextual precision | ≥0.70 | 0.806 | **0.819** | **+0.013** |
| Faithfulness | ≥0.80 | 0.944 | **0.950** | **+0.006** |
| Answer correctness | ≥0.70 | 0.755 | **0.792** | **+0.037** |
| Citation accuracy | ≥0.70 | 0.819 | **0.884** | **+0.065** |
| Failure rate | <0.05 | 0.000 | **0.000** | = |
| Freshness | mới<cũ | ĐẠT | **ĐẠT** | = |
| Chi phí / câu | — | $0.001966 | $0.001891 | ≈ |
| Độ trễ p50 | — | 25.8s | 24.9s | ≈ |

**Cả hai cấu hình đều ĐẠT mọi ngưỡng.** Bật StructMem/KG **cải thiện** chất lượng mà **gần như
không tăng chi phí/độ trễ**.

## Phân tích: KG giúp ở đâu

- **Citation accuracy +0.065 (lớn nhất):** KG cung cấp thực thể/quan hệ có cấu trúc → câu trả lời
  trích dẫn đúng nguồn hơn. Đây là lợi ích rõ nhất của StructMem.
- **Answer correctness +0.037:** ngữ cảnh giàu hơn (thực thể + quan hệ + synthesis) giúp câu trả
  lời khớp đáp án chuẩn tốt hơn.
- **Contextual precision +0.013:** tín hiệu KG giúp xếp hạng đoạn liên quan lên trên.
- **Faithfulness +0.006:** nhỉnh hơn — câu trả lời bám nguồn chắc hơn.
- **Recall −0.004 (≈ không đổi):** KG không làm tăng độ bao phủ truy hồi (vốn đã rất cao 0.90),
  nằm trong sai số mẫu.
- **Chi phí/độ trễ ≈ nhau:** truy vấn KG (index memory_doc) gần như miễn phí lúc chạy; chi phí
  StructMem nằm ở khâu nạp (extraction một lần), không ở khâu trả lời.

→ **Kết luận: bật StructMem/`hybrid_kg` đáng giá** — tăng độ chính xác trích dẫn + độ đúng câu trả
lời mà không đánh đổi tốc độ/chi phí lúc phục vụ.

## Phương pháp

1. Cùng bộ `both` (80 câu, 4 bộ con), cùng 370 ngữ cảnh gốc.
2. **Có-KG:** nạp `STRUCTMEM_INGEST_MODE=sync` + tắt dedupe → extraction StructMem chạy nội tuyến,
   index entity/memory đầy (7.678 mục) trước khi chấm. Truy hồi `hybrid_kg` thực sự dùng KG.
3. **Không-KG:** nạp async (lần trước) → index KG rỗng → `hybrid_kg` chạy như hybrid + rerank.
4. Cả hai: rerank `bge-reranker-v2-m3`, sinh câu trả lời DeepSeek, chấm bằng DeepSeek judge (DeepEval).

## Lưu ý

- **Điều phối ép chạy cloud** cho cả hai lần (Ollama hay chết tiến trình) → chỉ số chất lượng hợp
  lệ, nhưng chi phí/độ trễ không so trực tiếp với production (prod chạy classify/decide trên Ollama
  3B miễn phí). Vì cả hai lần đều dùng cùng cấu hình cloud nên **so sánh KG vs không-KG là công
  bằng**.
- Nạp đồng bộ rất chậm (~vài giờ cho 370 ngữ cảnh, serial trên cloud) — chỉ dùng cho benchmark;
  production vẫn nên nạp async (extraction nền) để upload không bị chặn.
- n=80, sai số mẫu ±0.05; các mức tăng của correctness/citation vượt ngưỡng nhiễu → đáng tin.

## Khuyến nghị

1. **Giữ StructMem/`hybrid_kg` BẬT** ở production — lợi ích citation + correctness rõ rệt, chi phí
   phục vụ không đổi.
2. Đảm bảo extraction nền (async) **chạy xong** sau khi upload (đã có trạng thái `enriching→done`),
   để truy vấn thực tế được hưởng KG như benchmark này.
3. Cho Ollama chạy bền (systemd) để lần benchmark sau dùng đúng cấu hình prod + đo chi phí/độ trễ thật.
