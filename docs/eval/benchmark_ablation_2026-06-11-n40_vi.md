# Báo cáo Benchmark Ablation n=40 — baseline vs CR+RAPTOR (11/06/2026)

So sánh hệ thống RAG **gốc (baseline)** với **bật Contextual Retrieval + RAPTOR (cr_raptor)** ở
quy mô **n=40** (gấp 4 lần n=10 trước) để có kết luận đáng tin hơn.

- Bộ dữ liệu: `both` (160 câu) — `vn_bkai`, `vn_legal`, `en_covidqa`, `en_pubmedqa`, n=40/bộ
- Judge: DeepSeek · Nhánh: `feat/ragas-langfuse-reranker` · Thời lượng: ~12 giờ
- Dữ liệu thô: `data/eval/_ablation_baseline.json`, `_ablation_cr_raptor.json` · Bảng: `benchmark_ablation_2026-06-11-n40.md`

## Kết quả

| Chỉ số | baseline | cr_raptor | Δ |
|---|---|---|---|
| Contextual recall | 0.874 | 0.891 | **+0.017** |
| Contextual precision | 0.799 | 0.800 | +0.001 |
| Faithfulness | 0.954 | 0.927 | **−0.027** |
| Answer correctness | 0.740 | 0.710 | **−0.030** |
| Citation accuracy | 0.833 | 0.804 | **−0.029** |
| Failure rate | 0.006 | 0.000 | −0.006 |
| Chi phí / câu | $0.00126 | $0.00128 | +1% |
| Độ trễ p50 | 26.4s | 27.0s | +0.6s |

## ⚠️ Kết luận: ở n=40, CR+RAPTOR KHÔNG thắng baseline — và kết quả n=10 trước là NHIỄU

- **n=10 (lần trước):** citation **+0.035** → có vẻ CR+RAPTOR giúp trích dẫn tốt hơn.
- **n=40 (lần này, đáng tin hơn):** citation **−0.029**, correctness **−0.030**, faithfulness
  **−0.027**; chỉ recall +0.017 (nhỏ). → **Lợi ích citation ở n=10 không bền vững** (đảo dấu khi
  tăng mẫu), tức là **nhiễu mẫu**, không phải hiệu ứng thật.
- cr_raptor có recall nhỉnh hơn (+0.017 — lấy được nhiều ngữ cảnh liên quan hơn, hợp lý vì RAPTOR
  thêm summary nodes + CR làm chunk dễ match hơn), **nhưng** correctness/faithfulness/citation lại
  **giảm**. Giả thuyết: thêm summary nodes (RAPTOR) + ngữ cảnh prepend (CR) đưa thêm nội dung
  tóm tắt/diễn giải vào context → model đôi khi trả lời theo bản tóm tắt thay vì trích dẫn nguyên
  văn nguồn gốc → citation/faithfulness/correctness giảm nhẹ.

→ **Ở quy mô n=40, bật CR+RAPTOR KHÔNG cải thiện chất lượng — thậm chí giảm nhẹ 3/5 chỉ số.**
Các Δ đều nhỏ (≤0.03, vẫn trong vùng nhiễu n=40), nên đọc là **"trung tính đến hơi xấu"**, không
phải "phá hỏng". Nhưng **không có bằng chứng CR+RAPTOR giúp** trên corpus benchmark này.

## Phương pháp (harness đã sửa — số liệu hợp lệ)

Lần này dùng harness ablation **đã sửa cả 3 lỗi** (dedupe no-op / chỉ-wipe-ES / sys.path-PG-wipe).
Bằng chứng hợp lệ: baseline recall 0.874 (không phải 0 giả như khi harness lỗi). Mỗi config
**wipe sạch ES+PG → re-ingest đúng cờ** (sync) → so sánh công bằng. cr_raptor thực sự chạy CR
(prepend ngữ cảnh từng chunk) + RAPTOR (summary tree), không bị dedupe bỏ qua như trước.

## Lưu ý

- n=40 đáng tin hơn n=10 nhưng vẫn có sai số (~±0.03). Kết luận chắc: **không có lợi ích rõ ràng**;
  hướng dịch chuyển (citation +0.035 → −0.029) cho thấy n=10 không đáng tin.
- Điều phối ép cloud (ổn định khi chạy); chỉ số chất lượng hợp lệ, chi phí/độ trễ không so prod.
- CR/RAPTOR build đồng bộ trong benchmark (đúng phương pháp); prod giữ async.

## Khuyến nghị (cập nhật — đảo so với báo cáo n=10)

1. **Xem lại quyết định bật CR+RAPTOR ở prod.** n=40 cho thấy **không có lợi ích chất lượng đo
   được**; correctness/citation/faithfulness giảm nhẹ. Khuyến nghị **TẮT** CR+RAPTOR theo metric
   thuần — TRỪ KHI muốn giữ vì lý do **UX định tính** (dòng "context" khi hover citation + badge
   `Σ Summary` của RAPTOR giúp người đọc), chấp nhận đánh đổi metric nhỏ.
2. Nếu muốn chắc chắn hơn nữa: chạy n≥60, hoặc tách riêng `cr` và `raptor` để biết cái nào gây
   giảm correctness/citation (nghi RAPTOR summary nodes lấn át trích dẫn nguyên văn).
3. Các flag query-time (CRAG/multihop/adaptive/cache) chưa benchmark ở n=40 — vẫn để TẮT.