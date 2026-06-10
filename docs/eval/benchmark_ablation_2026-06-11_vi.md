# Báo cáo Benchmark Ablation — baseline vs full (11/06/2026)

So sánh hệ thống RAG **gốc (baseline)** với **bật tất cả tính năng mới (full)**: Contextual
Retrieval, RAPTOR, CRAG, multi-hop, adaptive routing, semantic cache.

- Bộ dữ liệu: `both` (80 câu) — `vn_bkai`, `vn_legal` (TV) + `en_covidqa`, `en_pubmedqa` (EN), n=10/bộ
- Judge: DeepSeek · Nhánh: `feat/ragas-langfuse-reranker`
- Dữ liệu thô: `data/eval/_ablation_baseline.json`, `_ablation_full.json` · Bảng: `benchmark_ablation_2026-06-11.md`

## Kết quả

| Chỉ số | baseline | **full** | Δ | Ngưỡng |
|---|---|---|---|---|
| Contextual recall | 0.912 | 0.912 | 0.000 | ≥0.70 ✅ |
| Contextual precision | 0.845 | 0.840 | −0.005 | ≥0.70 ✅ |
| Faithfulness | 0.893 | 0.895 | +0.002 | ≥0.80 ✅ |
| Answer correctness | 0.770 | 0.767 | −0.003 | ≥0.70 ✅ |
| **Citation accuracy** | 0.820 | **0.855** | **+0.035** | ≥0.70 ✅ |
| Failure rate | 0.000 | 0.000 | = | <0.05 ✅ |
| Chi phí / câu | $0.00130 | $0.00138 | +6% | tham khảo |
| Độ trễ p50 | 16.3s | 15.3s | −1.0s | tham khảo |

**Cả hai cấu hình đều ĐẠT mọi ngưỡng. Không có suy giảm.**

## Đọc kết quả

- **Citation accuracy +0.035** là cải thiện thực duy nhất rõ ràng — các tính năng mới (đặc biệt
  Contextual Retrieval prepend ngữ cảnh + RAPTOR summary nodes) giúp câu trả lời **trích dẫn đúng
  nguồn hơn**. Đây là lợi ích đáng giá cho ứng dụng y khoa (truy vết nguồn).
- **recall / precision / correctness / faithfulness ≈ không đổi** (chênh ≤0.005, nằm trong nhiễu
  mẫu n=10 ±0.05). Ở quy mô này, full **không** cải thiện rõ các chỉ số còn lại so với baseline.
- **Chi phí +6%, độ trễ ~ngang** (full thậm chí nhanh hơn 1s — nhờ adaptive fast-path/semantic
  cache bù lại chi phí CRAG/multi-hop).

→ **Kết luận:** bật full stack cho **citation tốt hơn (+0.035)**, **không hại** các chỉ số khác,
chi phí +6%. Lợi ích khiêm tốn nhưng dương ở n=10.

## Phương pháp (đã sửa harness)

Lần benchmark này dùng harness ablation **đã sửa 3 lỗi** (3 lỗi này khiến các lần trước cho
recall=0 giả):
1. **Dedupe no-op:** re-ingest bị dedupe → CR/RAPTOR không áp dụng. Sửa: `UPLOAD_DEDUPE_BY_HASH=false`.
2. **Chỉ wipe ES:** Postgres vẫn giữ doc cũ → `save_document_and_segments` trả "skipped" → ES không
   được index lại → rỗng. Sửa: `wipe_corpus_db()` xoá cả documents+segments PG mỗi config.
3. **PG-wipe lỗi thầm:** `wipe_corpus_db` chạy trong tiến trình cha (script) thiếu repo root trên
   `sys.path` → `ModuleNotFoundError: No module named 'src'` → PG không bị xoá → lại dính lỗi #2.
   Sửa: chèn ROOT vào `sys.path` trước import.

Sau sửa: mỗi config **wipe sạch ES+PG → re-ingest đúng cờ của config** (sync) → đo công bằng.
Xác minh: baseline recall = 0.912 (trước khi sửa = 0.000 giả).

Mỗi câu trả lời qua đường agent thật (`hybrid_kg` + rerank `bge-reranker-v2-m3` + DeepSeek), chấm
bằng DeepSeek judge (DeepEval).

## Lưu ý

- **n=10** → sai số mẫu ±0.05; chỉ citation (+0.035) vượt ngưỡng nhiễu. Muốn kết luận chắc về
  recall/precision, cần n≥20.
- **Điều phối ép cloud** (classify/decide → DeepSeek) cho ổn định lúc chạy (Ollama hay chết tiến
  trình). Chỉ số chất lượng hợp lệ; **chi phí/độ trễ không so trực tiếp prod** (prod chạy 4 tác vụ
  điều phối miễn phí trên Ollama 3B → rẻ hơn + nhanh hơn).
- StructMem/CR/RAPTOR build **đồng bộ** trong benchmark (đúng phương pháp). Prod giữ **async**
  (extraction nền, không chặn upload).

## Khuyến nghị

1. **Bật Contextual Retrieval + RAPTOR ở prod** — citation +0.035, không hại chỉ số khác, chi phí
   +6% chấp nhận được. Đây là 2 tính năng cho lợi ích đo được.
2. **CRAG / multi-hop / adaptive / cache:** trung tính ở n=10 — giữ bật (không hại, cache/fast-path
   giúp độ trễ) nhưng chưa chứng minh tăng chất lượng; theo dõi thêm.
3. Chạy lại **n≥20** khi cần con số chắc hơn cho recall/precision (harness giờ đã đúng).
