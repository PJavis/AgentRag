# Báo cáo A/B — Abstain-on-thin-context + hiệu chỉnh RETRIEVAL_RELEVANCE_FLOOR (19/06/2026)

Kiểm chứng `ANSWER_ABSTAIN_ON_THIN_CONTEXT` (HEAD `788bef4`): khi chunk tốt nhất có điểm rerank
< `RETRIEVAL_RELEVANCE_FLOOR` → giữ context nhưng lật prompt sang "không có thông tin liên quan →
từ chối, không suy từ kiến thức nền, không trích nguồn" + xoá citations để thành **từ chối sạch**.

- Cùng kho grouped+CR+RAPTOR (skip-ingest → chỉ khác flag, so sánh công bằng)
- 80 câu in-corpus + 15 câu out-of-corpus · judge **DeepSeek** (`deepseek-chat`) · 0 lỗi
- Dữ liệu thô: `data/eval/abstain_ab_A_off.json`, `data/eval/abstain_ab_B_on.json`

## ⚠️ Hai bug môi trường phát hiện khi chạy (không phải bug code)

1. **Ollama daemon tắt** → routing (`classify/decide/domain_router/followup` + `AGENT_MODEL`)
   trỏ `llama3.2:3b` qua Ollama `:11434` → mọi câu hỏi `APIConnectionError` (lần chạy 1: 79/80
   fail). Khắc phục: `ollama serve` (model `llama3.2:3b` đã có sẵn).
2. **Judge Gemini dính quota 429** (free-tier 20 req) → 3 metric LLM-judge (faithfulness,
   answer_correctness, citation_accuracy) trả `ERR:ClientError` toàn bộ 80 case; chỉ
   recall/precision lọt qua trước khi hết quota. Khắc phục: `--judge-provider deepseek` (đã verify
   `deepseek-chat` + `DeepSeekModel.generate` chạy).

→ Khuyến nghị: thêm preflight ping Ollama + judge endpoint trong harness, fail loud thay vì 80×
silent error.

## Kết quả A/B (floor=0.3, judge DeepSeek)

**Chất lượng in-corpus (n=80):**

| Chỉ số | A (abstain OFF) | B (abstain ON) | Δ |
|---|---|---|---|
| Contextual recall | 0.870 | 0.861 | −0.009 |
| Contextual precision | 0.696 | 0.697 | +0.001 |
| Faithfulness | 0.914 | 0.910 | −0.004 |
| Answer correctness | 0.771 | 0.705 | −0.066 |
| Citation accuracy | 0.820 | 0.761 | −0.059 |
| p50 latency | 19.8s | 20.2s | +0.4s |
| cost/query | $0.00154 | $0.00151 | — |

**An toàn out-of-corpus (n=15):**

| Chỉ số | A (OFF) | B (ON) |
|---|---|---|
| refusal_rate (từ chối sạch — lý tưởng) | 0.000 | **0.000** |
| hedged_cited_rate (rào nhưng trích distractor) | 0.533 | 0.667 |
| hallucination_rate (NGUY HIỂM) | 0.467 | 0.333 |
| counts (abstained / hedged / hallucinated) | 0 / 8 / 7 | 0 / 10 / 5 |

## ⚠️ Kết luận: ở floor=0.3, feature LÀ NO-OP — A/B chỉ đo nhiễu

`refusal_rate` vẫn **0.0** cả hai chiều. Probe trực tiếp (`scripts/eval/probe_thin_context.py`,
15 câu out-of-corpus) cho thấy nguyên nhân gốc:

```
out-of-corpus: _is_thin_context = False ở MỌI câu, max rerank_score ≈ 0.50 (0.5000–0.5048)
```

**Cross-encoder bge-reranker-v2-m3 (sigmoid) không bao giờ chấm dưới ~0.50** kể cả với nội dung
hoàn toàn lạc đề (logit≈0 → sigmoid 0.5). Floor 0.3 nằm DƯỚI toàn bộ dải điểm → `_is_thin_context`
không bao giờ True → prompt abstain + xoá citation **không bao giờ kích hoạt**.

Hệ quả: các câu trả lời "không có thông tin" trong config B là model **tự rào theo bản năng** (như
baseline), KHÔNG phải do feature. `answer_correctness` −0.066 là **nhiễu sinh-lại-câu-trả-lời +
judge variance** (n=80, GEval ±0.03–0.05 giữa các lần), không thể do feature vì feature không chạy.

→ **Cảnh báo:** kết luận "gate PHẢN TÁC DỤNG" (16/06) cũng đáng nghi — gate `apply_relevance_floor`
cùng dùng floor 0.3, mà không gì chấm dưới 0.3 → gate cũng inert → các delta của nó (recall −0.030,
hallucination +0.133) nhiều khả năng cũng là nhiễu, gán nhầm cho gate.

## Phát hiện chính: FLOOR MẶC ĐỊNH 0.3 BỊ HIỆU CHỈNH SAI

Probe in-corpus (`scripts/eval/probe_thin_incorpus.py`, 10 câu gold):

```
in-corpus (liên quan):   max rerank_score ≈ 0.73 (0.7262–0.7310)
out-of-corpus (lạc đề):  max rerank_score ≈ 0.50 (0.5000–0.5048)
```

→ **Tín hiệu phân tách rất sạch (bimodal):** khoảng trống 0.505 → 0.726. Floor đặt ở **bất kỳ đâu
trong (0.51, 0.72)** đều tách hoàn hảo. Chọn **floor = 0.6** (giữa khe, biên an toàn cả hai phía):

- out-of-corpus: max 0.50 < 0.6 → thin-context TRUE → abstain kích hoạt → **từ chối sạch**.
- in-corpus: max 0.73 > 0.6 → thin-context FALSE → no-op → **không mất chất lượng**.

## Phép thử THẬT: A/B abstain ở floor=0.6 (B' vs A, judge DeepSeek, 0 lỗi)

Dữ liệu thô: `data/eval/abstain_ab_B_floor06.json`. So với A cũ (A độc lập floor — floor chỉ gate
khi abstain ON).

**An toàn out-of-corpus (n=15):**

| Chỉ số | A (OFF) | B' (floor 0.6) |
|---|---|---|
| refusal_rate (từ chối sạch) | 0.000 | **0.467** |
| hedged_cited_rate | 0.533 | **0.000** |
| hallucination_rate | 0.467 | 0.533 |
| counts (abstained/hedged/halluc) | 0 / 8 / 7 | **7 / 0 / 8** |

**Chất lượng in-corpus (n=80):**

| Chỉ số | A (OFF) | B' (floor 0.6) | Δ |
|---|---|---|---|
| Contextual recall | 0.870 | 0.873 | +0.003 |
| Contextual precision | 0.696 | 0.699 | +0.003 |
| Faithfulness | 0.914 | 0.951 | +0.037 |
| Answer correctness | 0.771 | 0.721 | −0.050 |
| Citation accuracy | 0.820 | 0.788 | −0.032 |
| p50 latency | 19.8s | 17.7s | −2.1s |

### ✅ Kết luận: ĐẠT — feature kích hoạt đúng ở floor 0.6

- **refusal_rate 0 → 0.467**: 7/8 câu "hedged_cited" cũ chuyển thành **từ chối sạch** (đúng mục tiêu).
  hedged_cited 0.533 → **0** (xoá sạch).
- **In-corpus PHẲNG**: recall/precision/faithfulness đều phẳng-hoặc-tăng. `answer_correctness` −0.050
  là **NHIỄU sinh-lại-câu-trả-lời, KHÔNG phải feature**: (1) lần chạy floor=0.3 (feature inert) cũng
  cho 0.705 — chứng tỏ dao động ~0.05 là nhiễu generation; (2) điểm rerank in-corpus cụm chặt
  0.726–0.731 (≫ 0.6) → `_is_thin_context` **không thể** True trên câu in-corpus → feature về mặt cơ
  chế không chạm vào các câu này.
- **Caveat trung thực**: hallucination KHÔNG giảm (7→8, nhiễu ±1 trên n=15). 8 ca "bịa tự tin" phớt
  lờ prompt abstain (vốn không rào nên không có gì để chuyển). Đây là **bài toán riêng** → cần
  **harden prompt abstain** (hoặc hạ temperature khi thin-context) cho nhóm này. Feature KHÔNG làm
  tệ hơn.

## Áp dụng (đã commit vào config.py 19/06)

1. ✅ **`RETRIEVAL_RELEVANCE_FLOOR` 0.3 → 0.6** — 0.3 thấp hơn cả dải output cross-encoder → mọi tính
   năng dựa floor đều chết.
2. ✅ **`ANSWER_ABSTAIN_ON_THIN_CONTEXT` False → True** — bật mặc định (lợi ròng: 7 từ chối sạch +
   xoá hedged-cited, in-corpus không đổi).
3. Không re-test gate — abstain giữ context (an toàn hơn gate vốn xoá context → ép model bịa).

## Việc tiếp theo

1. **Harden prompt abstain** cho 8 ca bịa tự tin còn lại (prompt mạnh hơn / temp thấp khi thin) → A/B lại.
2. Trên kho VN-medical thật: **đo lại phân bố điểm rerank** (in-corpus vs off-corpus) — ngưỡng 0.6 có
   thể lệch theo domain; chỉnh floor theo khe phân tách đo được.

## Lệnh tái lập

```bash
ollama serve &   # bắt buộc: routing dùng llama3.2:3b
# probe phân bố điểm:
ANSWER_ABSTAIN_ON_THIN_CONTEXT=true PYTHONPATH=. uv run python scripts/eval/probe_thin_context.py
ANSWER_ABSTAIN_ON_THIN_CONTEXT=true PYTHONPATH=. uv run python scripts/eval/probe_thin_incorpus.py
# A/B ở floor đúng:
ANSWER_ABSTAIN_ON_THIN_CONTEXT=true RETRIEVAL_RELEVANCE_FLOOR=0.6 \
  uv run python scripts/eval/run_benchmark.py --suite both --n 20 --group-size 4 \
  --refusal-set data/eval/refusal_set.json --skip-ingest --judge-provider deepseek \
  --out data/eval/abstain_ab_B_floor06.json
```

Báo cáo trước: `benchmark_gate_ab_2026-06-16_vi.md` (gate — nay nghi cũng inert),
`benchmark_grouped_2026-06-15_vi.md`. Hướng dẫn: `docs/INSTRUCTION-abstain-thin-context-2026-06-16.md`.
