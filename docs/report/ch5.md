# CHƯƠNG 5. THỰC NGHIỆM VÀ ĐÁNH GIÁ

Chương này trình bày toàn bộ quá trình thực nghiệm và đánh giá hệ thống VITAL (AgentRag) — chatbot hỏi đáp tài liệu y tế tiếng Việt dựa trên kiến trúc RAG kết hợp agent. Khác với cách trình bày thực nghiệm thông thường (chạy benchmark một lần rồi báo cáo con số), quá trình đánh giá của đề tài được tổ chức như một chuỗi thí nghiệm có kiểm soát, trong đó bản thân **thước đo** cũng là một đối tượng phải kiểm chứng trước khi tin vào con số mà nó đưa ra. Triết lý xuyên suốt là "đo trước, sửa sau" (measure before fix): mọi quyết định bật/tắt tính năng, thay đổi ngưỡng an toàn hay đầu tư kỹ thuật đều phải dựa trên số liệu thực nghiệm đã được kiểm chứng độ tin cậy, chứ không dựa trên trực giác hay kết quả mẫu nhỏ.

Nội dung chương được tổ chức như sau: mục 5.1 mô tả môi trường thực nghiệm; mục 5.2 trình bày phương pháp đánh giá và lý do phải xây dựng bộ chấm điểm mới; mục 5.3 và 5.4 kiểm chứng độ tin cậy của thước đo; mục 5.5 trình bày kết quả trên corpus y tế thật; các mục 5.6–5.8 trình bày các thực nghiệm ablation kiến trúc, an toàn từ chối trả lời (abstain) và fine-tune mô hình embedding; mục 5.9 mô tả chiến dịch phân loại lỗi đang triển khai; mục 5.10 tổng kết các bài học phương pháp luận.

## 5.1. Môi trường thực nghiệm

### 5.1.1. Phần cứng

Toàn bộ thực nghiệm được tiến hành trên một máy trạm cá nhân chạy hệ điều hành Windows với môi trường **WSL2** (Windows Subsystem for Linux 2, kernel Linux 6.6). Cấu hình phần cứng chủ đạo:

| Thành phần | Cấu hình |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM |
| Trình điều khiển | CUDA 12.8 |
| Môi trường thực thi | WSL2 (Ubuntu), Python quản lý bằng `uv` |

Giới hạn 16 GB VRAM là một ràng buộc thực tế ảnh hưởng trực tiếp đến thiết kế thực nghiệm fine-tune (mục 5.8): trên tổ hợp WSL2 + GPU thế hệ mới + driver CUDA 12.8, chế độ huấn luyện mixed-precision (amp/fp16) gây lỗi `CUDA device not ready`, buộc phải huấn luyện ở chế độ **fp32** với batch size 8 và độ dài chuỗi tối đa 512 token. Đây cũng là lý do mô hình embedding nền được chọn là `intfloat/multilingual-e5-base` (fp32 chiếm khoảng 10 GB VRAM, nằm trong ngân sách 14 GB) thay vì `bge-m3` đang dùng ở production (fp32 chạm trần ~16 GB).

### 5.1.2. Stack dịch vụ

Hệ thống được triển khai dưới dạng tập hợp dịch vụ chạy cục bộ (self-host), phản ánh đúng kịch bản triển khai on-premise cho môi trường y tế (dữ liệu bệnh án không rời khỏi hạ tầng nội bộ):

| Dịch vụ | Vai trò | Ghi chú |
|---|---|---|
| Elasticsearch (`:9200`) | Chỉ mục tìm kiếm lai (hybrid: BM25 + vector) | trạng thái cluster green/yellow là điều kiện tiên quyết của mọi lần benchmark |
| PostgreSQL | Lưu trữ tài liệu, segment, hội thoại, phản hồi người dùng | bảng `adapter_chat_feedback` phục vụ vòng lặp phản hồi |
| Redis | Cache | |
| Ollama | Phục vụ mô hình embedding cục bộ giai đoạn đầu (`nomic-embed-text`) | |
| TEI (`:8080`) | Text Embeddings Inference — phục vụ mô hình embedding fine-tune `agentrag-embed-v1` (768 chiều) | thay thế đường embedding cũ từ 2026-07 |
| Langfuse (`:3002`) | Quan trắc (observability): mỗi lượt `/chat` là một trace, phản hồi 👍/👎 gắn vào trace dưới dạng score `user_feedback` | self-host |
| Reranker cục bộ | Cross-encoder `BAAI/bge-reranker-v2-m3` chạy trên GPU | backend `local_cross_encoder` — backend duy nhất phát ra `rerank_score` |

Một chi tiết vận hành quan trọng: script benchmark (`run_benchmark.py`) có bước **preflight** kiểm tra Elasticsearch/embedding/judge trước khi chạy, và fail-fast nếu stack chưa lên đủ — kinh nghiệm rút ra từ các lần chạy hỏng do một dịch vụ chết giữa chừng.

### 5.1.3. Cấu hình mô hình

Hệ thống định tuyến mô hình theo tác vụ thông qua biến cấu hình `LLM_TASK_MODEL_MAP`, tách bạch vai trò "trả lời" khỏi các vai trò "chấm điểm" và "sinh đáp án chuẩn":

| Vai trò (slot) | Mô hình | Ghi chú |
|---|---|---|
| `answer` (sinh câu trả lời) | **deepseek-v4-flash** | quyết định giữ flash được xác nhận bằng A/B ở mục 5.6.2 |
| `decide`/HyDE (agent quyết định, viết lại truy vấn) | gemini-2.5-flash-lite (giai đoạn đầu) | có bộ phân tích JSON khoan dung (lenient parse) để chịu được output lỗi định dạng |
| `eval_judge` (giám khảo chính) | gemini-2.5-flash → gemini-2.5-pro (khóa trả phí, từ 2026-07-13) | |
| `eval_judge2` (giám khảo thứ hai) | gemini-2.5-pro / deepseek-v4-pro (tùy giai đoạn) | dùng để đo nhiễu judge |
| `oracle_gen` / `gold_gen` (sinh câu trả lời oracle và đáp án chuẩn) | gemini-2.5-pro hoặc deepseek-v4-pro | mô hình mạnh + ngữ cảnh vàng |
| Embedding | bge-m3 (1024 chiều) → `agentrag-embed-v1` (e5 fine-tune, 768 chiều, TEI) | chuyển đổi kèm re-ingest toàn bộ corpus |
| Reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder cục bộ, đầu ra sigmoid ∈ [0,1]) | nguồn của `rerank_score` điều khiển toàn bộ cơ chế abstain |

Các tham số an toàn liên quan trực tiếp đến thực nghiệm: `RETRIEVAL_RELEVANCE_FLOOR=0.55` (ngưỡng sàn điểm rerank, mục 5.7), `RETRIEVAL_INCLUDE_RAW_QUERY=true` (tiêm kết quả truy hồi của câu hỏi gốc vào pool rerank, mục 5.3), `AGENT_TOTAL_TIMEOUT_S=90` và `LLM_REQUEST_TIMEOUT_S=60` (ngân sách thời gian, khắc phục hiện tượng treo 42 phút quan sát được khi Gemini trả lỗi 503 hàng loạt).

### 5.1.4. Bộ dữ liệu

Thực nghiệm sử dụng ba nhóm dữ liệu:

**(1) Corpus y tế thật (`data/originals`).** Tập 114 tệp PDF tài liệu y tế tiếng Việt. Ở giai đoạn khai thác dữ liệu huấn luyện (2026-06-27), 101/114 PDF được nạp chỉ mục ở chế độ text-only, sinh khoảng 2828 segment trên Elasticsearch. Ở cấu hình production hiện hành (2026-07-13, chế độ ingest "lean": tắt Contextual Retrieval, RAPTOR, StructMem và vision), corpus gồm **115 tài liệu / 3359 segment**. Đây là corpus dùng cho các probe c2 ở mục 5.5.

**(2) Bộ đánh giá tổng hợp sinh từ corpus (synthetic eval set).** Công cụ `scripts/eval/build_prod_evalset.py` sinh câu hỏi tổng hợp bằng LLM trực tiếp trên nội dung `Segment.content` của chỉ mục đang chạy (không cần re-ingest), kèm theo: (i) segment nguồn làm ngữ cảnh vàng (gold context), và (ii) đáp án chuẩn (gold answer) do LLM viết dựa trên ngữ cảnh vàng. Các bộ cụ thể: `prod_corpus_evalset.jsonl` (v1, n=30 yêu cầu), `prod_corpus_evalset_v2.jsonl`, `prod_corpus_evalset_v3.jsonl` (n=50) — sinh trên corpus "residue" (phần còn lại của các bộ công khai đã nạp chỉ mục); và `c2_evalset_n40.jsonl` (n=40, có tùy chọn 12 câu multi-hop) — sinh trên corpus y tế thật. Một quy tắc hiệu lực quan trọng rút ra từ thực nghiệm: **một bộ eval chỉ có giá trị đối với đúng snapshot corpus mà nó được sinh ra** — bộ v3 (residue) khi chấm trên corpus y tế thật cho sys=0.00 ở mọi câu (kiểm chứng 2026-07-13), dẫn tới cơ chế "corpus fingerprint" ở mục 5.9.

**(3) Các bộ công khai và bộ từ chối.** Hai bộ tiếng Việt công khai `vn_bkai` và `vn_legal` (mỗi bộ lấy n=40, tổng 80 câu/cấu hình) dùng cho ablation kiến trúc (mục 5.6); bộ `refusal_set.json` gồm 15 câu hỏi bịa đặt/ngoài corpus (thuốc, bệnh, chủ đề phần mềm không tồn tại trong chỉ mục) dùng đánh giá an toàn từ chối trả lời (mục 5.7).

## 5.2. Phương pháp đánh giá

### 5.2.1. Vì sao RAGAS `answer_correctness` bị "trần" ở ~0.74

Giai đoạn đầu, hệ thống được đánh giá bằng bộ metric chuẩn của framework RAGAS, trong đó metric trung tâm là `answer_correctness` — một dạng **claim-F1**: giám khảo LLM tách câu trả lời của hệ thống và đáp án chuẩn (gold) thành các mệnh đề nguyên tử (claim), rồi tính F1 giữa hai tập mệnh đề. Về cơ chế:

- **False Positive (FP):** mệnh đề có trong câu trả lời nhưng không có trong gold → trừ điểm precision. Hệ quả nghịch lý: nếu hệ thống trả lời **đúng và đầy đủ hơn gold** (bổ sung thông tin đúng lấy từ tài liệu), mỗi thông tin đúng-nhưng-thừa đó vẫn bị tính là FP, vì gold tổng hợp của các bộ công khai rất **ngắn gọn** (terse).
- **False Negative (FN):** mệnh đề có trong gold nhưng giám khảo không khớp được với câu trả lời → trừ điểm recall. Với tiếng Việt, cùng một nội dung có thể diễn đạt lại (paraphrase) rất khác về mặt từ vựng; giám khảo thường không khớp được biến thể diễn đạt hợp lệ, sinh FN giả.

Hai cơ chế này cộng lại tạo một **trần điểm nhân tạo**: câu trả lời càng tự nhiên, càng đầy đủ thì càng dễ mất điểm, và điểm số bão hòa quanh 0.72–0.74 bất kể chất lượng thật. Bằng chứng thực nghiệm cho chẩn đoán này gồm hai thí nghiệm độc lập (chi tiết ở mục 5.6): (i) toàn bộ các can thiệp kiến trúc truy hồi (CR/RAPTOR/CRAG/multi-hop/cache) không dịch chuyển được correctness; (ii) thay mô hình trả lời từ deepseek-v4-flash lên deepseek-v4-pro (mạnh gấp đôi) chỉ tăng correctness **+0.006** — trong khi metric `faithfulness` (độ trung thực với ngữ cảnh, **không cần tham chiếu gold**) lại tăng rõ rệt +0.040 (0.931→0.971). Khi hệ thống chứng minh được là có phản hồi với mô hình tốt hơn (faithfulness tăng) mà metric tham chiếu gold không nhúc nhích, kết luận hợp lý là **nút thắt nằm ở thước đo (gold + giám khảo), không nằm ở hệ thống**.

### 5.2.2. Thiết kế bộ chấm ensemble (nugget recall + rubric)

Từ chẩn đoán trên, đề tài xây dựng bộ chấm correctness mới (`src/agentrag/eval/correctness_judge.py`) gồm hai thành phần, lấy trung bình:

1. **Nugget recall:** giám khảo LLM trích các "nugget" — đơn vị thông tin cốt lõi — từ đáp án chuẩn, rồi kiểm tra từng nugget có được câu trả lời của hệ thống bao phủ hay không (chấp nhận diễn đạt lại). Cách này đo phần *recall* của nội dung thiết yếu mà **không phạt thông tin đúng-nhưng-thừa** — khắc phục trực tiếp cơ chế FP của claim-F1.
2. **Rubric có tham chiếu (reference-guided rubric):** giám khảo chấm câu trả lời theo thang tiêu chí tổng thể (đúng/đủ/chính xác so với gold), cho phép ghi nhận chất lượng diễn đạt và mức độ hoàn chỉnh thay vì đếm mệnh đề cơ học.

Bộ chấm được chạy **song song trên hai mô hình giám khảo khác nhau** (judge1, judge2), định tuyến được qua slot `task` trong cấu hình. Điểm hệ thống báo cáo là điểm ensemble; tương quan giữa hai giám khảo là chỉ báo độ tin cậy (mục 5.2.4).

### 5.2.3. Oracle probe — tách "trần thước đo" khỏi "dư địa hệ thống"

Câu hỏi trung tâm của mọi kết quả đánh giá là: *điểm chưa đạt tối đa là do hệ thống còn kém, hay do thước đo không thể cho điểm cao hơn?* Công cụ trả lời là **oracle probe** (`scripts/eval/oracle_probe.py`), với thiết kế:

- **System:** agent thật, chạy đầy đủ pipeline truy hồi → rerank → sinh câu trả lời, chấm bằng bộ ensemble.
- **Oracle:** cùng bộ câu hỏi, nhưng câu trả lời được sinh bởi **mô hình mạnh nhất có sẵn được cung cấp thẳng ngữ cảnh vàng** — tức mô phỏng "truy hồi hoàn hảo + bộ sinh mạnh". Điểm oracle là **cận trên thực nghiệm** của những gì thước đo có thể ghi nhận trên bộ câu hỏi đó.

Đại lượng chẩn đoán là hiệu **oracle − system**:

- Nếu **oracle − system nhỏ** (theo quy ước của đề tài: < ~0.05, cỡ nhiễu đo): truy hồi hoàn hảo + bộ sinh mạnh cũng chỉ hơn hệ thống thật không đáng kể → phần điểm còn thiếu nằm ở **gold/thước đo**, việc tiếp tục tinh chỉnh hệ thống là vô ích vì không thể đo được lợi ích.
- Nếu **oracle − system lớn**: hệ thống thật đang **để rơi điểm thật** so với trần khả thi → còn dư địa cải tiến truy hồi/sinh, và dư địa đó **đo được** (system-bound).

Một hệ quả phương pháp luận quan trọng: oracle probe cho phép **định giá trước** một hạng mục đầu tư. Ví dụ ở mục 5.8, thay vì thực hiện cuộc di trú embedding tốn kém (đổi chiều vector 1024→768, re-index, re-ingest toàn bộ) chỉ để đo xem fine-tune có giúp gì cho correctness, ta đo oracle − system trước: nếu chỉ +0.030 thì *mọi* cải tiến truy hồi, kể cả fine-tune, tối đa thu về +0.030 — dưới ngưỡng nhiễu, không đáng làm.

### 5.2.4. Nhiễu giám khảo và hệ số tương quan Pearson

Vì giám khảo là LLM (LLM-as-judge), bản thân điểm số có nhiễu và có thể có thiên lệch. Đề tài dùng **hệ số tương quan Pearson** giữa điểm của hai giám khảo độc lập (judge1, judge2) trên cùng tập câu trả lời làm "sàn nhiễu giám khảo" (judge-noise floor). Hệ số Pearson đo mức độ tương quan tuyến tính giữa hai dãy số, nhận giá trị trong [−1, 1]: bằng 1 nghĩa là hai giám khảo hoàn toàn đồng thuận về thứ tự và tỷ lệ điểm; gần 0 nghĩa là điểm của giám khảo này không nói lên gì về điểm của giám khảo kia (điểm số vô nghĩa). Quy tắc đọc: **pearson thấp → phải sửa giám khảo trước khi tin bất kỳ con số correctness nào**; pearson cao → con số ổn định theo giám khảo.

Ngoài nhiễu, còn phải kiểm soát **thiên lệch tự ưu ái (self-preference bias)**: nếu giám khảo cùng nhà cung cấp/cùng họ mô hình với mô hình trả lời (ví dụ giám khảo DeepSeek chấm câu trả lời do DeepSeek sinh), điểm có thể bị thổi phồng, và pearson giữa hai giám khảo cùng họ (deepseek-flash vs deepseek-pro) chỉ là ước lượng *lạc quan* của độ đồng thuận. Kiểm chứng chéo nhà cung cấp (cross-provider: Gemini vs DeepSeek) được trình bày ở mục 5.4.

## 5.3. Kết quả kiểm chứng thước đo — chuỗi probe v1/v2/v3

Ba lần chạy oracle probe liên tiếp trên bộ eval sinh từ corpus đã nạp chỉ mục (thời điểm này là corpus "residue" — phần dữ liệu tiếng Việt hỗn hợp còn lại trong chỉ mục: xe máy, luật chứng khoán/tài chính, khoảng 176 segment trong đó 134 dùng được — chứ chưa phải corpus y tế) vừa kiểm chứng thước đo mới, vừa phát hiện và sửa một lỗi hệ thống thật.

| Probe | n hiệu dụng | System avg | Oracle avg | Oracle − System | Pearson (judge1, judge2) | Giám khảo |
|---|---|---|---|---|---|---|
| v1 (trước sửa lỗi) | 26/30 (4 câu skip do Gemini 503) | 0.842 | 0.976 | **+0.134** | 0.965 | gemini-2.5-flash / gemini-2.5-pro |
| v2 (sau sửa lỗi) | 20/30 (10 câu skip do 503) | 0.950 | 0.969 | **+0.019** | 0.962 | gemini (như v1) |
| v3 (chạy sạch) | 50/50 (0 skip) | **0.888** | 0.934 | **+0.046** | 0.730 | deepseek-flash / deepseek-pro |

### 5.3.1. Probe v1 — thước đo mới phá trần 0.74 và lộ ra đuôi lỗi thật

Kết quả v1 (26 câu chấm được) xác lập ba điều:

1. **Thước đo không bị trần.** Oracle đạt **0.976** — bộ chấm ensemble ghi nhận một câu trả lời đúng và đầy đủ ở mức gần tuyệt đối, khác hẳn trần 0.74 của claim-F1. Vậy mức bão hòa 0.74 trước đây là giới hạn của *metric cũ*, không phải giới hạn phổ quát của đánh giá bằng LLM.
2. **Thước đo đáng tin.** Hai giám khảo (gemini-flash vs gemini-pro) đồng thuận ở pearson **0.965** — điểm số ổn định theo giám khảo, không phải nhiễu.
3. **Thước đo phân giải được tín hiệu hệ thống thật.** Khoảng cách oracle − system = **+0.134** không phải trần đồng đều mà tập trung ở một đuôi lỗi nhỏ: phân bố điểm system của 26 câu là `17× 1.00, 2× 0.85, 0.83, 0.78, 0.70, 0.64, 0.25, 2× 0.00` — tức 3/26 câu hỏng nặng (hai câu 0.00, một câu 0.25) trong khi 17/26 câu đạt tuyệt đối. Thước đo cũ không thể làm lộ đuôi lỗi này; thước đo mới làm được, và đuôi lỗi là thứ hành động được.

**Truy nguyên 3 câu hỏng (rows 11/18/21).** Điều tra theo từng ranh giới thành phần cho kết quả bất ngờ: *truy hồi không có lỗi* (chunk vàng đứng hạng 0 ở cả ba câu), *sinh câu trả lời không có lỗi* (khi không bị chặn, agent trả lời đúng). Thủ phạm là **cổng abstain ngữ-cảnh-mỏng hoạt động chập chờn**: hàm `_is_thin_context` từ chối trả lời khi `max(rerank_score) < RETRIEVAL_RELEVANCE_FLOOR (0.6)`. Cross-encoder bge chấm chunk tiếng Việt liên-quan-nhưng-diễn-đạt-lại ở mức ~0.61–0.73 và mọi chunk khác ở mức phẳng 0.5; agent lại phát câu hỏi gốc kèm 3 câu viết lại (rewrite) bằng LLM không tất định, nên `max(rerank_score)` dao động **ngay quanh ngưỡng 0.6** giữa các lần chạy: có lần chunk tốt được chấm 0.619 → trả lời đúng; lần khác tụt dưới 0.6 → từ chối ("Tài liệu hiện có không có thông tin để trả lời câu hỏi này") → giám khảo chấm 0. Đây là hiện tượng **từ chối nhầm chập chờn (flaky false-abstention)** tại biên ngưỡng — không phải lỗi truy hồi, không phải lỗi sinh.

### 5.3.2. Chuỗi sửa lỗi

Ba sửa đổi được áp dụng, mỗi sửa nhắm một nguồn dao động (kèm ràng buộc an toàn: hạ ngưỡng không được làm hỏng khả năng từ chối câu hỏi ngoài corpus):

1. **Hạ sàn 0.6 → 0.55** (commit `f5dfe76`): 0.55 nằm giữa khoảng trống phân bố (ngoài corpus ~0.50 | sàn | trong corpus ~0.61+), tạo biên an toàn cho các chunk liên quan bị diễn đạt lại.
2. **Viết lại truy vấn tất định** (temperature 0 xuyên suốt `json_response`): cùng một câu hỏi không còn cho các bộ rewrite khác nhau giữa các lần chạy.
3. **Tiêm truy hồi câu hỏi gốc vào pool rerank** (commit `c706d6c`, `RETRIEVAL_INCLUDE_RAW_QUERY=true`, top_k=8): điều tra bổ sung cho thấy sau sửa (1) và (2), row 21 *vẫn* từ chối dù truy hồi bằng chính câu hỏi gốc cho điểm rerank tối đa 0.716 ổn định qua 3 lần chạy — nguyên nhân tồn dư là bước decide của agent sinh các truy vấn con/viết lại **kéo về chunk kém hơn câu hỏi gốc**. Sửa đổi này bảo đảm các rewrite chỉ có thể **bổ sung** ứng viên chứ không bao giờ làm rơi chunk tốt nhất xuống dưới sàn. Kiểm chứng trực tiếp: row 18 và row 21 chuyển từ từ chối sang **trả lời đúng**; các câu thật sự ngoài corpus ("thủ đô nước Pháp", "tổng thống Mỹ đầu tiên") **vẫn từ chối** ở sàn 0.55 — an toàn không suy giảm.

Kèm theo là hai sửa vận hành: timeout 60 giây mỗi lời gọi LLM và ngân sách tổng `AGENT_TOTAL_TIMEOUT_S=90` cho cả vòng `agent.chat` (commit `dcbe196`) — chặn hiện tượng treo 42 phút quan sát được dưới bão lỗi Gemini 503.

### 5.3.3. Probe v2 — xác nhận sửa lỗi, nhưng con số tuyệt đối bị nhiễu chọn mẫu

Chạy lại sau sửa lỗi (v2, 20/30 câu chấm được vì 10 câu bị skip do Gemini 503 quá tải nặng): system tăng **0.842 → 0.950** (+0.108), oracle − system sụp từ +0.134 xuống **+0.019** (về mức nhiễu), **0 câu hỏng nặng** (trước là 3) — phân bố `16× 1.00, 0.92, 0.85, 0.70, 0.53`, điểm thấp nhất là câu trả lời một phần (0.53) chứ không còn từ chối nhầm. Pearson 0.962 — thước đo vẫn đáng tin.

Tuy nhiên v2 mang một khuyết tật thống kê được nhận diện ngay trong báo cáo: 10/30 câu bị loại vì lỗi 503 tạo ra **thiên lệch chọn mẫu (selection bias)** — các câu "sống sót" thiên về câu dễ — cộng thêm khác giám khảo, nên con số 0.950 **không được lấy làm số chính thức** mà chỉ dùng để đọc tác động *tương đối* của chuỗi sửa lỗi (từ chối nhầm đã hết, an toàn ngoài corpus giữ nguyên).

### 5.3.4. Probe v3 — chạy sạch n=50: trần-do-thước-đo được XÁC NHẬN trên corpus residue

Lần chạy sạch (n=50, 0 skip, giám khảo all-DeepSeek do Gemini free-tier không phục vụ gemini-2.5-pro) cho: system **0.888**, oracle **0.934**, oracle − system **+0.046**, pearson **0.730**. Hai phát hiện:

1. **Kết luận "trần là thước đo" đứng vững ở mẫu lớn gấp 2.5 lần:** +0.046 < 0.05 — truy hồi hoàn hảo cộng bộ sinh mạnh cũng chỉ hơn hệ thống thật trong phạm vi nhiễu. Trên corpus residue này, hệ thống đã chạm trần đo được; tiếp tục tinh chỉnh truy hồi/sinh sẽ không hiện ra trên thước đo.
2. **Con số 0.950 của v2 là lạc quan; con số sạch là 0.888.** Chênh lệch đến từ thiên lệch chọn mẫu 503 của v2 cộng khác giám khảo. Bài học lặp lại của toàn bộ chương (trùng với ablation CR+RAPTOR ở mục 5.6): **số liệu mẫu nhỏ chạy "nóng"; chỉ tin số ở n lớn, chạy sạch.**

Hai giới hạn được ghi nhận trung thực: (i) pearson 0.730 là số đồng thuận giữa hai giám khảo *cùng họ* DeepSeek (flash vs pro) — trường hợp lạc quan — và giám khảo cùng nhà cung cấp với mô hình trả lời (deepseek-v4-flash) nên 0.888 có nguy cơ tự ưu ái nhẹ; cần giám khảo trả phí khác nhà cung cấp trước khi trích dẫn con số correctness một cách tự tin (giải quyết ở mục 5.4); (ii) corpus vẫn là residue `vn_bkai`/`vn_legal`, chưa phải corpus y tế thật (giải quyết ở mục 5.5).

## 5.4. Kiểm chứng giám khảo độc lập (cross-provider)

Vấn đề tồn đọng từ v3: toàn bộ chuỗi số dựa trên giám khảo DeepSeek — cùng nhà cung cấp với mô hình trả lời. Nếu có tự ưu ái, mọi con số hệ thống đều bị thổi phồng một cách hệ thống; và pearson 0.730 (deepseek-flash vs deepseek-pro) chưa phải phép đo đồng thuận thật vì hai giám khảo cùng "gia đình" dễ chia sẻ thiên lệch. Việc kiểm chứng đòi hỏi khóa Gemini trả phí (free-tier phục vụ gemini-2.5-pro với hạn mức 0 và giới hạn 5 request/phút với flash — đã kiểm chứng thực nghiệm là không thể chạy probe đa-lời-gọi).

Ngày 2026-07-13, lần chạy đầu tiên với khóa Gemini trả phí được thực hiện trên bộ `c2_evalset_n40.jsonl` (corpus y tế thật): `eval_judge=gemini-2.5-pro` (giám khảo chính, **độc lập** với mô hình trả lời DeepSeek), `eval_judge2=deepseek-v4-pro`. Kết quả (`docs/eval/c2_probe_n40_gemini-judge.md`):

| Chỉ số | Giá trị | Ý nghĩa |
|---|---|---|
| Pearson (gemini-2.5-pro vs deepseek-v4-pro) | **0.921** | đồng thuận chéo nhà cung cấp thật sự, so với 0.730 cùng-họ ở v3 |
| System avg (giám khảo gemini) | **0.759** | |
| System avg (giám khảo deepseek, cùng bộ eval — probe FT) | 0.764 | chênh 0.005, trong phạm vi nhiễu |
| Oracle − system | +0.088 | dư địa thật trên corpus thật (mục 5.5) |

Hai kết luận:

1. **Khoảng trống độc-lập-giám-khảo được ĐÓNG.** Một giám khảo Gemini và một giám khảo DeepSeek chấm cùng tập câu trả lời đồng thuận ở pearson 0.921 — nghĩa là lịch sử số liệu chấm bằng DeepSeek (baseline 0.813, FT 0.764 ở mục 5.5) **không** bị thổi phồng bởi tự ưu ái. Các con số correctness từ rig này từ nay trích dẫn được mà không cần caveat giám khảo phụ thuộc.
2. **Con số bền vững theo lựa chọn giám khảo:** 0.759 (gemini chấm) ≈ 0.764 (deepseek chấm) trên cùng bộ eval — chênh lệch nằm trong nhiễu. Đây là dạng kiểm chứng "đổi thước mà số không đổi" — điều kiện cần để một con số đo lường có ý nghĩa khoa học.

## 5.5. Kết quả trên corpus y tế thật (c2, n=40)

Sau khi thước đo được kiểm chứng (mục 5.3) và giám khảo được xác nhận độc lập (mục 5.4), câu hỏi trung tâm được trả lời trên đúng đối tượng: **corpus y tế thật** (115 tài liệu / 3359 segment, cấu hình lean, mô hình trả lời deepseek-v4-flash). Ba lần probe trên cùng bộ `c2_evalset_n40.jsonl` (n=40):

| Probe | Hệ thống đo | Giám khảo chính | System avg | Oracle avg | Oracle − System | Pearson |
|---|---|---|---|---|---|---|
| c2 baseline | embedding bge-m3 | deepseek | 0.813 | 0.892 | **+0.080** | 0.943 |
| c2 FT | embedding e5-FT (`agentrag-embed-v1`) | deepseek | 0.764 | 0.926 | **+0.162** | 0.937 |
| c2 gemini-judge | embedding e5-FT (prod hiện hành) | **gemini-2.5-pro** (độc lập) | 0.759 | 0.847 | **+0.088** | 0.921 |

Diễn giải:

**Thứ nhất — corpus thật là system-bound, ngược với corpus residue.** Trên residue, oracle − system = +0.046 (< 0.05, trần thước đo); trên corpus y tế thật, khoảng cách nhất quán ở mức **+0.080 đến +0.088** ở hai probe đáng tin nhất (baseline deepseek-judged và gemini-judged), **vượt ngưỡng nhiễu ~0.05**. Nghĩa là trên tài liệu y tế thật, hệ thống còn **dư địa cải thiện thật khoảng ~0.09** mà truy hồi/sinh tốt hơn có thể thu về — khác về bản chất với residue nơi mọi cải tiến đều "vô hình" trước thước đo. Phân tích mức câu cho thấy phần điểm mất **tập trung ở khoảng 5/40 câu hỏng hẳn (sys = 0.00)** chứ không rải đều — đây chính là căn cứ định lượng khởi động chiến dịch phân loại lỗi ở mục 5.9 (đặt tên cho từng câu hỏng thay vì tinh chỉnh mù).

**Thứ hai — oracle không phải hằng số: dao động 0.847–0.926 trên cùng bộ câu hỏi.** Ba lần chạy oracle trên cùng eval set cho 0.892 / 0.926 / 0.847 — biên độ ±0.04. Nguyên nhân khả dĩ gồm nhiễu sinh (oracle_gen không tất định tuyệt đối) và khác giám khảo giữa các lần. Hệ quả phương pháp luận: **không so trực tiếp các hiệu số giữa hai probe khác lần chạy** (ví dụ gap +0.162 của probe FT so với +0.080 của baseline không đủ căn cứ kết luận FT làm hệ thống tệ đi — chênh lệch system 0.813 vs 0.764 cũng nằm sát biên nhiễu khi oracle cùng bộ dao động ±0.04). Tín hiệu đáng tin là tín hiệu **lặp lại qua nhiều lần đo**: gap ~+0.08–0.09 xuất hiện ở hai probe độc lập về giám khảo, còn giá trị +0.162 là quan sát đơn lẻ chưa tái lập. Mọi so sánh A/B về sau (mục 5.9) do đó được thiết kế chạy **cùng giám khảo, cùng eval set, cùng đợt**.

**Thứ ba — con số chính thức hiện hành của hệ thống trên corpus y tế:** correctness ensemble **0.759** (giám khảo độc lập gemini-2.5-pro, pearson chéo nhà cung cấp 0.921), oracle 0.847, dư địa +0.088. Lần chạy này đồng thời được thiết kế đóng vai "nhánh CR-off" của phép A/B CR+RAPTOR trên corpus thật còn treo từ mục 5.6 (tái sử dụng đúng judge map + eval set cho nhánh CR-on sau này).

## 5.6. Thực nghiệm ablation kiến trúc

Ablation (loại trừ từng thành phần để đo đóng góp riêng) được tiến hành trên hai trục: kiến trúc truy hồi và mô hình trả lời. Cả hai dùng bộ công khai `vn` (vn_bkai + vn_legal), giám khảo deepseek, các metric RAGAS truyền thống (giai đoạn này thước đo mới chưa ra đời — chính chuỗi ablation này là bằng chứng dẫn tới việc xây thước đo mới).

### 5.6.1. CR + RAPTOR: cú lừa của mẫu nhỏ

Contextual Retrieval (CR — bổ sung ngữ cảnh tài liệu vào từng chunk khi nạp chỉ mục) và RAPTOR (cây tóm tắt phân cấp) là hai kỹ thuật nâng cấp truy hồi nặng chi phí ingest. Ở các lần chạy thăm dò n=10 (20 câu/cấu hình), CR+RAPTOR có vẻ tăng contextual_precision (độ chính xác ngữ cảnh — tỷ lệ đoạn được truy hồi thật sự liên quan) tới +0.034/+0.041. Lần chạy xác nhận n=40 mỗi bộ (**80 câu/cấu hình**, dải nhiễu ≈ ±0.03 so với ±0.06 ở n=20) cho bức tranh khác hẳn:

| Cấu hình | contextual_recall | contextual_precision | faithfulness | answer_correctness | citation_accuracy | latency p50 (ms) | cost/query (USD) | failure_rate |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.854 | 0.780 | 0.943 | 0.723 | 0.844 | 19725.3 | 0.002 | 0.000 |
| cr_raptor | 0.858 | 0.794 | 0.904 | 0.726 | 0.846 | 20625.0 | 0.002 | 0.000 |

Hiệu số cr_raptor − baseline: recall +0.004, precision **+0.014**, faithfulness **−0.039**, correctness +0.003, citation +0.002, latency +0.9 giây. Đối chiếu ba lần chạy:

| Lần chạy | Cỡ mẫu | Δ precision | Δ faithfulness |
|---|---|---|---|
| ma trận gs=0 | n=10 (20 câu) | +0.034 | −0.031 |
| re-bench gs=8 | n=10 (20 câu) | +0.041 | +0.078 |
| **gs=8 n=40** | **80 câu** | **+0.014** | **−0.039** |

Diễn giải: mức tăng precision +0.04 ở mẫu nhỏ là **ảo ảnh thống kê (small-sample artifact)** — ở mẫu quyết định (80 câu), delta sụp về +0.014 (trong dải nhiễu), trong khi faithfulness **thấp hơn baseline 0.039** (các chunk được contextual hóa của CR và các nút tóm tắt của RAPTOR đưa vào ngữ cảnh chất liệu kém bám nguồn hơn — đáng lo với hệ thống y tế), đổi lại một chi phí ingest nặng (~4 giờ/lần ingest). Khuyến nghị của báo cáo ablation là tắt CR+RAPTOR; tuy nhiên vì toàn bộ bằng chứng nằm trên bộ tổng hợp công khai dày distractor (khó hơn production), quyết định cuối là **không lật cờ chỉ dựa trên bằng chứng synthetic** — giữ nguyên hiện trạng và phân xử bằng A/B trên corpus thật (nhánh CR-off đã chạy ở mục 5.5; ở cấu hình lean hiện hành CR/RAPTOR đã tắt khi ingest corpus y tế). Các cờ query-time còn lại (CRAG, fast-path, semantic-cache, multi-hop) đều không cho mức tăng vượt nhiễu ở ablation T6 → giữ OFF mặc định.

### 5.6.2. Mô hình trả lời: flash vs pro

Thí nghiệm A/B cô lập đóng góp của riêng mô hình sinh câu trả lời: cùng chỉ mục (skip-ingest, truy hồi giống hệt), cùng bộ vn n=40 (80 câu), chỉ khác slot `answer`:

| Nhánh | contextual_recall | contextual_precision | faithfulness | answer_correctness | latency p50 | cost/query |
|---|---|---|---|---|---|---|
| answer=deepseek-v4-flash | 0.858 | 0.805 | 0.931 | 0.734 | 18.4 s | $0.00172 |
| answer=deepseek-v4-pro | 0.858 | 0.803 | 0.971 | 0.740 | 34.1 s | $0.00194 |
| **Δ (pro − flash)** | 0.000 | −0.002 | **+0.040** | **+0.006** | **+85%** | +12% |

Diễn giải: mô hình trả lời lớn/chậm gấp đôi chỉ dịch correctness **+0.006 (nhiễu)** — mô hình trả lời **không phải** đòn bẩy correctness. Nó nâng faithfulness +0.040 (0.931→0.971 — pro càng ít bịa hơn nữa) nhưng trả giá gần gấp đôi độ trễ (18.4→34.1 giây) và +12% chi phí. **Quyết định: giữ `answer=deepseek-v4-flash`** — mức faithfulness 0.93 của flash đã đủ mạnh cho bài toán, còn độ trễ 34 giây là không chấp nhận được về trải nghiệm.

Ý nghĩa lớn hơn của cặp thí nghiệm này đã nêu ở mục 5.2.1: hai đòn bẩy độc lập và mạnh (kiến trúc truy hồi, chất lượng bộ sinh) **cùng** thất bại trong việc dịch chuyển correctness khỏi ~0.73–0.74, trong khi faithfulness (không tham chiếu gold) chuyển động rõ — kết luận trần-là-thước-đo, dẫn tới toàn bộ mục 5.2–5.4.

### 5.6.3. Bài học ablation

Bài học vận hành được rút thành quy tắc: **không bật cờ tính năng dựa trên kết quả mẫu nhỏ; mọi mức tăng phải được xác nhận ở n lớn (nhiễu ~±0.03) trước khi vào cấu hình mặc định.** Quy tắc này về sau được nâng cấp thành "luật quyết định đăng ký trước" (pre-registered decision rule, mục 5.9): tiêu chí bật cờ được viết ra *trước khi* chạy thí nghiệm, khóa chặt đường lùi "hợp lý hóa hậu nghiệm".

## 5.7. An toàn từ chối trả lời (abstain)

Với chatbot y tế, lỗi nguy hiểm nhất không phải là trả lời thiếu mà là **bịa ra câu trả lời có trích dẫn cho một thứ không tồn tại trong tài liệu**. Nhóm thực nghiệm này đánh giá năng lực từ chối trả lời của hệ thống trước 15 câu hỏi bịa đặt/ngoài corpus (thuốc không tồn tại, bệnh không có trong tài liệu, chủ đề phần mềm). Phép phân lớp `classify_refusal` dựa trên luật (không cần giám khảo LLM), chia phản hồi thành ba lớp: **abstained** (từ chối sạch, có dấu hiệu bất định chuẩn), **hedged_cited** (trả lời nước đôi nhưng vẫn kèm trích dẫn — nguy hiểm vì tạo vẻ có căn cứ), **hallucinated** (trả lời bịa).

### 5.7.1. A/B từ chối trả lời: cổng sàn điểm rerank là cơ chế hiệu quả

| Cấu hình | refusal_rate ↑ | hedged_cited ↓ | hallucination ↓ | Trích dẫn distractor |
|---|---|---|---|---|
| A. rerank hỏng (`llm_chat`, không có điểm) | 0.000 | 0.667 | 0.267 | **8–22 mỗi câu trả lời** |
| B. rerank đã sửa, chỉ abstain bằng prompt (tắt cổng sàn) | 0.267 | 0.000 | 0.667 | 8–15 |
| C. rerank đã sửa + cổng sàn cứng BẬT | **0.400** | 0.000 | 0.533 | **0** |

Ba phát hiện:

1. **Cấu hình A phơi bày một hồi quy nghiêm trọng:** toàn bộ cơ chế an toàn dựa trên sàn điểm (thin-context abstain, cổng sàn, cổng answerability) đọc `rerank_score` từ ngữ cảnh đã đóng gói — nhưng điểm này **chưa từng tồn tại** vì hai lỗi chồng nhau: (i) `maybe_rerank` khóa ứng viên theo `id` trong khi pipeline mang `content_hash` → rerank bị bỏ qua âm thầm ở *mọi* truy vấn (đã sửa: backfill `id=content_hash`, commit `af29043`); (ii) backend `llm_chat` sắp xếp lại ứng viên nhưng không gắn điểm — chỉ `local_cross_encoder` phát ra `rerank_score`. Kết quả là refusal_rate = 0.000: hệ thống trích dẫn **8–22 đoạn distractor** cho thuốc/bệnh bịa đặt. Điểm đáng ghi: các bài test đơn vị đều xanh trong suốt thời gian này (retrieval bị mock) — chỉ đánh giá end-to-end mới phát hiện được.
2. **Abstain bằng prompt là không đủ (cấu hình B):** dù thin-context đã kích hoạt đúng (câu ngoài corpus có max ≈ 0.50 < sàn), mô hình **phớt lờ chỉ dẫn từ chối trong prompt 2/3 số lần** và tự tin trả lời kèm 8–15 trích dẫn distractor (hallucination 0.667). Tri thức tham số (parametric knowledge) của LLM không thể bị chặn bằng lời nhắc.
3. **Cổng sàn cứng (cấu hình C) là cơ chế đúng:** loại bỏ ngữ cảnh dưới sàn *trước* nút trả lời → refusal 0.400, và quan trọng nhất **trích dẫn distractor về 0, hedged_cited về 0** — mô hình không thể trích dẫn thứ nó không nhìn thấy. Đây là thuộc tính phục vụ trực tiếp ưu tiên an toàn y tế: *không bao giờ trích dẫn một nguồn ngụy tạo*.

Phần hallucination tồn dư 0.533 của cấu hình C (mô hình trả lời "React useState", "TCP handshake"… từ tri thức huấn luyện ngay cả khi ngữ cảnh rỗng) được đóng bằng bước tiếp theo: **từ chối tất định khi ngữ cảnh rỗng** (commit `b1ca39e`) — khi cổng sàn làm rỗng ngữ cảnh, hệ thống trả lời từ chối soạn sẵn **mà không gọi LLM trả lời**, triệt tiêu hoàn toàn con đường bịa từ tri thức tham số; câu từ chối mang dấu hiệu bất định chuẩn để được phân lớp là abstain sạch. Cổng answerability dải-xám `[0.60, 0.73)` được đo là không kích hoạt trên failure mode này (câu ngoài corpus nằm ở ~0.50, *dưới* dải) → giữ tắt, tránh phức tạp không có lợi ích đo được.

### 5.7.2. Căn chỉnh sàn: phân bố điểm rerank hai đỉnh

Nền tảng thực nghiệm của giá trị sàn là **phân bố điểm rerank tách đôi rõ rệt**. Đo lần đầu (2026-06-24, cross-encoder `dengcao/bge-reranker-v2-m3`, đầu ra sigmoid):

| Lớp câu hỏi | max rerank_score | Nguồn |
|---|---|---|
| Ngoài corpus (không liên quan) | ≈ 0.500 (0.500–0.502, cực chụm) | đo trên bộ từ chối n=15 |
| Trong corpus (liên quan) | ≈ 0.73 | báo cáo 19/06 |

Cross-encoder "đặt sàn" các đoạn không liên quan tại sigmoid(0) ≈ 0.5 bất kể nội dung — một bất biến hữu ích. Sàn 0.6 khi đó nằm giữa khe 0.50–0.73 và được xác nhận hợp lệ; về sau hạ xuống 0.55 vì phát hiện các chunk tiếng Việt liên-quan-nhưng-diễn-đạt-lại rơi vào vùng ~0.61 sát ngưỡng (mục 5.3.1). Đề xuất sàn theo chuyên khoa (per-specialty floor) bị **từ chối có chủ đích**: không có tín hiệu tách ngưỡng giữa các chuyên khoa, dữ liệu gắn nhãn không đủ — thêm map cấu hình khi chưa có lợi ích đo được là phức tạp hóa vô ích.

### 5.7.3. Tái căn chỉnh sàn trên corpus e5-FT (2026-07-13)

Khi production chuyển sang embedding fine-tune e5 (`agentrag-embed-v1`), phân bố điểm rerank phải được đo lại trên corpus y tế thật (embedding đổi làm đổi *tập ứng viên* đưa vào reranker, dù bản thân cross-encoder không đổi). Phép đo bọc `ContextAssembler.assemble` trên đúng đường chạy thật (đường 13-node graph), dùng câu hỏi y tế thật từ `c2_evalset_n40.jsonl` (trong corpus) và tên thuốc/bệnh bịa đặt (ngoài corpus):

| Tập | min | median | max | n |
|---|---|---|---|---|
| Ngoài corpus (bịa đặt) | 0.5014 | 0.5045 | **0.5176** | 4 |
| Trong corpus (y tế thật) | 0.5268 | 0.7209 | 0.7310 | 12 |

Diễn giải: phân bố **vẫn hai đỉnh sạch**, khớp thời bge-m3 — ngoài corpus phẳng 0.50–0.52, khối trong-corpus 0.66–0.73. **Quyết định: giữ `RETRIEVAL_RELEVANCE_FLOOR=0.55` (không đổi).** 0.55 nằm đúng khe trống: max ngoài-corpus 0.5176 < 0.55 < khối trong-corpus 0.66+ — câu ngoài corpus bị chặn, câu thật đi qua. Đáng chú ý, **0.55 thắng 0.6 trên corpus này**: nếu nâng lên 0.6 sẽ cắt nhầm hai truy vấn trong-corpus hợp lệ có điểm 0.5694 và 0.5956. Một outlier trong-corpus tại 0.5268 ("chép lại câu hỏi thi 8–12") là yêu cầu meta/mỏng, việc bị điểm thấp là hợp lý. Kết luận phụ: embedding FT không dịch chuyển phân bố reranker đủ để phải tái căn chỉnh — reranker là cross-encoder riêng, không đổi.

## 5.8. Fine-tune mô hình embedding

Đây là đòn bẩy cấu trúc cuối cùng chưa kéo: fine-tune mô hình truy hồi theo miền y tế tiếng Việt. Mục tiêu được phát biểu tường minh trước khi chạy: *"chứng minh hoặc khai tử đòn bẩy fine-tune — bằng một con số đáng tin — và promote nếu thắng."*

**Dữ liệu huấn luyện:** `mine_finetune_pairs.py` sinh **5888 bộ ba (triplet)** — câu hỏi tổng hợp (sinh bằng deepseek) + đoạn dương (chunk nguồn) + hard-negative (đào bằng Elasticsearch) — trên corpus y tế thật; chia 90/10 thành 5300 train / 588 test. Huấn luyện trên RTX 5060 Ti 16GB, fp32, batch 8, max-seq 512; mô hình nền `intfloat/multilingual-e5-base`.

**Gate 1 — chất lượng truy hồi trên tập test giữ lại (held-out):** đo bằng recall@k và MRR. *Recall@k* là tỷ lệ truy vấn mà đoạn vàng xuất hiện trong top-k kết quả; *MRR@10* (Mean Reciprocal Rank) là trung bình nghịch đảo thứ hạng của đoạn vàng (đoạn vàng đứng hạng 1 → 1.0, hạng 2 → 0.5…), phản ánh không chỉ "có mặt" mà "đứng cao".

| Metric | e5-base (nền) | FT (`agentrag-embed-v1`) | Delta |
|---|---|---|---|
| recall@5 | 0.726 | 0.976 | **+0.250** |
| recall@10 | 0.789 | 0.993 | **+0.204** |
| mrr@10 | 0.578 | 0.913 | **+0.335** |

**Kết luận Gate 1: PROMOTE = YES** — recall@10 +0.204 vượt xa ngưỡng promote +0.05, mrr@10 +0.335 vượt ngưỡng +0.03. Đòn bẩy fine-tune **hoạt động** ở tầng truy hồi, với caveat trung thực: 588 triplet test đến từ *cùng* bộ sinh câu hỏi tổng hợp với tập train, nên **độ lớn** +0.20–0.33 bị thổi phồng bởi việc mô hình học phong cách câu hỏi synthetic — đây là bằng chứng định hướng mạnh rằng đòn bẩy có thật, không phải ước lượng mức tăng ngoài đời. (Gate 2 — fine-tune reranker — bị vô hiệu về mặt đo lường: công cụ `eval_retrieval.py` chỉ đánh giá được bi-encoder qua `.encode()`, trong khi reranker là cross-encoder, nên cả nền lẫn FT đều cho recall ~1% rác; mô hình reranker FT đã huấn luyện xong và chờ một phép đo đúng — không được đọc là "reranker fail".)

**Gate 2 (C2) — lan truyền end-to-end:** liệu +0.204 recall@10 có chuyển thành correctness? Thay vì thực hiện ngay cuộc di trú tốn kém (đổi chiều embedding 1024→768, dựng TEI, re-index, re-ingest), câu hỏi được trả lời bằng oracle probe trên hệ baseline: n=10 trên corpus thật cho system **0.920**, oracle **0.950**, oracle − system **+0.030**, pearson 0.812. Suy luận: oracle là trần của *mọi* cải tiến truy hồi — hệ thống chỉ "để rơi" tối đa +0.030 cho truy hồi + sinh chưa hoàn hảo, mà +0.030 ở n=10 (nhiễu ±0.05–0.07) là **trong nhiễu**. Kết luận: **mức tăng truy hồi của fine-tune KHÔNG lan truyền thành correctness trên corpus này** — hệ thống đã bão hòa correctness so với trần thước đo lúc đó. (Lưu ý mẫu n=10 này nhỏ; các probe n=40 sau đó ở mục 5.5 cho bức tranh đầy đủ hơn: trên eval set khó hơn có multi-hop, corpus thật hóa ra vẫn còn dư địa +0.088.)

**Phán quyết cuối:** GIỮ embedding FT (truy hồi thật sự tốt hơn nhiều — giá trị cho recall trên corpus khó/lớn hơn, và đã được promote lên production qua TEI), nhưng **không kỳ vọng correctness tăng** từ nó trên corpus hiện tại. Bài học phương pháp mang tên **"đo đúng tầng"**: một cải tiến có thể thắng lớn ở tầng của nó (retrieval: +0.204 recall@10) mà hoàn toàn không hiện hình ở tầng trên (answer correctness) khi tầng trên bị chặn bởi thứ khác (trần thước đo, hoặc đuôi lỗi thuộc lớp khác). Nếu chỉ đo end-to-end, ta kết luận nhầm "fine-tune vô dụng"; nếu chỉ đo tầng truy hồi, ta kết luận nhầm "chất lượng đáp án sẽ tăng +0.20". Cả hai phép đo, cộng với oracle làm trọng tài, mới cho bức tranh đúng. Đây cũng là lần thứ ba liên tiếp một đòn bẩy mạnh (sau kiến trúc truy hồi và mô hình trả lời) không dịch chuyển correctness trên bộ đo cũ — củng cố chuỗi suy luận ở mục 5.2.

## 5.9. Chiến dịch phân loại lỗi (miss-bucketing) — đang triển khai

Kết quả mục 5.5 (dư địa +0.088 tập trung ở ~5/40 câu hỏng) đặt ra câu hỏi kế tiếp: **5 câu đó hỏng vì lý do gì?** Câu trả lời quyết định hướng đầu tư của giai đoạn tiếp theo — xây đồ thị tri thức đa bước (HippoRAG-2), tinh chỉnh cổng an toàn, hay sửa prompt trả lời. Chiến dịch miss-bucketing (nhánh `feat/miss-buckets-crag-flywheel`) được xây dựng để trả lời câu hỏi đó một cách có kỷ luật. Bộ công cụ đã hoàn thành và kiểm thử (suite eval 84/84, sau bổ sung 97/97); các lần chạy sống đang trong kế hoạch thực thi.

**Công cụ mới:**

1. **Ghi nhận từng dòng (per-row capture):** `oracle_probe.py --rows-out x.jsonl` xuất cho *mỗi câu hỏi* toàn bộ chứng cứ: câu trả lời system/oracle, điểm judge1/judge2, các đoạn văn đã đóng gói kèm điểm rerank, trích dẫn inline `[n]`, lớp `classify_refusal`, và các truy vấn công cụ mà agent đã phát. Đây là hạ tầng biến một "câu 0 điểm" từ con số thành hồ sơ khám nghiệm được.
2. **Bộ phân lớp miss ba lớp** (`src/agentrag/eval/miss_buckets.py` + `report_miss_buckets.py`): mỗi câu hỏng (sys < 0.5) được xếp vào một trong ba bucket, mỗi bucket "bật đèn xanh" cho một hướng hành động đã đăng ký trước:

| Bucket | Ý nghĩa | Hành động được bật đèn xanh |
|---|---|---|
| `false_abstention` | từ chối một câu trả lời được | tinh chỉnh sàn/cổng (KHÔNG phải việc đồ thị) |
| `retrieval_miss` | chunk vàng không bao giờ tới được LLM trả lời (Jaccard < 0.35 giữa gold và ngữ cảnh đóng gói) | kế hoạch HippoRAG-2 StructMem/graph (nếu chiếm đa số → viết spec đó) |
| `generation_miss` | gold đã nằm trong ngữ cảnh mà câu trả lời vẫn sai | sửa prompt/mô hình trả lời |

   Các dòng có chênh lệch giám khảo lớn (|sys − judge2| ≥ 0.4) được gắn cờ riêng để soi thủ công. (Jaccard là độ đo trùng lặp giữa hai tập: kích thước phần giao chia phần hợp.)
3. **Bánh đà trích dẫn (citation-reward flywheel, RMM):** `mine_citation_pairs.py` dùng chính trích dẫn inline của LLM trả lời để gắn nhãn pool rerank — đoạn được trích dẫn = mẫu dương, đoạn không được trích dẫn "cứng" nhất = hard negative, chỉ lấy từ các dòng sys ≥ 0.75 (câu trả lời tốt mới đáng tin làm nhãn) — xuất đúng định dạng đầu vào của `finetune_reranker.py`/`finetune_embedding.py`, tích lũy dần qua `--append`. **Không tốn một nhãn tay nào.** Bản song sinh `mine_citation_pairs_prod.py` khai thác cùng tín hiệu từ lưu lượng production thật (các lượt được người dùng 👍, join với `chat_messages.citations`; đánh giá của người dùng thay cho điểm judge) — bánh đà tích lũy từ sử dụng thật, không chỉ từ eval.
4. **Chốt chặn dấu vân tay corpus (corpus fingerprint guard):** triệt tiêu lớp "mìn v3" (mục 5.1.4 — bộ eval sinh trên corpus residue âm thầm chấm sys=0.00 trên corpus thật): `build_prod_evalset.py` đóng dấu mỗi dòng bằng `corpus_fp` (sha1 trên danh sách sắp thứ tự các cặp tiêu-đề-tài-liệu:số-segment); `oracle_probe.py --eval-set` tính lại vân tay của chỉ mục đang chạy và **từ chối chạy** khi lệch (ghi đè được bằng `--allow-corpus-mismatch`; bộ cũ chưa đóng dấu chỉ cảnh báo). 9 test riêng cho thành phần này.

**Luật quyết định đăng ký trước (pre-registered):** hai quyết định của chiến dịch được viết thành tiêu chí *trước khi* chạy, không cho phép hợp lý hóa hậu nghiệm:

- **Cờ `CRAG_ENABLED`** (vòng lặp tự phê bình — critique → truy hồi sửa lỗi → trả lời lại — đã xây từ WS3, mặc định OFF) chỉ được bật khi và chỉ khi: **Δsystem_avg ≥ +0.02** giữa nhánh CRAG-on và CRAG-off trên cùng bộ c2 n=40 **VÀ** bộ từ chối ngoài-corpus cho **0 câu hallucinated** (CRAG coi câu trả lời bất định là thiếu căn cứ và thử lại — rủi ro biến một lần từ chối sạch thành một lần bịa, nên cổng an toàn là điều kiện cứng).
- **Bucket chiếm đa số quyết định hạng mục xây tiếp theo:** `retrieval_miss` → xây HippoRAG-2 (spec DRAFT đã viết, **bị gate**: chỉ xây nếu bucket này thắng); `false_abstention` → tinh chỉnh sàn/cổng; `generation_miss` → sửa prompt trả lời.

**Kế hoạch chạy** (tuần tự, đã soạn thành runbook copy-paste): (1) nhánh baseline CRAG OFF trên `c2_evalset_n40.jsonl` với per-row capture; (2) chạy báo cáo bucket trên các miss — **sản phẩm chính của chiến dịch**; (3) nhánh CRAG ON cùng bộ, cùng judge map (gemini-2.5-pro / deepseek-v4-pro — giữ tính so sánh được với probe 2026-07-13); (4) cổng an toàn từ chối với CRAG ON; (5) gieo bánh đà trích dẫn từ row dump + lượt production được đánh giá; (6) viết tài liệu quyết định `crag_ab_2026-07-14.md` theo template định sẵn. Tại thời điểm viết báo cáo, các bước chạy sống này đang được thực thi; kết quả sẽ quyết định hạng mục kỹ thuật lớn tiếp theo của hệ thống.

## 5.10. Thảo luận

Chuỗi thực nghiệm của chương này, ngoài các con số cụ thể, để lại bốn bài học phương pháp luận có giá trị tổng quát cho việc phát triển hệ thống RAG.

**Bài học 1 — Đo trước, sửa sau; và số mẫu nhỏ luôn "chạy nóng".** Ba lần trong chương này, con số mẫu nhỏ đẹp hơn sự thật: CR+RAPTOR +0.04 precision ở n=10 sụp về +0.014 (nhiễu) kèm faithfulness −0.039 ở n=80; probe v2 cho system 0.950 nhưng số sạch n=50 là 0.888 (10/30 câu bị loại vì lỗi 503 tạo thiên lệch chọn mẫu về phía câu dễ); gap +0.030 ở n=10 với nhiễu ±0.05–0.07 không đủ phân giải. Nếu đội phát triển hành động theo các con số mẫu nhỏ, hệ thống đã gánh một tính năng ingest đắt đỏ làm *giảm* độ bám nguồn, và báo cáo một con số correctness thổi phồng 0.062. Quy tắc rút ra: mức tăng nào chưa xác nhận ở cỡ mẫu có dải nhiễu ≤ ±0.03 thì chưa tồn tại.

**Bài học 2 — Trước khi hỏi "hệ thống được mấy điểm", phải hỏi "thước đo có đo được không".** Toàn bộ mạch chính của chương xoay quanh việc phân xử "trần 0.74": hai đòn bẩy mạnh không dịch chuyển metric trong khi metric không-tham-chiếu vẫn chuyển động → nghi ngờ thước đo; xây thước đo mới (ensemble nugget+rubric) và **kiểm chứng chính nó** bằng oracle (trần 0.976, không bị nghẹt) và pearson liên-giám-khảo (0.965); rồi dùng hiệu oracle − system làm la bàn phân bổ nỗ lực: corpus residue +0.046 → metric-bound, dừng tinh chỉnh; corpus y tế thật +0.088 → system-bound, còn ~0.09 dư địa thật đáng đầu tư, và đã khoanh được vào ~5/40 câu. Không có oracle probe, hai tình huống này không thể phân biệt — và mọi quyết sách tối ưu đều là đoán mò. Đồng thời, chính oracle cũng có nhiễu (dao động 0.847–0.926, ±0.04 trên cùng bộ câu hỏi), nên hiệu số giữa các lần chạy khác đợt không được đem so trực tiếp — các phép so sánh phải cùng giám khảo, cùng eval set, cùng đợt.

**Bài học 3 — Giám khảo LLM phải bị nghi ngờ có hệ thống, và sự nghi ngờ đó đo được.** Chương này xử lý hai tầng rủi ro giám khảo: *nhiễu* (đo bằng pearson giữa hai giám khảo: 0.965/0.962 với cặp gemini, 0.730 với cặp cùng-họ deepseek — con số sau được dán nhãn rõ là "trường hợp lạc quan") và *thiên lệch tự ưu ái* (giám khảo cùng nhà cung cấp với mô hình trả lời). Cả hai chỉ được đóng lại bằng một phép đo trả tiền thật: giám khảo gemini-2.5-pro độc lập đồng thuận với deepseek-v4-pro ở pearson 0.921, và con số hệ thống gần như không đổi khi đổi giám khảo (0.759 vs 0.764) — từ đó lịch sử số liệu mới được "giải oan" và trích dẫn được. Điểm đáng nhấn mạnh về văn hóa báo cáo: các caveat (corpus residue chứ chưa phải y tế, gold synthetic chưa hiệu chuẩn người, độ lớn recall bị thổi phồng bởi synthetic test) đều được ghi thẳng vào báo cáo gốc tại thời điểm công bố nội bộ, thay vì để người đọc sau tự phát hiện.

**Bài học 4 — Luật quyết định đăng ký trước là hàng rào chống tự lừa dối.** Từ bài học ablation, quy trình được nâng cấp: tiêu chí bật `CRAG_ENABLED` (Δ ≥ +0.02 VÀ 0 hallucinated trên bộ ngoài-corpus) và bảng ánh xạ bucket-đa-số → hạng-mục-xây-tiếp được viết ra *trước khi* có số liệu. Cách làm này — vay từ chuẩn mực đăng ký trước (pre-registration) của khoa học thực nghiệm — loại bỏ khả năng "nhìn số rồi kể chuyện": nếu CRAG tăng +0.015, nó sẽ ở OFF, bất kể +0.015 "trông có vẻ hứa hẹn" đến đâu; nếu bucket đa số là `false_abstention`, kế hoạch đồ thị HippoRAG-2 sẽ không được xây, dù nó hấp dẫn về mặt kỹ thuật đến đâu. Song hành là các hàng rào kỹ thuật chống lỗi quy trình lặng lẽ: chốt vân tay corpus (chặn eval set lệch snapshot), preflight benchmark (chặn stack nửa vời), và quy tắc "chỉ backend `local_cross_encoder` mới có `rerank_score`" được cài thành guard khởi động — vì bài học đắt nhất của cả chương là hồi quy an toàn ở mục 5.7.1: **hệ thống từng bịa trích dẫn cho thuốc không tồn tại trong khi mọi test đơn vị đều xanh.** An toàn của một hệ thống RAG y tế không nằm ở prompt, mà nằm ở các cổng cơ học được căn chỉnh bằng phân bố đo thật (0.50–0.52 vs 0.66–0.73, sàn 0.55) và được tái kiểm mỗi khi một thành phần thượng nguồn thay đổi.

Tổng kết trạng thái hệ thống tại thời điểm kết thúc chương: correctness ensemble **0.759** trên corpus y tế thật với giám khảo độc lập (pearson chéo nhà cung cấp 0.921), oracle 0.847, dư địa cải thiện +0.088 tập trung ở ~5/40 câu; an toàn từ chối ngoài-corpus được bảo toàn qua hai lần căn chỉnh sàn (0.55) và cơ chế từ chối tất định; các cờ kiến trúc đắt đỏ được giữ ở trạng thái do số liệu quyết định thay vì do kỳ vọng; và một chiến dịch phân loại lỗi có luật quyết định đăng ký trước đang chạy để định hướng chặng phát triển kế tiếp.
