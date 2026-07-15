# Miss buckets — c2_clean_v2-crag-off

- rows scored: 42
- misses (system_mean < 0.5): 7
- **generation_miss**: 3
- **retrieval_miss**: 4
- judge-disagreement rows (|sys−judge2| ≥ 0.4): prod_corpus_multihop-5, prod_corpus_multihop-1

## Miss detail

### prod_corpus-7 — `retrieval_miss`

- Q: Các bước đặt ống thông tiểu?
- sys=0.3875 oracle=1.0 judge2=0.48636363636363633 refusal=hallucinated
- gold_overlap=0.24 max_rerank=0.7263281467319471 cited=[1, 2]
- tool_queries: ['Các bước đặt ống thông tiểu?']
- answer: ### Chuẩn bị
- **Người bệnh**: Chào hỏi, giải thích, nhận định tình trạng (dị ứng, dấu hiệu sinh tồn, bàng quang, lỗ tiểu) [1].
- **Điều dưỡng**: Sát khuẩn tay, mở khay dụng cụ vô khuẩn [2].
- **Dụng cụ**: Chuẩn bị đầy đủ [2].

### Tiến hành
1. Bộc lộ bộ phận sinh dục, điều dưỡng sát khuẩn, đi găng 

### prod_corpus-12 — `retrieval_miss`

- Q: Triệu chứng cai nghiện thuốc phiện là gì?
- sys=0.0 oracle=1.0 judge2=0.0 refusal=hallucinated
- gold_overlap=0.10 max_rerank=0.7149120076798713 cited=[]
- tool_queries: ['Triệu chứng cai nghiện thuốc phiện là gì?', 'triệu chứng cai nghiện thuốc phiện là gì? ICD-10 code definition']
- answer: Theo tài liệu tham khảo (nguồn 1 và 3), hội chứng cai chất dạng thuốc phiện theo ICD-10 cần có **ít nhất ba** trong số các dấu hiệu. Tuy nhiên, danh sách cụ thể các dấu hiệu này không được cung cấp trong đoạn trích.

### prod_corpus-15 — `retrieval_miss`

- Q: Những yếu tố nào cần được khai thác trong tiền sử bệnh?
- sys=0.07142857142857142 oracle=1.0 judge2=0.07142857142857142 refusal=hallucinated
- gold_overlap=0.09 max_rerank=0.7205260663967841 cited=[1, 2, 4]
- tool_queries: ['Những yếu tố nào cần được khai thác trong tiền sử bệnh?']
- answer: Khi khai thác tiền sử bệnh, cần tập trung vào các yếu tố sau:

### Đối với hiếm muộn nam
- **Thời gian hiếm muộn**, số con hiện tại hoặc tiền sử thai sản của vợ, tuổi và tình trạng sức khỏe sinh sản của người vợ [1].
- Các bệnh hoặc can thiệp ảnh hưởng đến tinh hoàn: **tinh hoàn không xuống bìu**, *

### prod_corpus-25 — `generation_miss`

- Q: Một phụ nữ 22 tuổi bị thương chân và yếu cẳng chân trái sau tai nạn xe máy, sinh thiết cơ dạ dày sau 4 tháng, đặc điểm tốt nhất của bệnh lý này là gì?
- sys=0.2 oracle=0.5 judge2=0.0 refusal=hallucinated
- gold_overlap=1.00 max_rerank=0.7309962993892906 cited=[]
- tool_queries: ['Một phụ nữ 22 tuổi bị thương chân và yếu cẳng chân trái sau tai nạn xe máy, sinh thiết cơ dạ dày sau 4 tháng, đặc điểm tốt nhất của bệnh lý này là gì?', 'Một phụ nữ 22 tuổi bị thương chân và yếu cẳng chân tré sau tai nạn xe máy, sinh thiết của dưới da tị trong hình ảnh']
- answer: Denervation (mất phân bố thần kinh)

### prod_corpus-23 — `retrieval_miss`

- Q: Các nguyên nhân thường gặp gây tiểu không tự chủ là gì?
- sys=0.0 oracle=1.0 judge2=0.0 refusal=hallucinated
- gold_overlap=0.13 max_rerank=0.6102883503777808 cited=[1, 2]
- tool_queries: ['Các nguyên nhân thường gặp gây tiểu không tự chủ là gì?', 'nguyên nhân thường gây tiểu không tự chủ là gì']
- answer: Các nguyên nhân thường gặp gây tiểu không tự chủ (đái dầm) theo tài liệu [1] bao gồm:

- **Rối loạn phát triển thần kinh cơ và nhận thức**: Ảnh hưởng đến khả năng kiểm soát bàng quang.
- **Yếu tố cảm xúc xã hội**: Các vấn đề tâm lý xã hội có thể làm chậm hoàn thiện chức năng bài tiết.
- **Giáo dục c

### prod_corpus_multihop-2 — `generation_miss`

- Q: Loại tinh thể nào trong dịch khớp có liên quan đến các hạt dưới da được gọi là 'Hạt tophi'?
- sys=0.0 oracle=1.0 judge2=0.0 refusal=hallucinated
- gold_overlap=1.00 max_rerank=0.5840815531385711 cited=[]
- tool_queries: ["Loại tinh thể nào trong dịch khớp có liên quan đến các hạt dưới da được gọi là 'Hạt tophi'?", "Loại tinh thể nào trong dịch khớp cð liên quan đến các hạt dưới da được gọi là 'Hạt tophi'?"]
- answer: Thông tin trong ngữ cảnh hiện tại không đủ để trả lời câu hỏi về loại tinh thể trong dịch khớp liên quan đến hạt tophi.

### prod_corpus_multihop-6 — `generation_miss`

- Q: Các thụ thể nào trên bề mặt tế bào hủy xương đóng vai trò trong cơ chế cận tiết giữa hủy cốt bào và tạo cốt bào trong quá trình tái tạo xương?
- sys=0.0 oracle=1.0 judge2=0.0 refusal=hallucinated
- gold_overlap=1.00 max_rerank=0.7260329152105172 cited=[1, 3, 10]
- tool_queries: ['Các yếu tố cận tiết chính tham gia giao tiếp giữa hủy cốt bào và tạo cốt bào trong tái tạo xương là gì?', 'Thụ thể bề mặt trên hủy cốt bào có vai trò nhận tín hiệu từ tạo cốt bào qua cơ chế cận tiết.', 'Các thụ thể nào trên bề mặt tế bào hủy xương đóng vai trò trong cơ chế cận tiết giữa hủy cốt bào và tạo cốt bào trong quá trình tái tạo xương?']
- answer: Các thụ thể trên bề mặt tế bào hủy xương đóng vai trò trong cơ chế cận tiết giữa hủy cốt bào và tạo cốt bào bao gồm **receptor vitronectin** (integrin **αvβ3**) và **receptor calcitonin** [3]. Các thụ thể này có mặt trên tế bào hủy xương đa nhân, giúp bám dính vào chất nền xương và điều hòa hoạt độn

