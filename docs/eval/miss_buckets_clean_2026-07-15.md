# Miss buckets — c2_clean-crag-off

- rows scored: 40
- misses (system_mean < 0.5): 8
- **false_abstention**: 2
- **generation_miss**: 3
- **retrieval_miss**: 3
- judge-disagreement rows (|sys−judge2| ≥ 0.4): prod_corpus_multihop-5, prod_corpus_multihop-1

## Miss detail

### prod_corpus-7 — `retrieval_miss`

- Q: Các bước đặt ống thông tiểu?
- sys=0.3875 oracle=1.0 judge2=0.2 refusal=hallucinated
- gold_overlap=0.24 max_rerank=0.7263281467319471 cited=[1, 2]
- tool_queries: ['Các bước đặt ống thông tiểu?']
- answer: ### Chuẩn bị
- Chào hỏi, xác định người bệnh (tên, tuổi, mã định danh) [1].
- Giới thiệu bản thân, thông báo và giải thích kỹ thuật, động viên người bệnh hợp tác [1].
- Nhận định tình trạng: tiền sử dị ứng, dấu hiệu sinh tồn, tình trạng bàng quang, lỗ tiểu, thời gian đi tiểu lần cuối [1].
- Chuẩn bị

### prod_corpus-12 — `false_abstention`

- Q: Triệu chứng cai nghiện thuốc phiện là gì?
- sys=0.0 oracle=1.0 judge2=0.0 refusal=hedged_cited
- gold_overlap=0.10 max_rerank=0.7149120076798713 cited=[]
- tool_queries: ['Triệu chứng cai nghiện thuốc phiện là gì?', 'Triệu chứng cai nghiện thuốc phiện là gì? ICD-10 code']
- answer: Theo ICD-10, hội chứng cai chất dạng thuốc phiện gồm ít nhất ba trong số các dấu hiệu nhất định, nhưng tài liệu cung cấp không liệt kê cụ thể các dấu hiệu đó. Vì vậy, không thể trả lời đầy đủ dựa trên ngữ cảnh hiện có.

### prod_corpus-15 — `retrieval_miss`

- Q: Những yếu tố nào cần được khai thác trong tiền sử bệnh?
- sys=0.08333333333333333 oracle=1.0 judge2=0.08333333333333333 refusal=hallucinated
- gold_overlap=0.09 max_rerank=0.7205260663967841 cited=[]
- tool_queries: ['Những yếu tố nào cần được khai thác trong tiền sử bệnh?']
- answer: Dựa trên các tài liệu tham khảo, các yếu tố cần khai thác trong tiền sử bệnh được đề cập trong các bối cảnh khác nhau:

### Đối với hiếm muộn nam (nguồn 1):
- **Thời gian hiếm muộn**
- **Số con hiện tại** hoặc tiền sử thai sản của vợ
- **Tuổi và tình trạng sức khỏe sinh sản của người vợ**
- Các bệnh

### prod_corpus-25 — `generation_miss`

- Q: Một phụ nữ 22 tuổi bị thương chân và yếu cẳng chân trái sau tai nạn xe máy, sinh thiết cơ dạ dày sau 4 tháng, đặc điểm tốt nhất của bệnh lý này là gì?
- sys=0.0 oracle=1.0 judge2=0.0 refusal=hallucinated
- gold_overlap=1.00 max_rerank=0.7309962993892906 cited=[1]
- tool_queries: ['Chấn thương chân sau tai nạn xe máy gây yếu cẳng chân, tổn thương dây thần kinh nào?', 'Sinh thiết cơ dạ dày sau chấn thương thần kinh có đặc điểm mô bệnh học gì?', 'Một phụ nữ 22 tuổi bị thương chân và yếu cẳng chân trái sau tai nạn xe máy, sinh thiết cơ dạ dày sau 4 tháng, đặc điểm tốt nhất của bệnh lý này là gì?', 'Sinh thiết cơ dạ dày sau chấn thương thần kinh cô đặc điểm mô bệnh học gì?']
- answer: Đặc điểm tốt nhất của bệnh lý này là **Denervation** (mất phân bố thần kinh) [1].

### prod_corpus-23 — `retrieval_miss`

- Q: Các nguyên nhân thường gặp gây tiểu không tự chủ là gì?
- sys=0.08333333333333333 oracle=1.0 judge2=0.0 refusal=hallucinated
- gold_overlap=0.13 max_rerank=0.6102883503777808 cited=[1, 2, 3, 4]
- tool_queries: ['Các nguyên nhân thường gặp gây tiểu không tự chủ là gì?', 'Các nguyân nhâm thường gâp gây tiểu không tự chủ là gì?']
- answer: Các nguyên nhân thường gặp gây tiểu không tự chủ bao gồm nhiều yếu tố khác nhau, tùy thuộc vào lứa tuổi và bối cảnh lâm sàng. Dưới đây là các nhóm nguyên nhân chính được đề cập trong tài liệu:

### Nguyên nhân sinh lý và phát triển
- **Rối loạn kiểm soát bàng quang**: Khả năng kiểm soát bàng quang b

### prod_corpus-33 — `generation_miss`

- Q: Các câu hỏi về đám rối thần kinh cánh tay là gì?
- sys=0.0625 oracle=0.6666666666666666 judge2=0.0 refusal=hallucinated
- gold_overlap=1.00 max_rerank=0.7293105065663977 cited=[1, 2, 3, 5, 6]
- tool_queries: ['Các câu hỏi về đám rối thần kinh cánh tay là gì?']
- answer: Các câu hỏi về đám rối thần kinh cánh tay trong ngữ cảnh chủ yếu xoay quanh các chủ đề sau:

### Thành phần và cấu trúc
- **Nguồn gốc từ các dây thần kinh sống**: Đám rối cánh tay được tạo nên từ nhánh trước của các dây thần kinh sống **C5 đến T1**; các câu hỏi thường yêu cầu xác định dây thần kinh 

### prod_corpus-39 — `generation_miss`

- Q: Whose screen is being viewed?
- sys=0.45 oracle=0.6 judge2=0.25 refusal=hallucinated
- gold_overlap=1.00 max_rerank=0.6571548611627616 cited=[1, 2]
- tool_queries: ['Whose screen is being viewed?']
- answer: Based on the provided context, two different screens are being viewed:

- **Phạm Đình Hòa**'s screen (source 1) [1]
- **Trần Thanh Tùng**'s screen (source 2) [2]

The context shows multiple viewer windows, each indicating the name of the person whose screen is being shared.

### prod_corpus_multihop-2 — `false_abstention`

- Q: Loại tinh thể nào trong dịch khớp có liên quan đến các hạt dưới da được gọi là 'Hạt tophi'?
- sys=0.0 oracle=1.0 judge2=0.0 refusal=hedged_cited
- gold_overlap=1.00 max_rerank=0.5840815531385711 cited=[]
- tool_queries: ["Loại tinh thể nào trong dịch khớp có liên quan đến các hạt dưới da được gọi là 'Hạt tophi'?", "Loại tinh thể nào trong dịch khớp cð liên quan đến các hạt dưới da được gọi là 'Hạt tophi'?"]
- answer: Ngữ cảnh được cung cấp không đề cập đến loại tinh thể trong dịch khớp liên quan đến hạt tophi.

