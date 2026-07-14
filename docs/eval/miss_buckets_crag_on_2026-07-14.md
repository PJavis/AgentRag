# Miss buckets — c2_evalset_n40-crag-on

- rows scored: 40
- misses (system_mean < 0.5): 8
- **false_abstention**: 1
- **generation_miss**: 2
- **retrieval_miss**: 5
- judge-disagreement rows (|sys−judge2| ≥ 0.4): prod_corpus-23

## Miss detail

### prod_corpus-0 — `generation_miss`

- Q: Tài liệu học tập của môn học này là gì?
- sys=0.0 oracle=1.0 judge2=0.0 refusal=hallucinated
- gold_overlap=1.00 max_rerank=0.5694117368362787 cited=[1]
- tool_queries: ['Tài liệu học tập của môn học này là gì?']
- answer: Tài liệu học tập của môn học này bao gồm:
- **Dược lý học**, NXB Giáo dục [1]
- **Dược lý học lâm sàng**, NXB Y Học [1]
- Tài liệu tham khảo: **Dược thư Quốc gia VN 2002** [1]

### prod_corpus-5 — `retrieval_miss`

- Q: Nội dung các câu hỏi từ câu 8 đến câu 12 trong đề thi là gì?
- sys=0.0 oracle=0.5909090909090909 judge2=0.0 refusal=hallucinated
- gold_overlap=0.22 max_rerank=0.5267669974563589 cited=[1, 3, 4]
- tool_queries: ['Nội dung các câu hỏi từ câu 8 đến câu 12 trong đề thi là gì?']
- answer: ### Nội dung các câu hỏi từ câu 8 đến câu 12 trong các tài liệu đề thi

#### Từ tài liệu "5ff1e288-7474-4a4c-9a95-899f78adfd0c" (trang 2-3) và "18056887-feaa-46c7-8b32-f67212b33d9e" (trang 6)

- **Câu 8:** Chất nào sau đây **không** có vai trò kích thích **Hủy cốt bào** chết theo chương trình? Lựa c

### prod_corpus-16 — `false_abstention`

- Q: Nội dung cụ thể của câu hỏi số 18 đến 21 trong đề thi giải phẫu là gì?
- sys=0.0 oracle=0.85 judge2=0.0 refusal=abstained
- gold_overlap=0.25 max_rerank=0.5216230635339086 cited=[1]
- tool_queries: ['Nội dung cụ thể của câu hỏi số 18 đến 21 trong đề thi giải phẫu là gì?', 'Nội dung cụ thể của câu hỏi số 18 đến 21 trong đề thi giải phẫu là gì? (bối cảnh tổng quát, định nghĩa, nguyên nhân)']
- answer: Dựa trên nội dung từ tài liệu có tiêu đề "1617bcff-9bf1-41de-aa8d-9b5fcfc5f78e", câu hỏi số 18 đến 20 trong đề thi giải phẫu (cụ thể là giải phẫu bệnh) như sau:

### Câu 18
Bệnh nhân nam 58 tuổi đến khám vì đau khớp, giải phẫu bệnh cho hình ảnh trên. Đây là tổn thương khớp viêm giai đoạn nào?
- **A.

### prod_corpus-14 — `generation_miss`

- Q: Mô học xương đùi của bệnh nhân này cho thấy gì?
- sys=0.0 oracle=0.725 judge2=0.0 refusal=hallucinated
- gold_overlap=1.00 max_rerank=0.709904760798842 cited=[1]
- tool_queries: ['Mô học xương đùi của bệnh nhân này cho thấy gì?']
- answer: Mô học xương đùi của bệnh nhân (trẻ 6 tuổi, tử thi) cho thấy: **không gian tủy bị tiêu biến**; **trabeculae xương vô tổ chức** với sự giữ lại các **bọt biển sơ cấp** và các khoang tủy bị hủy hoại bởi các **lớp đệm thứ cấp**; **tế bào tủy xương tạo máu thưa thớt** [1].

### prod_corpus-21 — `retrieval_miss`

- Q: Những dấu hiệu lâm sàng và cận lâm sàng trong cai nghiện ma túy là gì?
- sys=0.3666666666666667 oracle=0.9791666666666667 judge2=0.2681818181818182 refusal=hallucinated
- gold_overlap=0.24 max_rerank=0.6817917471980462 cited=[1, 2, 3, 4, 5]
- tool_queries: ['Những dấu hiệu lâm sàng và cận lâm sàng trong cai nghiện ma túy là gì?']
- answer: ## Dấu hiệu lâm sàng và cận lâm sàng trong cai nghiện ma túy

### Dấu hiệu lâm sàng
- **Chẩn đoán nghiện chất ma túy** (theo ICD-10): cần có ≥3 biểu hiện sau trong ít nhất 1 tháng (hoặc lặp lại trong 12 tháng) [1][5]:
  - Thèm muốn mạnh mẽ hoặc cảm thấy buộc phải sử dụng chất.
  - Khó khăn trong kiể

### prod_corpus-26 — `retrieval_miss`

- Q: Các bước thực hiện sau khi rút ống thông tiểu cho người bệnh là gì?
- sys=0.39230769230769236 oracle=0.8115384615384615 judge2=0.34705882352941175 refusal=hallucinated
- gold_overlap=0.27 max_rerank=0.722611852837524 cited=[]
- tool_queries: ['Các bước thực hiện sau khi rút ống thông tiểu cho người bệnh là gì?']
- answer: ### Các bước sau khi rút ống thông tiểu
- **Gập hoặc kẹp ống rút** ra bỏ vào khay hạt đậu hoặc túi nilon.
- **Thấm khô lỗ tiểu** bằng gạc củ ấu, bỏ dụng cụ, nilon, **mặc quần**, bỏ ga đắp, giúp người bệnh về tư thế thoải mái, **dặn dò** người bệnh.
- **Thu dọn dụng cụ, rửa tay.** Ghi hồ sơ: ghi họ t

### prod_corpus-39 — `retrieval_miss`

- Q: Đoạn văn này chứa những câu hỏi gì?
- sys=0.0 oracle=0.5 judge2=0.0 refusal=hallucinated
- gold_overlap=0.11 max_rerank=0.526507262565869 cited=[]
- tool_queries: ['Đoạn văn này chứa những câu hỏi gì?', 'Đoạn văn này chứa những câu hỏi gì? (bối cảnh tổng quát, định nghĩa, nguyên nhân)']
- answer: Đoạn văn này chứa các câu hỏi trắc nghiệm sau:

**Từ tài liệu về da liễu (source 1):**
- Các câu hỏi từ Câu 48 đến Câu 51 (có thể kèm một câu hỏi trước đó với các lựa chọn A-D về sự giãn/co mao mạch).

**Từ tài liệu về giải phẫu chi trên (source 3):**
- Câu 1: Có bao nhiêu cơ nội tại ở vai và nách?


### prod_corpus-38 — `retrieval_miss`

- Q: Những câu hỏi nào được đưa ra trong đoạn văn?
- sys=0.0 oracle=1.0 judge2=0.0 refusal=hallucinated
- gold_overlap=0.11 max_rerank=0.5276804678588385 cited=[]
- tool_queries: ['Những câu hỏi nào được đưa ra trong đoạn văn?']
- answer: ### Từ tài liệu "9d0b8134-e52a-427e-aa3e-8ea16914518e" (nguồn 1):
- **Câu 1**: Đám rối thần kinh cánh tay không được tạo nên từ dây thần kinh sống nào sau đây?
- **Câu 2**: Thân trước của đám rối cánh tay được tạo thành từ?
- **Câu 3**: Bó mạch thần kinh nào ở đám rối cánh tay nằm sau động mạch nách

