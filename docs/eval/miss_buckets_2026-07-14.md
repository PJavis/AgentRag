# Miss buckets — c2_evalset_n40-crag-off

- rows scored: 40
- misses (system_mean < 0.5): 9
- **false_abstention**: 1
- **generation_miss**: 2
- **retrieval_miss**: 6
- judge-disagreement rows (|sys−judge2| ≥ 0.4): prod_corpus-23, prod_corpus-30

## Miss detail

### prod_corpus-0 — `generation_miss`

- Q: Tài liệu học tập của môn học này là gì?
- sys=0.0 oracle=1.0 judge2=0.0 refusal=hallucinated
- gold_overlap=1.00 max_rerank=0.5694117368362787 cited=[1]
- tool_queries: ['Tài liệu học tập của môn học này là gì?']
- answer: Tài liệu học tập của môn học về **Thuốc hạ sốt, giảm đau, chống viêm không steroid** (do TS. Trần Thanh Tùng giảng dạy) bao gồm:
- **Dược lý học**, NXB Giáo dục
- **Dược lý học lâm sàng**, NXB Y Học
- Tài liệu tham khảo: **Dược thư Quốc gia VN 2002** [1]

### prod_corpus-5 — `retrieval_miss`

- Q: Nội dung các câu hỏi từ câu 8 đến câu 12 trong đề thi là gì?
- sys=0.0 oracle=1.0 judge2=0.0 refusal=hallucinated
- gold_overlap=0.22 max_rerank=0.5401416582392619 cited=[1, 2, 3, 5, 6]
- tool_queries: ['Nội dung các câu hỏi từ câu 8 đến câu 12 trong đề thi là gì?', 'Câu 12 trong đề thi là gì?']
- answer: ## Nội dung các câu hỏi từ câu 8 đến câu 12 trong các đề thi

### Document: 5ff1e288-7474-4a4c-9a95-899f78adfd0c
- **Câu 8**: Chất nào sau đây không có vai trò kích thích Hủy cốt bào chết theo chương trình?
  - A. Estrogen
  - B. TGF-β
  - C. IFN-β
  - D. Biphosphonate [5]
- **Câu 9**: Có bao nhiêu 

### prod_corpus-16 — `retrieval_miss`

- Q: Nội dung cụ thể của câu hỏi số 18 đến 21 trong đề thi giải phẫu là gì?
- sys=0.0 oracle=0.4653846153846154 judge2=0.0 refusal=hallucinated
- gold_overlap=0.25 max_rerank=0.5216230635339086 cited=[]
- tool_queries: ['Nội dung cụ thể của câu hỏi số 18 đến 21 trong đề thi giải phẫu là gì?']
- answer: Dưới đây là nội dung cụ thể của các câu hỏi số 18 đến 20 trong đề thi giải phẫu bệnh. Câu hỏi số 21 không xuất hiện trong ngữ cảnh được cung cấp.

### Câu 18 (Nguồn 1)
**Bệnh nhân nam 58 tuổi đến khám vì đau khớp, giải phẫu bệnh cho hình ảnh trên:** Đây là tổn thương khớp viêm giai đoạn nào?
- A. **

### prod_corpus-14 — `generation_miss`

- Q: Mô học xương đùi của bệnh nhân này cho thấy gì?
- sys=0.0 oracle=0.725 judge2=0.0 refusal=hallucinated
- gold_overlap=1.00 max_rerank=0.709904760798842 cited=[1]
- tool_queries: ['Mô học xương đùi của bệnh nhân này cho thấy gì?']
- answer: Mô học xương đùi của bệnh nhân (trẻ 6 tuổi, não úng thủy nhẹ, nhiễm trùng mạn tính, thiếu máu mạn tính khó chữa) cho thấy:
- **Sự vô tổ chức** của **trabeculae xương**, với việc giữ lại các **bọt biển sơ cấp**.
- Các **khoang tủy** bị hủy hoại bởi các **lớp đệm thứ cấp**.
- **Tế bào tủy xương tạo má

### prod_corpus-21 — `retrieval_miss`

- Q: Những dấu hiệu lâm sàng và cận lâm sàng trong cai nghiện ma túy là gì?
- sys=0.3944444444444445 oracle=1.0 judge2=0.3956521739130435 refusal=hallucinated
- gold_overlap=0.24 max_rerank=0.6817917471980462 cited=[1, 3, 4, 5, 6]
- tool_queries: ['Những dấu hiệu lâm sàng và cận lâm sàng trong cai nghiện ma túy là gì?', 'Những dấu hiệu lâm sàng vâ cận lâm sàng trong cai nghiện ma túy là gì?']
- answer: ## Dấu hiệu lâm sàng trong cai nghiện ma túy

Dựa theo ICD-10, chẩn đoán nghiện các chất ma túy khi có từ **3 biểu hiện** sau xảy ra cùng nhau trong ít nhất 1 tháng (hoặc lặp lại trong 12 tháng) [1][6]:

- Thèm muốn mạnh mẽ hoặc cảm thấy buộc phải sử dụng.
- Khó khăn trong việc kiểm soát thời gian b

### prod_corpus-27 — `false_abstention`

- Q: Quản lý và tiên lượng của rối loạn tâm thần trong bệnh gan là gì?
- sys=0.0 oracle=1.0 judge2=0.0 refusal=hedged_cited
- gold_overlap=1.00 max_rerank=0.7221216238102397 cited=[]
- tool_queries: ['Quản lý và tiên lượng của rối loạn tâm thần trong bệnh gan là gì?']
- answer: Tài liệu hiện có không đề cập cụ thể đến quản lý và tiên lượng của rối loạn tâm thần trong bệnh gan. Các thông tin trong ngữ cảnh chủ yếu nói về rối loạn tâm thần liên quan đến sử dụng chất gây nghiện (cocaine, amphetamin, cần sa) và rối loạn tâm thần thực tổn do nguyên nhân khác. Không có dữ liệu t

### prod_corpus-26 — `retrieval_miss`

- Q: Các bước thực hiện sau khi rút ống thông tiểu cho người bệnh là gì?
- sys=0.4307692307692308 oracle=1.0 judge2=0.34705882352941175 refusal=hallucinated
- gold_overlap=0.27 max_rerank=0.722611852837524 cited=[1]
- tool_queries: ['Các bước thực hiện sau khi rút ống thông tiểu cho người bệnh là gì?', 'Các bước thực hiện sau khi rút ống thông tiểu cho người bệnh là gì?']
- answer: Các bước thực hiện sau khi rút ống thông tiểu (lấy nước tiểu làm xét nghiệm) bao gồm:

1. **Gập hoặc kẹp ống rút ra** bỏ vào khay hạt đậu hoặc túi nilon [1].
2. **Thấm khô lỗ tiểu** bằng gạc củ ấu [1].
3. **Bỏ dụng cụ, nilon**, mặc quần cho người bệnh, bỏ ga đắp [1].
4. **Giúp người bệnh về tư thế t

### prod_corpus-39 — `retrieval_miss`

- Q: Đoạn văn này chứa những câu hỏi gì?
- sys=0.0 oracle=0.5 judge2=0.0 refusal=hallucinated
- gold_overlap=0.11 max_rerank=0.526507262565869 cited=[]
- tool_queries: ['Đoạn văn này chứa những câu hỏi gì?']
- answer: Context chứa các câu hỏi trắc nghiệm từ ba đoạn văn khác nhau:

### Đoạn văn 1 (Da liễu - Nhi khoa)
- **Câu 48:** Về mùa lạnh trẻ nhỏ dễ bị cứng bì hoặc phù cứng bì là do?
- **Câu 49:** Trẻ dễ bị nhiễm nóng và nhiễm lạnh những tháng đầu đời do?
- **Câu 50:** Yếu tố gây lão hóa ngoại sinh quan trọng 

### prod_corpus-38 — `retrieval_miss`

- Q: Những câu hỏi nào được đưa ra trong đoạn văn?
- sys=0.0 oracle=1.0 judge2=0.0 refusal=hallucinated
- gold_overlap=0.11 max_rerank=0.5276804678588385 cited=[1, 2, 3]
- tool_queries: ['Những câu hỏi nào được đưa ra trong đoạn văn?']
- answer: Các câu hỏi được đưa ra trong đoạn văn bao gồm:

### Từ nguồn 1 (đám rối thần kinh cánh tay) [1]
- **Câu 1:** Đám rối thần kinh cánh tay không được tạo nên từ dây thần kinh sống nào sau đây?
- **Câu 2:** Thân trước của đám rối cánh tay được tạo thành từ?
- **Câu 3:** Bó mạch thần kinh nào ở đám rối 

