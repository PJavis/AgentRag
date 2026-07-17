# Miss buckets — c2_clean_v2-fix

- rows scored: 42
- misses (system_mean < 0.5): 3
- **generation_miss**: 1
- **retrieval_miss**: 2
- judge-disagreement rows (|sys−judge2| ≥ 0.4): prod_corpus-0, prod_corpus-35, prod_corpus_multihop-5

## Miss detail

### prod_corpus-15 — `retrieval_miss`

- Q: Những yếu tố nào cần được khai thác trong tiền sử bệnh?
- sys=0.08333333333333333 oracle=1.0 judge2=0.07142857142857142 refusal=hallucinated
- gold_overlap=0.09 max_rerank=0.7205260663967841 cited=[1, 2, 5, 6, 7, 8, 11, 12, 13]
- tool_queries: ['Những yếu tố nào cần được khai thác trong tiền sử bệnh?']
- answer: Việc khai thác tiền sử bệnh cần tập trung vào nhiều yếu tố khác nhau tùy theo bệnh lý nghi ngờ. Dưới đây là các yếu tố chính dựa trên các ngữ cảnh được cung cấp:

### Hiếm muộn nam [1][6]
- **Thời gian hiếm muộn**, số con hiện tại và tiền sử thai sản của vợ.
- **Tuổi và tình trạng sức khỏe sinh sản*

### prod_corpus-23 — `retrieval_miss`

- Q: Các nguyên nhân thường gặp gây tiểu không tự chủ là gì?
- sys=0.0 oracle=1.0 judge2=0.2 refusal=hallucinated
- gold_overlap=0.13 max_rerank=0.6102883503777808 cited=[1, 2, 3]
- tool_queries: ['Các nguyên nhân thường gặp gây tiểu không tự chủ là gì?']
- answer: ### Nguyên nhân thường gặp gây tiểu không tự chủ

Theo các tài liệu được cung cấp, nguyên nhân gây tiểu không tự chủ (chủ yếu đề cập đến **đái dầm** ở trẻ em và tiểu không tự chủ do **sảng** ở người cao tuổi) bao gồm:

- **Chậm phát triển thần kinh cơ và nhận thức** trong kiểm soát bàng quang, ảnh h

### prod_corpus_multihop-6 — `generation_miss`

- Q: Các thụ thể nào trên bề mặt tế bào hủy xương đóng vai trò trong cơ chế cận tiết giữa hủy cốt bào và tạo cốt bào trong quá trình tái tạo xương?
- sys=0.0 oracle=1.0 judge2=0.0 refusal=hallucinated
- gold_overlap=1.00 max_rerank=0.7260329152105172 cited=[5]
- tool_queries: ['Thụ thể trên bề mặt tế bào hủy xương liên quan đến cơ chế cận tiết', 'Cận tiết giữa hủy cốt bào và tạo cốt bào trong tái tạo xương', 'Các thụ thể nào trên bề mặt tế bào hủy xương đóng vai trò trong cơ chế cận tiết giữa hủy cốt bào và tạo cốt bào trong quá trình tái tạo xương?']
- answer: Các thụ thể trên bề mặt tế bào hủy xương tham gia cơ chế cận tiết giữa hủy cốt bào và tạo cốt bào trong quá trình tái tạo xương bao gồm:
- **Thụ thể vitronectin** (integrin αVβ3): giúp hủy cốt bào bám dính vào osteopontin trong chất nền xương, tạo điều kiện cho quá trình tiêu xương và giải phóng các

