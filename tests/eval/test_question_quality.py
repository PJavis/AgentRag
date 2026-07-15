from src.agentrag.eval.question_quality import is_context_dependent


# The 6 eval-set-artifact questions from the 2026-07-14 home run that inflated
# the retrieval_miss / generation_miss buckets — all must be flagged.
BROKEN = [
    "Tài liệu học tập của môn học này là gì?",
    "Nội dung các câu hỏi từ câu 8 đến câu 12 trong đề thi là gì?",
    "Nội dung cụ thể của câu hỏi số 18 đến 21 trong đề thi giải phẫu là gì?",
    "Mô học xương đùi của bệnh nhân này cho thấy gì?",
    "Đoạn văn này chứa những câu hỏi gì?",
    "Những câu hỏi nào được đưa ra trong đoạn văn?",
    # 2026-07-15 clean-rebuild: bare "câu N" exam-item refs + "tình huống trên"
    # dangling scenario slipped past the first filter.
    "Câu 6 hỏi gì?",
    "Bệnh nhân trong câu 12 có tiền sử dị ứng với loại thuốc nào?",
    "Bệnh nhân trong tình huống trên có triệu chứng dị cảm ở vị trí nào?",
]

# Genuine standalone medical questions from the same run — must NOT be flagged.
GOOD = [
    "Những dấu hiệu lâm sàng và cận lâm sàng trong cai nghiện ma túy là gì?",
    "Các bước thực hiện sau khi rút ống thông tiểu cho người bệnh là gì?",
    "Quản lý và tiên lượng của rối loạn tâm thần trong bệnh gan là gì?",
    "Metformin được chỉ định trong điều trị bệnh gì?",
    "Liều dùng paracetamol cho trẻ em là bao nhiêu?",
    # legit anatomical "trên bề mặt" / self-contained vignette — must NOT be flagged
    "Các thụ thể nào trên bề mặt tế bào hủy xương đóng vai trò trong cơ chế cận tiết?",
    "Một phụ nữ 22 tuổi yếu cẳng chân trái sau tai nạn xe máy, sinh thiết cơ cho thấy gì?",
]


def test_flags_all_known_broken_questions():
    for q in BROKEN:
        bad, reason = is_context_dependent(q)
        assert bad, f"should flag: {q}"
        assert reason  # non-empty explanation


def test_passes_all_genuine_questions():
    for q in GOOD:
        bad, reason = is_context_dependent(q)
        assert not bad, f"should pass: {q} (flagged: {reason})"


def test_meta_exam_reference_flagged():
    assert is_context_dependent("Đáp án câu 3 trong đề thi?")[0]


def test_dangling_demonstrative_flagged():
    assert is_context_dependent("Chẩn đoán của bệnh nhân này là gì?")[0]
    assert is_context_dependent("Nội dung tài liệu này gồm những gì?")[0]


def test_context_reference_flagged():
    assert is_context_dependent("Theo ngữ cảnh được cung cấp, chất X có tác dụng gì?")[0]


def test_disease_named_this_not_flagged():
    # "trong bệnh gan" is a real disease reference, not a dangling demonstrative
    bad, _ = is_context_dependent("Biến chứng thần kinh trong bệnh gan là gì?")
    assert not bad


def test_empty_and_short():
    assert is_context_dependent("")[0]
    assert is_context_dependent("là gì?")[0]  # too short to be a real question
