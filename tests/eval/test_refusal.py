from src.agentrag.eval.refusal import (
    classify_refusal,
    group_contexts,
    is_abstention,
    is_hallucination,
)


def test_group_contexts_merges_into_buckets():
    ctx = ["a", "b", "c", "d", "e"]
    out = group_contexts(ctx, group_size=2)
    assert len(out) == 3                      # [a+b], [c+d], [e]
    assert "a" in out[0] and "b" in out[0]
    assert out[2].strip() == "e"


def test_group_contexts_zero_means_one_per_doc():
    ctx = ["a", "b", "c"]
    assert group_contexts(ctx, group_size=0) == ["a", "b", "c"]


def test_is_abstention_true_on_uncertainty_and_no_citations():
    assert is_abstention("Tôi không tìm thấy thông tin trong tài liệu.", []) is True
    assert is_abstention("I don't have enough information.", []) is True


def test_is_abstention_false_on_confident_answer():
    assert is_abstention("Nhồi máu cơ tim là tắc động mạch vành [1].", [{"source": 1}]) is False


def test_is_abstention_confident_but_uncited_is_not_abstention():
    # confident claim with no citation = hallucination, NOT abstention
    assert is_abstention("Thủ đô nước Pháp là Paris.", []) is False


def test_is_hallucination_true_only_for_confident_answer():
    # confident answer to an out-of-corpus question = dangerous
    assert is_hallucination("Thủ đô nước Pháp là Paris.") is True
    # hedged answer is NOT a hallucination (even if it cited a distractor)
    assert is_hallucination("Ngữ cảnh không có thông tin về điều này.") is False
    assert is_hallucination("") is False


def test_classify_refusal_three_states():
    # ideal: hedged + no citation
    assert classify_refusal("Tôi không tìm thấy thông tin.", []) == "abstained"
    # soft: hedged but cited a distractor (the case the benchmark surfaced)
    assert classify_refusal("Ngữ cảnh không có thông tin, nhưng có nhắc Paris [1].",
                            [{"source": 1}]) == "hedged_cited"
    # dangerous: confident, no hedge
    assert classify_refusal("Thủ đô nước Pháp là Paris.", []) == "hallucinated"
    assert classify_refusal("", []) == "empty"


# Regression (2026-07-13): the live answer LLM refuses out-of-corpus questions
# with phrasings the old marker list missed ("... không có trong tài liệu",
# "không chứa thông tin", "không thể xác định"), so genuine refusals were
# mis-scored as hallucinations and kept their distractor citations. These are
# the EXACT strings the model emitted in the reproduction.
_REAL_REFUSALS = [
    "Thông tin về thuốc Blorbocide không có trong các tài liệu được cung cấp.",
    "Thông tin về tác dụng phụ của thuốc Flebotrin không có trong ngữ cảnh được cung cấp.",
    "Thông tin về quy trình phẫu thuật Mendoza-Lê không có trong tài liệu được cung cấp.",
    "Các tài liệu được cung cấp không chứa thông tin về kết quả thử nghiệm này.",
    "Thông tin về Protein XQ-7 không có trong các tài liệu. Do đó, không thể xác định vai trò của nó.",
    "The provided documents do not contain information on this drug.",
    "This topic is not mentioned in the provided context.",
]


def test_real_llm_refusals_are_not_hallucinations():
    for r in _REAL_REFUSALS:
        assert is_hallucination(r) is False, f"mis-scored as hallucination: {r!r}"


def test_real_llm_refusals_are_abstentions_without_citations():
    for r in _REAL_REFUSALS:
        assert is_abstention(r, []) is True, f"not recognized as abstention: {r!r}"


def test_real_llm_refusals_with_distractor_citations_are_hedged_not_hallucinated():
    # thin-context refusal that carried distractor citations must classify as the
    # soft 'hedged_cited', never the dangerous 'hallucinated'
    for r in _REAL_REFUSALS:
        assert classify_refusal(r, [{"source": 1}]) == "hedged_cited", r
