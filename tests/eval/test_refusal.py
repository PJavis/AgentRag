from src.agentrag.eval.refusal import group_contexts, is_abstention


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
