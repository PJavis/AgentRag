"""0b — retrieval-only rank probe. Tests pin the pure parts, not the outcome."""
import math

from scripts.eval.table_probe_retrieval_ab import (
    bm25_scores,
    bootstrap_doc_clustered_ci,
    build_query,
    gold_chunk_ids,
    pick_gold_row,
    rank_of,
    reciprocal_rank,
    rrf_fuse,
    tokenize,
)

ROWS = [
    ["Thuốc", "Liều", "Đường dùng"],
    ["Paracetamol", "500 mg", "uống"],
    ["Ibuprofen", "400 mg", "uống"],
]


def test_pick_gold_row_takes_the_first_data_row_with_three_populated_cells():
    assert pick_gold_row(ROWS) == 1


def test_pick_gold_row_never_returns_the_header():
    thin = [["Thuốc", "Liều", "Đường"], ["Paracetamol", "", ""]]
    assert pick_gold_row(thin) is None


def test_build_query_pairs_the_row_label_with_a_column_header():
    q = build_query(ROWS, 1)
    assert q["row_label"] == "Paracetamol"
    assert q["column"] in ("Liều", "Đường dùng")
    assert q["row_label"] in q["query"] and q["column"] in q["query"]
    assert q["answer_cell"] in ("500 mg", "uống")


def test_build_query_refuses_a_table_whose_header_cell_is_empty():
    """Without a column name there is no alignment question to ask."""
    rows = [["Thuốc", "", ""], ["Paracetamol", "500 mg", "uống"]]
    assert build_query(rows, 1) is None


def test_gold_chunk_match_is_arm_neutral_bag_of_words_not_substring():
    """Arm A shreds cells across lines. If the GOLD definition needed contiguous
    text, arm A could never have a gold chunk and the probe would be rigged."""
    cells = ["Paracetamol", "500 mg", "uống", "người lớn", "mỗi 6 giờ"]
    shredded = "Paracetamol\nSau nhiễm\n500\nmg uống người lớn mỗi 6 giờ\n"
    assert gold_chunk_ids([{"content": shredded}], gold_cells=cells) == [0]


def test_gold_chunk_does_not_filter_on_the_chunker_page_label():
    """A chunk holding one page's tail and the next page's marker is labelled with
    the NEXT page. Filtering on that deleted arm B's gold chunks in a real run."""
    cells = ["Paracetamol", "500 mg", "uống", "người lớn", "mỗi 6 giờ"]
    mislabelled = [{"content": "Paracetamol 500 mg uống người lớn mỗi 6 giờ",
                    "page_start": 17, "page_end": 17}]
    assert gold_chunk_ids(mislabelled, gold_cells=cells) == [0]


def test_gold_chunk_needs_most_of_the_row_not_one_word():
    cells = ["Paracetamol", "500 mg", "uống", "người lớn", "mỗi 6 giờ"]
    assert gold_chunk_ids([{"content": "Paracetamol only"}], gold_cells=cells) == []


def test_a_row_too_generic_to_identify_is_dropped_not_guessed_at():
    from scripts.eval.table_probe_retrieval_ab import row_is_identifiable

    assert not row_is_identifiable(["1", "10mg", "uống"])
    assert row_is_identifiable(
        ["Paracetamol", "500 mg", "uống", "người lớn", "mỗi 6 giờ"]
    )
    assert gold_chunk_ids([{"content": "1 10mg uống"}], gold_cells=["1", "10mg", "uống"]) == []


def test_rank_of_is_one_based_and_none_when_absent():
    assert rank_of([7, 3, 9], {3}) == 2
    assert rank_of([7, 3, 9], {42}) is None
    assert reciprocal_rank(None) == 0.0
    assert reciprocal_rank(2) == 0.5


def test_tokenize_lowercases_and_keeps_vietnamese_diacritics():
    assert tokenize("Đường DÙNG, 500mg") == ["đường", "dùng", "500mg"]


def test_bm25_ranks_the_matching_document_first():
    docs = [tokenize("thuốc paracetamol liều 500 mg"), tokenize("phẫu thuật vùng bìu")]
    scores = bm25_scores(docs, tokenize("paracetamol liều"))
    assert scores[0] > scores[1]


def test_bm25_handles_a_query_term_absent_from_every_document():
    docs = [tokenize("a b"), tokenize("c d")]
    scores = bm25_scores(docs, tokenize("zzz"))
    assert scores == [0.0, 0.0]


def test_rrf_fuse_rewards_agreement_between_rankers():
    fused = rrf_fuse([[2, 0, 1], [2, 1, 0]], k=60)
    assert fused[0] == 2  # ranked first by both


def test_bootstrap_ci_resamples_documents_not_tables():
    """Resampling tables would treat 14 tables from one file as 14 draws."""
    per_doc = {"a.pdf": [1.0, 1.0, 1.0], "b.pdf": [0.0]}
    lo, hi = bootstrap_doc_clustered_ci(per_doc, iterations=200, seed=7)
    assert lo <= hi
    assert -0.001 <= lo and hi <= 1.001
    # Two clusters that disagree completely must produce a wide interval, never
    # the false precision that per-table resampling would report.
    assert hi - lo > 0.5


def test_bootstrap_ci_of_nothing_is_undefined_not_zero():
    assert bootstrap_doc_clustered_ci({}, iterations=10, seed=1) == (None, None)


def test_compare_reports_recall_and_never_invents_a_p_value_for_all_ties():
    from scripts.eval.table_probe_retrieval_ab import compare

    a = [{"doc": "d.pdf", "bm25_rank": 3}, {"doc": "d.pdf", "bm25_rank": None}]
    b = [{"doc": "d.pdf", "bm25_rank": 3}, {"doc": "d.pdf", "bm25_rank": None}]
    out = compare(a, b, ["bm25"])["retrievers"]["bm25"]
    assert out["wins"] == out["losses"] == 0 and out["ties"] == 2
    assert out["wilcoxon_p"] is None          # nothing to test, not "p = 1"
    assert out["recall_at_10_a"] == 0.5       # one ranked 3rd, one never retrieved
    assert out["not_retrieved_b"] == 1


def test_compare_counts_a_real_improvement():
    from scripts.eval.table_probe_retrieval_ab import compare

    a = [{"doc": "d.pdf", "bm25_rank": 20}]
    b = [{"doc": "d.pdf", "bm25_rank": 1}]
    out = compare(a, b, ["bm25"])["retrievers"]["bm25"]
    assert out["wins"] == 1 and out["losses"] == 0
    assert out["mean_delta"] > 0.9
