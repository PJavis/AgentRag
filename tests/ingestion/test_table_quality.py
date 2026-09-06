"""Tests for the probe-scoped table-quality gate.

Fixtures are shaped after real detections in the medical corpus — see
docs/eval/table_probe_corpus_survey_2026-09-06.md.
"""

import pytest

from src.agentrag.ingestion.parsers.table_quality import (
    classify_table,
    estimate_tokens,
    is_data_grid,
    is_safe_to_markdown,
    render_markdown,
)

# Real detection: 2f499de4 page 8, a skills-assessment grid (1/0 per column).
REAL_GRID = [
    ["STT", "Tên kỹ năng", "Quan sát", "Làm đúng", "Làm thành thạo"],
    ["1.1", "Chuẩn bị người bệnh", "1", "1", "1"],
    ["1.2", "Chuẩn bị điều dưỡng", "1", "1", "1"],
    ["1.4", "Đặt thông tiểu nam/nữ", "1", "1", "0"],
]

# Real detection: 881e7c3a page 134, a bordered prose box; to_markdown mirrors
# the paragraph into both columns and invents a "Col2" header.
MIRRORED_PROSE = [
    ["1. ĐỊNH NGHĨA", "Col2"],
    ["Lo lắng là hiện tượng phản ứng", "Lo lắng là hiện tượng phản ứng"],
    ["dọa của tự nhiên, xã hội mà con", "dọa của tự nhiên, xã hội mà con"],
]


def test_real_numeric_grid_classifies_as_data():
    assert classify_table(REAL_GRID) == "real_data"
    assert is_data_grid(REAL_GRID)
    assert is_safe_to_markdown(REAL_GRID)


def test_mirrored_layout_box_is_rejected():
    assert classify_table(MIRRORED_PROSE) == "layout_dup"
    assert not is_safe_to_markdown(MIRRORED_PROSE)
    assert not is_data_grid(MIRRORED_PROSE)


def test_paragraph_cells_classify_as_prose():
    prose = [
        ["Xét nghiệm nước tiểu, xét nghiệm tìm chất ma tuý, huyết thanh chẩn đoán giang mai và các chỉ định khác", ""],
        ["Trắc nghiệm tâm lý: nhóm trắc nghiệm tâm lý đánh giá lo âu Zung Hamilton và đánh giá trầm cảm phối hợp", ""],
        ["cách MMPI EPI đánh giá rối loạn giấc ngủ PSQI và các thang đo bổ sung khác cho người bệnh nội trú", ""],
    ]
    assert classify_table(prose) == "layout_prose"
    assert not is_safe_to_markdown(prose)


def test_text_only_grid_is_safe_but_not_a_data_grid():
    """A numberless comparison matrix must still survive arm B's gate."""
    grid = [
        ["Thuốc", "Chỉ định", "Chống chỉ định"],
        ["Thuốc A", "Nhiễm khuẩn", "Dị ứng"],
        ["Thuốc B", "Sốt cao", "Suy gan"],
    ]
    assert classify_table(grid) == "nonnumeric"
    assert is_safe_to_markdown(grid)
    assert not is_data_grid(grid)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        None,
        [["only one row", "x"]],
        [["single col"], ["single col"]],
        [["", ""], ["", ""]],
    ],
)
def test_degenerate_shapes_rejected(rows):
    assert classify_table(rows) == "degenerate"
    assert not is_safe_to_markdown(rows)


def test_none_cells_are_tolerated():
    rows = [["STT", None, "Liều"], ["1", None, "10mg"], ["2", None, "20mg"]]
    assert classify_table(rows) == "real_data"


# --- Regressions found 2026-09-06 by review of the shipped gate -------------

# Real detection shape: PyMuPDF reports 2 columns but `extract()` populates one
# and returns None for the other. `to_markdown()` is what mirrors the text into
# both columns and invents the "Col2" header — the gate never saw either.
SINGLE_COLUMN = [
    ["1. ĐỊNH NGHĨA", None],
    ["Lo lắng là phản ứng của cơ thể", None],
    ["trước một mối đe doạ nào đó", None],
]


def test_detected_columns_are_not_populated_columns():
    """44 of 90 gate-passing corpus detections filled at most one cell per row.

    `max(len(row))` counted the None placeholders, so a one-column prose strip
    read as a 2-column table and passed. There is no alignment to restore here.
    """
    assert classify_table(SINGLE_COLUMN) == "single_column"
    assert not is_safe_to_markdown(SINGLE_COLUMN)
    assert not is_data_grid(SINGLE_COLUMN)


def test_one_structured_row_is_not_a_grid():
    rows = [["header a", "header b"], ["value", None], ["other", None]]
    assert classify_table(rows) == "single_column"


def test_two_structured_rows_are_enough():
    rows = [["Thuốc", "Chỉ định"], ["A", "Sốt"], ["B", None]]
    assert classify_table(rows) == "nonnumeric"
    assert is_safe_to_markdown(rows)


def test_prose_mirrored_across_adjacent_columns_is_rejected():
    """PyMuPDF duplicates per column-pair, not only across a whole row."""
    rows = [
        ["Tiêu chuẩn chẩn đoán rối loạn", "Tiêu chuẩn chẩn đoán rối loạn", "ghi chú"],
        ["Biểu hiện lo âu kéo dài sáu tháng", "Biểu hiện lo âu kéo dài sáu tháng", "x"],
        ["Kèm theo rối loạn giấc ngủ nặng", "Kèm theo rối loạn giấc ngủ nặng", "y"],
    ]
    assert classify_table(rows) == "layout_dup"
    assert not is_safe_to_markdown(rows)


def test_repeated_short_scores_are_data_not_mirroring():
    """A skills grid legitimately repeats `1` across adjacent score columns."""
    assert classify_table(REAL_GRID) == "real_data"


def test_prose_mentioning_a_year_is_not_a_data_grid():
    """`\\d` anywhere promoted sentences to real_data and made them probe targets."""
    rows = [
        ["Nội dung hướng dẫn", "Ghi chú"],
        [
            "Thông tư ban hành năm 2020 quy định về quản lý chất lượng dịch vụ",
            "áp dụng cho toàn bộ các khoa lâm sàng trong bệnh viện",
        ],
        [
            "Người bệnh được tư vấn đầy đủ trước khi thực hiện thủ thuật",
            "theo quy định hiện hành của đơn vị",
        ],
    ]
    assert classify_table(rows) == "nonnumeric"
    assert not is_data_grid(rows)


def test_short_measurement_cells_still_count_as_numeric():
    rows = [["Thuốc", "Liều"], ["A", "10mg"], ["B", "0.5%"]]
    assert classify_table(rows) == "real_data"


# --- render_markdown: arm B emits THIS, not table.to_markdown() -------------


def test_render_uses_the_documents_own_header_never_coln():
    md = render_markdown(REAL_GRID)
    assert md.splitlines()[0] == "| STT | Tên kỹ năng | Quan sát | Làm đúng | Làm thành thạo |"
    assert "Col2" not in md


def test_render_returns_empty_for_anything_the_gate_rejects():
    assert render_markdown(MIRRORED_PROSE) == ""
    assert render_markdown(SINGLE_COLUMN) == ""
    assert render_markdown([]) == ""


def test_render_repeats_the_header_so_a_split_cannot_orphan_rows():
    rows = [["STT", "Liều"]] + [[str(i), f"{i}mg"] for i in range(1, 20)]
    md = render_markdown(rows, rows_per_block=8)
    blocks = md.split("\n\n")
    assert len(blocks) == 3  # 19 data rows / 8
    for block in blocks:
        assert block.startswith("| STT | Liều |")
        assert block.splitlines()[1] == "| --- | --- |"


def test_render_escapes_pipes_and_flattens_newlines():
    rows = [["a|b", "c"], ["line1\nline2", "1"], ["x", "2"]]
    md = render_markdown(rows)
    assert r"a\|b" in md
    assert "line1 line2" in md
    assert len(md.splitlines()) == 4  # header, separator, 2 data rows


def test_render_pads_short_rows_to_the_table_width():
    rows = [["a", "b", "c"], ["1", "2", "3"], ["4", "5"]]
    md = render_markdown(rows)
    assert md.splitlines()[-1] == "| 4 | 5 |  |"


def test_render_packs_blocks_to_a_token_budget():
    """Fixed row counts gave 200–1900 token blocks; half overflowed the window."""
    rows = [["STT", "Nội dung đánh giá kỹ năng", "Điểm"]] + [
        [str(i), f"bước thực hiện chi tiết số {i} của quy trình", "1"]
        for i in range(1, 61)
    ]
    md = render_markdown(rows, max_tokens=128)
    blocks = md.split("\n\n")
    assert len(blocks) > 1
    for block in blocks:
        assert block.startswith("| STT | Nội dung đánh giá kỹ năng | Điểm |")
        assert estimate_tokens(block) <= 128
    # no row lost or duplicated across the split
    data_lines = [ln for b in blocks for ln in b.splitlines()[2:]]
    assert len(data_lines) == 60
    assert all(f"| {i} |" in md for i in range(1, 61))


def test_render_emits_an_oversized_single_row_whole():
    """Splitting one row would destroy the alignment the block packing protects."""
    rows = [["a", "b"], ["x " * 400, "y"]]
    md = render_markdown(rows, max_tokens=16)
    assert len([ln for ln in md.splitlines() if ln.startswith("| x")]) == 1
