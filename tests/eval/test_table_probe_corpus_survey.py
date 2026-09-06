"""Tests for the table-probe corpus survey. No real PDFs, no fitz."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.eval import table_probe_corpus_survey as survey


def _fake_table(rows, row_count=None, col_count=None):
    t = MagicMock()
    t.extract.return_value = rows
    t.row_count = row_count if row_count is not None else len(rows)
    t.col_count = col_count if col_count is not None else max(len(r) for r in rows)
    return t


def _fake_page(text, tables):
    page = MagicMock()
    page.get_text.return_value = text
    page.find_tables.return_value = MagicMock(tables=tables)
    return page


def _fake_doc(pages):
    doc = MagicMock()
    doc.__iter__.return_value = iter(pages)
    return doc


GRID = [["STT", "Liều"], ["1", "10mg"], ["2", "20mg"]]
MIRRORED = [["A", "Col2"], ["same text here", "same text here"], ["x y", "x y"]]


def test_unique_by_content_collapses_identical_bytes(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"SAME")
    (tmp_path / "b.pdf").write_bytes(b"SAME")
    (tmp_path / "c.pdf").write_bytes(b"OTHER")

    unique, groups = survey.unique_by_content(sorted(tmp_path.glob("*.pdf")))

    assert len(unique) == 2
    assert sorted(len(v) for v in groups.values()) == [1, 2]


def test_survey_pdf_separates_data_grids_from_layout_boxes():
    doc = _fake_doc([
        _fake_page("x" * 500, [_fake_table(GRID), _fake_table(MIRRORED)]),
        _fake_page("", []),  # scanned page: no text layer
    ])
    with patch.object(survey, "_open", return_value=doc):
        rep = survey.survey_pdf("x.pdf")

    assert rep["pages"] == 2
    assert rep["text_layer_pages"] == 1
    assert rep["detected_tables"] == 2
    assert rep["data_grid_count"] == 1
    assert rep["kinds"]["layout_dup"] == 1


def test_survey_pdf_reports_unreadable_file():
    with patch.object(survey, "_open", side_effect=RuntimeError("bad pdf")):
        assert "error" in survey.survey_pdf("x.pdf")


def test_survey_pdf_survives_find_tables_error():
    page = MagicMock()
    page.get_text.return_value = "x" * 500
    page.find_tables.side_effect = RuntimeError("no table layer")
    with patch.object(survey, "_open", return_value=_fake_doc([page])):
        rep = survey.survey_pdf("x.pdf")
    assert rep["detected_tables"] == 0
    assert rep["pages"] == 1


def test_survey_corpus_counts_unique_docs_only(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"SAME")
    (tmp_path / "b.pdf").write_bytes(b"SAME")
    doc = _fake_doc([_fake_page("x" * 500, [_fake_table(GRID)])])
    with patch.object(survey, "_open", side_effect=lambda p: _fake_doc(
        [_fake_page("x" * 500, [_fake_table(GRID)])]
    )):
        rep = survey.survey_corpus(str(tmp_path))

    assert rep["files_on_disk"] == 2
    assert rep["unique_documents"] == 1
    assert rep["redundant_copies"] == 1
    # the duplicate must not be double-counted into the table totals
    assert rep["data_grids"] == 1
    assert doc is not None


def test_format_summary_mentions_key_numbers(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"ONE")
    with patch.object(survey, "_open", side_effect=lambda p: _fake_doc(
        [_fake_page("x" * 500, [_fake_table(GRID)])]
    )):
        rep = survey.survey_corpus(str(tmp_path))
    out = survey.format_summary(rep)
    assert "unique documents" in out
    assert "data grids" in out


# --- Regressions found 2026-09-06 by review of the shipped survey -----------


def test_one_unreadable_page_does_not_abort_the_document():
    """`doc.close()` sat only on the success path and get_text was unguarded.

    One page PyMuPDF cannot decode used to propagate out of survey_pdf with the
    document still open, killing a whole-corpus run on document 27 of 29.
    """
    good = _fake_page("x" * 200, [_fake_table(GRID)])
    bad = _fake_page("x" * 200, [])
    bad.get_text.side_effect = RuntimeError("mupdf: cannot decode page")
    doc = _fake_doc([good, bad, good])

    with patch.object(survey, "_open", return_value=doc):
        out = survey.survey_pdf("broken.pdf")

    assert "error" not in out
    assert out["pages"] == 3
    assert out["data_grid_count"] == 2
    assert any("page 2" in e for e in out["page_errors"])
    doc.close.assert_called_once()


def test_document_is_closed_even_when_the_page_iterator_dies():
    doc = MagicMock()
    doc.__iter__.side_effect = RuntimeError("mupdf: broken xref")
    with patch.object(survey, "_open", return_value=doc):
        out = survey.survey_pdf("broken.pdf")
    assert "error" not in out and out["pages"] == 0
    doc.close.assert_called_once()


def test_find_tables_failure_still_counts_the_page_text_layer():
    page = _fake_page("y" * 200, [])
    page.find_tables.side_effect = RuntimeError("no table layer")
    with patch.object(survey, "_open", return_value=_fake_doc([page])):
        out = survey.survey_pdf("x.pdf")
    assert out["pages"] == 1 and out["text_layer_pages"] == 1


def test_survey_corpus_survives_a_document_that_blows_up(tmp_path):
    (tmp_path / "ok.pdf").write_bytes(b"A")
    (tmp_path / "bad.pdf").write_bytes(b"B")

    def _fake_survey(path):
        if path.endswith("bad.pdf"):
            raise RuntimeError("boom")
        return {"doc": "ok.pdf", "pages": 1, "text_layer_pages": 1,
                "detected_tables": 0, "kinds": {}, "data_grids": [],
                "data_grid_count": 0}

    with patch.object(survey, "survey_pdf", side_effect=_fake_survey):
        rep = survey.survey_corpus(str(tmp_path))

    assert rep["unique_documents"] == 2
    assert len(rep["errors"]) == 1 and rep["errors"][0]["doc"] == "bad.pdf"


def test_unique_by_content_skips_an_unreadable_file(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"A")
    missing = tmp_path / "gone.pdf"
    unique, _ = survey.unique_by_content(
        [tmp_path / "a.pdf", missing], tmp_path
    )
    assert [p.name for p in unique] == ["a.pdf"]


def test_duplicate_groups_keep_membership_and_subdirectory_paths(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "same.pdf").write_bytes(b"DUP")
    (tmp_path / "sub" / "same.pdf").write_bytes(b"DUP")
    _, groups = survey.unique_by_content(sorted(tmp_path.rglob("*.pdf")), tmp_path)
    members = next(m for m in groups.values() if len(m) > 1)
    assert sorted(members) == ["same.pdf", "sub/same.pdf"]


def test_survey_corpus_emits_the_unique_document_list(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"A")
    (tmp_path / "b.pdf").write_bytes(b"A")  # duplicate of a
    (tmp_path / "c.pdf").write_bytes(b"C")

    with patch.object(survey, "_open", return_value=_fake_doc([])):
        rep = survey.survey_corpus(str(tmp_path))

    assert rep["unique_documents"] == 2
    assert rep["unique_documents_list"] == ["a.pdf", "c.pdf"]


def test_write_dedupe_dir_materialises_the_unique_corpus(tmp_path):
    src = tmp_path / "originals"
    src.mkdir()
    (src / "a.pdf").write_bytes(b"A")
    (src / "b.pdf").write_bytes(b"A")
    out = tmp_path / "unique"

    n = survey.write_dedupe_dir(str(src), ["a.pdf"], str(out))

    assert n == 1
    assert sorted(p.name for p in out.iterdir()) == ["a.pdf"]
    assert (out / "a.pdf").read_bytes() == b"A"


def test_data_grids_record_measured_render_tokens():
    """Cell count predicts rendered size badly; rank_targets filters on this."""
    with patch.object(survey, "_open", return_value=_fake_doc([
        _fake_page("x" * 200, [_fake_table(GRID)])
    ])):
        out = survey.survey_pdf("x.pdf")
    grid = out["data_grids"][0]
    assert grid["est_tokens"] > 0


# --- write_dedupe_dir data-loss guards (found 2026-09-06 by verification) ---


def test_dedupe_dir_refuses_to_write_into_the_corpus(tmp_path):
    """--corpus defaults to data/originals and sits next to --dedupe-dir in the
    usage block. Clearing dest when dest IS the corpus deleted every original."""
    src = tmp_path / "originals"
    src.mkdir()
    (src / "a.pdf").write_bytes(b"A")

    with pytest.raises(ValueError, match="must not be the corpus"):
        survey.write_dedupe_dir(str(src), ["a.pdf"], str(src))

    assert (src / "a.pdf").read_bytes() == b"A"


def test_dedupe_dir_keeps_two_distinct_documents_with_the_same_basename(tmp_path):
    """The second symlink_to raised FileExistsError (an OSError), and the
    write_bytes fallback then wrote THROUGH the first symlink, truncating a
    source PDF inside the corpus."""
    src = tmp_path / "originals"
    (src / "a").mkdir(parents=True)
    (src / "b").mkdir(parents=True)
    (src / "a" / "x.pdf").write_bytes(b"AAA")
    (src / "b" / "x.pdf").write_bytes(b"BBB")
    out = tmp_path / "unique"

    n = survey.write_dedupe_dir(str(src), ["a/x.pdf", "b/x.pdf"], str(out))

    assert n == 2
    assert len(list(out.iterdir())) == 2
    assert {p.read_bytes() for p in out.iterdir()} == {b"AAA", b"BBB"}
    # the corpus is untouched
    assert (src / "a" / "x.pdf").read_bytes() == b"AAA"
    assert (src / "b" / "x.pdf").read_bytes() == b"BBB"


def test_dedupe_dir_reports_a_missing_source_instead_of_counting_it(tmp_path):
    src = tmp_path / "originals"
    src.mkdir()
    out = tmp_path / "unique"
    with pytest.raises(FileNotFoundError, match="missing"):
        survey.write_dedupe_dir(str(src), ["ghost.pdf"], str(out))


def test_dedupe_dir_replaces_stale_entries_without_following_them(tmp_path):
    src = tmp_path / "originals"
    src.mkdir()
    (src / "a.pdf").write_bytes(b"A")
    out = tmp_path / "unique"
    out.mkdir()
    (out / "old.pdf").symlink_to(src / "a.pdf")

    survey.write_dedupe_dir(str(src), ["a.pdf"], str(out))

    assert sorted(p.name for p in out.iterdir()) == ["a.pdf"]
    assert (src / "a.pdf").read_bytes() == b"A"


def test_an_unreadable_file_is_named_not_counted_as_a_duplicate(tmp_path):
    """A skipped file inflated redundant_copies (files_on_disk - unique) and was
    reported to the operator as a duplicate that does not exist."""
    (tmp_path / "a.pdf").write_bytes(b"A")
    (tmp_path / "b.pdf").write_bytes(b"B")
    locked = tmp_path / "locked.pdf"
    locked.write_bytes(b"C")
    locked.chmod(0o000)

    try:
        with patch.object(survey, "_open", return_value=_fake_doc([])):
            rep = survey.survey_corpus(str(tmp_path))
    finally:
        locked.chmod(0o644)

    assert rep["files_on_disk"] == 3
    assert rep["unique_documents"] == 2
    assert rep["redundant_copies"] == 0
    assert rep["duplicate_groups"] == {}
    assert any("locked.pdf" in u for u in rep["unreadable_files"])


def test_data_grids_record_the_largest_block_not_just_the_total():
    wide = [["STT", "Nội dung đánh giá kỹ năng chi tiết", "Điểm"]] + [
        [str(i), f"bước thực hiện số {i} của quy trình", "1"] for i in range(1, 40)
    ]
    with patch.object(survey, "_open", return_value=_fake_doc([
        _fake_page("x" * 200, [_fake_table(wide)])
    ])):
        out = survey.survey_pdf("x.pdf")
    grid = out["data_grids"][0]
    assert grid["max_block_tokens"] <= survey.CHUNK_MAX_TOKENS
    assert grid["max_block_tokens"] < grid["est_tokens"]


def test_corpus_docs_sha_is_content_based_and_arm_independent(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"A")
    (tmp_path / "b.pdf").write_bytes(b"A")  # duplicate — must not shift the hash
    with patch.object(survey, "_open", return_value=_fake_doc([])):
        first = survey.survey_corpus(str(tmp_path))["corpus_docs_sha"]
        again = survey.survey_corpus(str(tmp_path))["corpus_docs_sha"]
    assert first == again and len(first) == 12

    (tmp_path / "c.pdf").write_bytes(b"C")  # a NEW document must shift it
    with patch.object(survey, "_open", return_value=_fake_doc([])):
        after = survey.survey_corpus(str(tmp_path))["corpus_docs_sha"]
    assert after != first


def test_dedupe_dir_refuses_a_directory_it_does_not_own(tmp_path):
    """`--dedupe-dir data/eval` (one path segment dropped) used to unlink every
    eval artefact in that directory and exit 0.

    Asserts on BYTES, not just names: the failure mode being guarded is silent
    deletion, so "the file is still listed" is not enough.
    """
    src = tmp_path / "originals"
    src.mkdir()
    (src / "a.pdf").write_bytes(b"A")

    out = tmp_path / "eval"
    out.mkdir()
    victims = {
        "benchmark_v3.json": b'{"scores": [0.81, 0.74]}',
        "run.log": b"ablation run 2026-08-01\n",
        "notes.md": b"# do not delete",
        "goldset.jsonl": b'{"q": "..."}\n',
        # a PDF sitting in the same directory must ALSO survive the refusal
        "unrelated.pdf": b"%PDF-1.4 keep me",
    }
    for name, body in victims.items():
        (out / name).write_bytes(body)
    (out / "subdir").mkdir()
    (out / "subdir" / "nested.json").write_bytes(b"{}")

    with pytest.raises(ValueError, match="non-PDF"):
        survey.write_dedupe_dir(str(src), ["a.pdf"], str(out))

    for name, body in victims.items():
        assert (out / name).read_bytes() == body, f"{name} was modified"
    assert (out / "subdir" / "nested.json").read_bytes() == b"{}"
    assert (src / "a.pdf").read_bytes() == b"A"


def test_dedupe_dir_error_names_the_files_it_refused_to_delete(tmp_path):
    """The operator has to be able to see WHY, or they will just retry."""
    src = tmp_path / "originals"
    src.mkdir()
    (src / "a.pdf").write_bytes(b"A")
    out = tmp_path / "eval"
    out.mkdir()
    (out / "benchmark.json").write_text("{}")

    with pytest.raises(ValueError) as exc:
        survey.write_dedupe_dir(str(src), ["a.pdf"], str(out))
    assert "benchmark.json" in str(exc.value)


def _case_insensitive_tmpdir() -> Path | None:
    """A real case-insensitive directory, if this machine has one."""
    import tempfile

    for root in ("/mnt/c/Windows/Temp", "/mnt/c/Temp", "/tmp"):
        base = Path(root)
        if not base.is_dir():
            continue
        try:
            d = Path(tempfile.mkdtemp(dir=str(base), prefix="ci_probe_"))
            (d / "Probe").mkdir()
            insensitive = (d / "probe").is_dir()
            shutil.rmtree(d, ignore_errors=True)
            if insensitive:
                return base
        except OSError:
            continue
    return None


def test_dedupe_dir_refuses_a_case_different_spelling_of_the_corpus(tmp_path, monkeypatch):
    """`Corpus/unique` vs `corpus/unique` — the scenario that motivated the
    inode check.

    Path comparison is case-SENSITIVE and `resolve()` does not case-normalise,
    so on a case-insensitive filesystem (/mnt/c under WSL, default macOS APFS)
    the two spellings named the same directory while comparing as different
    ones — and `write_dedupe_dir` cleared it, deleting a corpus PDF.

    A Linux tmpfs cannot produce that collision, so the filesystem's behaviour
    is simulated at the only place it matters: `stat()` returning the SAME
    (st_dev, st_ino) for both spellings. That is precisely what a
    case-insensitive filesystem does, and it is what `_identity` reads.
    """
    corpus = tmp_path / "Corpus"
    corpus.mkdir()
    (corpus / "keep_me.pdf").write_bytes(b"IRREPLACEABLE")
    (corpus / "unique").mkdir()
    (corpus / "unique" / "also_keep.pdf").write_bytes(b"ALSO IRREPLACEABLE")

    real_stat = Path.stat

    def case_insensitive_stat(self, *args, **kwargs):
        text = str(self)
        marker = str(tmp_path / "corpus")
        if text == marker or text.startswith(marker + "/"):
            return real_stat(Path(str(tmp_path / "Corpus") + text[len(marker):]),
                             *args, **kwargs)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", case_insensitive_stat)

    # Same directory, lowercase spelling. A string-only check sees a new path.
    dest = tmp_path / "corpus" / "unique"
    with pytest.raises(ValueError, match="must not be"):
        survey.write_dedupe_dir(str(corpus), ["keep_me.pdf"], str(dest))

    monkeypatch.undo()
    assert (corpus / "keep_me.pdf").read_bytes() == b"IRREPLACEABLE"
    assert (corpus / "unique" / "also_keep.pdf").read_bytes() == b"ALSO IRREPLACEABLE"


def test_overlaps_uses_inode_identity_not_string_comparison(tmp_path, monkeypatch):
    """Unit-level: the mechanism itself, independent of write_dedupe_dir."""
    corpus = tmp_path / "Corpus"
    (corpus / "unique").mkdir(parents=True)

    assert survey._overlaps(tmp_path / "elsewhere", corpus) is False

    real_stat = Path.stat

    def case_insensitive_stat(self, *args, **kwargs):
        text = str(self)
        marker = str(tmp_path / "corpus")
        if text == marker or text.startswith(marker + "/"):
            return real_stat(Path(str(tmp_path / "Corpus") + text[len(marker):]),
                             *args, **kwargs)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", case_insensitive_stat)
    assert survey._overlaps(tmp_path / "corpus", corpus) is True
    assert survey._overlaps(tmp_path / "corpus" / "unique", corpus) is True


@pytest.mark.skipif(
    _case_insensitive_tmpdir() is None,
    reason="no case-insensitive filesystem available on this machine",
)
def test_dedupe_dir_refuses_case_difference_on_a_real_case_insensitive_fs():
    """Same guard, on a filesystem that really is case-insensitive."""
    import tempfile

    base = _case_insensitive_tmpdir()
    work = Path(tempfile.mkdtemp(dir=str(base), prefix="ci_dedupe_"))
    try:
        corpus = work / "Corpus"
        corpus.mkdir()
        (corpus / "keep_me.pdf").write_bytes(b"IRREPLACEABLE")
        # THIS is the file the pre-fix code deleted: it lives in the corpus
        # subdirectory that `corpus/unique` resolves to, so clearing "dest"
        # cleared it. Verified against the old string-only check on this very
        # filesystem — it does not survive without the inode guard.
        (corpus / "unique").mkdir()
        (corpus / "unique" / "also_keep.pdf").write_bytes(b"ALSO IRREPLACEABLE")

        with pytest.raises(ValueError, match="must not be"):
            survey.write_dedupe_dir(
                str(corpus), ["keep_me.pdf"], str(work / "corpus" / "unique")
            )

        assert (corpus / "keep_me.pdf").read_bytes() == b"IRREPLACEABLE"
        assert (corpus / "unique" / "also_keep.pdf").read_bytes() == b"ALSO IRREPLACEABLE"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_dedupe_dir_allows_sibling_directories(tmp_path):
    src = tmp_path / "originals"
    src.mkdir()
    (src / "a.pdf").write_bytes(b"A")
    assert survey.write_dedupe_dir(str(src), ["a.pdf"], str(tmp_path / "unique")) == 1


def test_candidate_tables_are_the_eligible_set_not_just_data_grids():
    """`real_data` is the RANKING class; arm B rewrites `nonnumeric` too.

    Treating data_grids as the eligible set undercounts the probe's usable pool
    by a third. See docs/eval/table_probe_power_analysis_2026-09-06.md.
    """
    numeric = [["STT", "Liều", "Số ca"], ["1", "10mg", "12"], ["2", "20mg", "30"]]
    text_matrix = [
        ["Thuốc", "Chỉ định", "Chống chỉ định"],
        ["A", "Nhiễm khuẩn", "Dị ứng"],
        ["B", "Sốt cao", "Suy gan"],
    ]
    prose = [["Một đoạn văn dài", None], ["tiếp tục đoạn văn", None]]

    with patch.object(survey, "_open", return_value=_fake_doc([
        _fake_page("x" * 200, [_fake_table(numeric), _fake_table(text_matrix),
                               _fake_table(prose)])
    ])):
        out = survey.survey_pdf("x.pdf")

    kinds = {c["kind"] for c in out["candidate_tables"]}
    assert kinds == {"real_data", "nonnumeric"}      # prose is gated out
    assert out["candidate_count"] == 2
    assert out["data_grid_count"] == 1               # ranking class is narrower
    for c in out["candidate_tables"]:
        assert c["structured_rows"] >= 2 and c["cols"] == 3
        assert c["max_block_tokens"] <= c["est_tokens"]


def test_structured_candidates_counts_only_question_worthy_tables(tmp_path):
    thin = [["a", "b"], ["1", "2"]]           # 2 cols -> cannot ask alignment
    wide = [["a", "b", "c"], ["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]
    (tmp_path / "d.pdf").write_bytes(b"D")
    with patch.object(survey, "_open", return_value=_fake_doc([
        _fake_page("x" * 200, [_fake_table(thin), _fake_table(wide)])
    ])):
        rep = survey.survey_corpus(str(tmp_path))
    assert rep["candidate_tables"] == 2
    assert rep["structured_candidates"] == 1
