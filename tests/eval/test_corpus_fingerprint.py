from src.agentrag.eval.corpus_fingerprint import (
    check_fingerprint, fingerprint_from_pairs, read_evalset_fingerprint,
)


def test_fingerprint_deterministic_and_order_insensitive():
    a = fingerprint_from_pairs([("doc-b.pdf", 10), ("doc-a.pdf", 5)])
    b = fingerprint_from_pairs([("doc-a.pdf", 5), ("doc-b.pdf", 10)])
    assert a == b
    assert len(a) == 12


def test_fingerprint_changes_on_corpus_change():
    base = fingerprint_from_pairs([("doc-a.pdf", 5)])
    assert fingerprint_from_pairs([("doc-a.pdf", 6)]) != base       # re-ingest changed segs
    assert fingerprint_from_pairs([("doc-a.pdf", 5), ("new.pdf", 1)]) != base  # new doc


def test_fingerprint_empty_corpus():
    assert fingerprint_from_pairs([]) == fingerprint_from_pairs([])


def test_check_fingerprint_match_passes():
    ok, msg = check_fingerprint(evalset_fp="abc123", live_fp="abc123", allow_mismatch=False)
    assert ok and msg == ""


def test_check_fingerprint_mismatch_blocks_with_reason():
    ok, msg = check_fingerprint(evalset_fp="abc123", live_fp="def456", allow_mismatch=False)
    assert not ok
    assert "abc123" in msg and "def456" in msg


def test_check_fingerprint_mismatch_allowed_warns():
    ok, msg = check_fingerprint(evalset_fp="abc123", live_fp="def456", allow_mismatch=True)
    assert ok
    assert "mismatch" in msg.lower()


def test_check_fingerprint_unstamped_evalset_warns_only():
    ok, msg = check_fingerprint(evalset_fp=None, live_fp="def456", allow_mismatch=False)
    assert ok
    assert "no corpus fingerprint" in msg.lower()


def test_read_evalset_fingerprint(tmp_path):
    p = tmp_path / "set.jsonl"
    p.write_text('{"id":"x","question":"q","corpus_fp":"abc123def456"}\n', encoding="utf-8")
    assert read_evalset_fingerprint(str(p)) == "abc123def456"


def test_read_evalset_fingerprint_absent(tmp_path):
    p = tmp_path / "set.jsonl"
    p.write_text('{"id":"x","question":"q"}\n', encoding="utf-8")
    assert read_evalset_fingerprint(str(p)) is None
