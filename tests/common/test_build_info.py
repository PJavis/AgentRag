"""Provenance must be visible in logs and artefacts, and must never break a run.

The risk being covered: docker-compose.deploy.yml and docker-compose.fullstack.yml
are both invoked with `-f`, which disables auto-merge of docker-compose.override.yml
and so of the ./src bind mount. A stale image on those paths produces a complete,
plausible result set with no error.
"""

import logging

import pytest

from src.agentrag.common import build_info as bi


@pytest.fixture(autouse=True)
def _clear_caches():
    bi.build_info.cache_clear()
    bi.source_sha.cache_clear()
    yield
    bi.build_info.cache_clear()
    bi.source_sha.cache_clear()


def _write_tree(root, files):
    for name, body in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")


def test_source_sha_is_content_addressed_and_path_relative(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    files = {"x.py": "print(1)\n", "pkg/y.py": "VALUE = 2\n"}
    _write_tree(a, files)
    _write_tree(b, files)
    # Same content under a different absolute path must hash the same, so the
    # figure is comparable between the image (/app/src) and a host checkout.
    assert bi.source_sha(str(a)) == bi.source_sha(str(b))

    bi.source_sha.cache_clear()
    (b / "pkg" / "y.py").write_text("VALUE = 3\n", encoding="utf-8")
    assert bi.source_sha(str(a)) != bi.source_sha(str(b))


def test_source_sha_ignores_pycache_and_non_python(tmp_path):
    _write_tree(tmp_path, {"x.py": "print(1)\n"})
    first = bi.source_sha(str(tmp_path))
    bi.source_sha.cache_clear()
    _write_tree(tmp_path, {"__pycache__/x.cpython-311.pyc": "junk",
                           "notes.md": "unrelated"})
    assert bi.source_sha(str(tmp_path)) == first


def test_mismatch_between_running_source_and_image_is_reported(monkeypatch, tmp_path):
    """The whole point: a shadowed or stale /app/src must be visible."""
    monkeypatch.setenv("AGENTRAG_GIT_SHA", "abc1234")
    monkeypatch.setenv("AGENTRAG_BUILD_ID", "ci-99")
    monkeypatch.setenv("AGENTRAG_SOURCE_SHA", "deadbeefcafe")  # what the image shipped

    info = bi.build_info()
    assert info["baked_source_sha"] == "deadbeefcafe"
    assert info["running_source_sha"] != "deadbeefcafe"
    assert info["source_matches_image"] is False

    banner = bi.format_build_banner()
    assert "SOURCE DOES NOT MATCH IMAGE" in banner
    assert "abc1234" in banner


def test_matching_source_reports_clean(monkeypatch):
    monkeypatch.setenv("AGENTRAG_GIT_SHA", "abc1234")
    monkeypatch.setenv("AGENTRAG_SOURCE_SHA", bi.source_sha())
    info = bi.build_info()
    assert info["source_matches_image"] is True
    assert "source matches image" in bi.format_build_banner()


def test_unstamped_image_is_unverified_not_a_false_all_clear(monkeypatch):
    for var in ("AGENTRAG_GIT_SHA", "AGENTRAG_BUILD_ID", "AGENTRAG_SOURCE_SHA"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(bi, "BAKED_SHA_FILE", bi.Path("/nonexistent/.build-source-sha"))
    info = bi.build_info()
    # None, not False: "we could not check" must not read as "checked and fine".
    assert info["source_matches_image"] is None
    assert "provenance unverified" in bi.format_build_banner()


def test_baked_sha_falls_back_to_the_file_the_dockerfile_writes(monkeypatch, tmp_path):
    """The bind mount covers /app/src, not /app, so this file still describes
    what the IMAGE shipped even while a mount changes what runs."""
    monkeypatch.delenv("AGENTRAG_SOURCE_SHA", raising=False)
    baked = tmp_path / ".build-source-sha"
    baked.write_text("f00dfeed1234\n", encoding="utf-8")
    monkeypatch.setattr(bi, "BAKED_SHA_FILE", baked)
    assert bi.build_info()["baked_source_sha"] == "f00dfeed1234"


def test_a_mismatch_logs_at_warning_level(monkeypatch, caplog):
    monkeypatch.setenv("AGENTRAG_SOURCE_SHA", "not-the-running-source")
    with caplog.at_level(logging.INFO):
        bi.log_build_banner("api")
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert "[api]" in caplog.text


def test_provenance_never_raises_even_when_hashing_fails(monkeypatch):
    monkeypatch.setattr(bi, "source_sha", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    info = bi.build_info()
    assert info["running_source_sha"] == bi.UNKNOWN
    assert info["source_matches_image"] is None
    bi.format_build_banner()  # must not raise
