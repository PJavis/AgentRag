"""The answer cache must never outlive the corpus it answered from.

2026-09-06 is the cautionary case: a re-ingest replaced every segment. A cached
answer keyed without a corpus version would have gone on citing segment ids that
no longer existed.
"""
from src.agentrag.common import corpus_version


class _FakeRedis:
    def __init__(self, value=None, broken=False):
        self.value = value
        self.broken = broken

    def get(self, key):
        if self.broken:
            raise RuntimeError("valkey down")
        return self.value

    def set(self, key, value):
        if self.broken:
            raise RuntimeError("valkey down")
        self.value = value


def test_version_is_none_when_never_set(monkeypatch):
    monkeypatch.setattr(corpus_version, "_client", lambda: _FakeRedis(None))
    assert corpus_version.get_corpus_version() is None


def test_bump_sets_and_returns_a_new_version(monkeypatch):
    fake = _FakeRedis(None)
    monkeypatch.setattr(corpus_version, "_client", lambda: fake)
    first = corpus_version.bump_corpus_version()
    assert first and fake.value == first
    second = corpus_version.bump_corpus_version()
    assert second != first, "each ingest must produce a distinct version"


def test_unreachable_valkey_reports_unknown_rather_than_raising(monkeypatch):
    monkeypatch.setattr(corpus_version, "_client", lambda: _FakeRedis(broken=True))
    assert corpus_version.get_corpus_version() is None
    assert corpus_version.bump_corpus_version() is None


def test_no_client_at_all_is_unknown(monkeypatch):
    monkeypatch.setattr(corpus_version, "_client", lambda: None)
    assert corpus_version.get_corpus_version() is None
