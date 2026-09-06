"""Exact-match answer cache. Deliberately NOT semantic — see the module docstring.

The dangerous case this design refuses: at a 0.97 cosine threshold
"liều dùng cho người lớn" and "liều dùng cho trẻ em" are neighbours, and serving
one for the other is a clinical failure, not a cache miss.
"""
import json

import pytest

from src.agentrag.config import settings
from src.agentrag.services.answer_cache import AnswerCache


class _FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value


@pytest.fixture
def cache(monkeypatch):
    monkeypatch.setattr(settings, "ANSWER_CACHE_ENABLED", True)
    c = AnswerCache()
    fake = _FakeRedis()
    monkeypatch.setattr(c, "_client", lambda: fake)
    return c


def test_flag_defaults_off():
    from src.agentrag.config import Settings

    assert Settings.model_fields["ANSWER_CACHE_ENABLED"].default is False


def test_the_same_question_produces_the_same_key(cache):
    a = cache.key(question="Liều paracetamol?", corpus_version="v1",
                  document_title=None, domain_filter=None, verbosity=None, model="m")
    b = cache.key(question="  liều   PARACETAMOL? ", corpus_version="v1",
                  document_title=None, domain_filter=None, verbosity=None, model="m")
    assert a == b, "whitespace and case must normalise"


def test_a_different_question_produces_a_different_key(cache):
    a = cache.key(question="liều dùng cho người lớn", corpus_version="v1",
                  document_title=None, domain_filter=None, verbosity=None, model="m")
    b = cache.key(question="liều dùng cho trẻ em", corpus_version="v1",
                  document_title=None, domain_filter=None, verbosity=None, model="m")
    assert a != b


@pytest.mark.parametrize("field,value", [
    ("corpus_version", "v2"),
    ("document_title", "doc.pdf"),
    ("verbosity", "detailed"),
    ("model", "other-model"),
])
def test_every_scoping_input_changes_the_key(cache, field, value):
    base = dict(question="q", corpus_version="v1", document_title=None,
                domain_filter=None, verbosity=None, model="m")
    changed = {**base, field: value}
    assert cache.key(**base) != cache.key(**changed)


def test_domain_filter_changes_the_key_regardless_of_dict_ordering(cache):
    base = dict(question="q", corpus_version="v1", document_title=None,
                verbosity=None, model="m")
    one = cache.key(**base, domain_filter={"system": "tim_mach", "specialties": ["a"]})
    two = cache.key(**base, domain_filter={"specialties": ["a"], "system": "tim_mach"})
    three = cache.key(**base, domain_filter={"system": "noi"})
    assert one == two, "key order must not matter"
    assert one != three


def test_no_corpus_version_means_no_key_and_therefore_no_caching(cache):
    assert cache.key(question="q", corpus_version=None, document_title=None,
                     domain_filter=None, verbosity=None, model="m") is None


def test_put_then_get_round_trips(cache):
    key = cache.key(question="q", corpus_version="v1", document_title=None,
                    domain_filter=None, verbosity=None, model="m")
    cache.put(key, {"answer": "42", "citations": []})
    assert cache.get(key)["answer"] == "42"


def test_get_of_a_missing_key_is_none(cache):
    assert cache.get("nope") is None


def test_a_none_key_never_touches_the_store(cache):
    assert cache.get(None) is None
    cache.put(None, {"answer": "x"})  # must not raise


def test_disabled_cache_never_returns_anything(monkeypatch):
    monkeypatch.setattr(settings, "ANSWER_CACHE_ENABLED", False)
    c = AnswerCache()
    fake = _FakeRedis()
    fake.store["k"] = json.dumps({"answer": "stale"})
    monkeypatch.setattr(c, "_client", lambda: fake)
    assert c.get("k") is None


def test_corrupt_cached_json_is_ignored_not_raised(cache):
    fake = _FakeRedis()
    fake.store["k"] = "{not json"
    cache._client = lambda: fake
    assert cache.get("k") is None


def test_a_conversational_turn_is_never_cached():
    """Follow-ups depend on history — graph_service rewrites the question from
    it — so a history-blind key would serve the wrong answer."""
    from src.agentrag.services.answer_cache import cacheable_turn

    assert cacheable_turn(chat_history=None)
    assert cacheable_turn(chat_history=[])
    assert not cacheable_turn(chat_history=[{"role": "user", "content": "trước đó"}])


def test_a_timeout_result_is_never_cached():
    """graph_service.chat returns a load message on graceful timeout. Caching it
    would serve "Hệ thống đang bận" as the answer for a full TTL."""
    from src.agentrag.services.answer_cache import cacheable_result

    assert not cacheable_result({
        "answer": "Hệ thống đang bận, vui lòng thử lại sau giây lát.",
        "citations": [], "timed_out": True,
    })
    assert cacheable_result({"answer": "Liều là 500 mg.", "citations": []})


def test_an_empty_answer_is_never_cached():
    from src.agentrag.services.answer_cache import cacheable_result

    assert not cacheable_result({"answer": "   "})
    assert not cacheable_result({})
    assert not cacheable_result(None)
