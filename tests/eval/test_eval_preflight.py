from scripts.eval.run_benchmark import _eval_preflight_problems


def _health(ok=True, es_reachable=True, emb_reachable=True, emb_token=True, emb_provider="ollama"):
    return {
        "ok": ok,
        "validation_error": None if ok else "boom",
        "infra": {"elasticsearch": {"reachable": es_reachable, "error": None if es_reachable else "refused"}},
        "providers": {
            "embedding": {
                "provider": emb_provider,
                "reachable": emb_reachable,
                "token_present": emb_token,
                "base_url": "http://x",
            }
        },
    }


def test_no_problems_when_healthy():
    assert _eval_preflight_problems(_health(), judge_key="k") == []


def test_flags_es_unreachable():
    p = _eval_preflight_problems(_health(es_reachable=False), judge_key="k")
    assert any("Elasticsearch" in x for x in p)


def test_flags_embedding_unreachable():
    p = _eval_preflight_problems(_health(emb_reachable=False), judge_key="k")
    assert any("embedding" in x.lower() for x in p)


def test_flags_missing_judge_key():
    p = _eval_preflight_problems(_health(), judge_key=None)
    assert any("judge" in x.lower() for x in p)


def test_flags_validation_error():
    p = _eval_preflight_problems(_health(ok=False), judge_key="k")
    assert any("validation" in x.lower() for x in p)


def test_cloud_embedding_reachable_none_is_ok():
    # cloud providers report reachable=None → must NOT be flagged
    h = _health(emb_provider="gemini")
    h["providers"]["embedding"]["reachable"] = None
    assert _eval_preflight_problems(h, judge_key="k") == []
