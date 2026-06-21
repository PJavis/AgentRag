import os
from scripts.eval.run_ablation import ABLATIONS, build_env, build_cmd


def test_ablations_matrix_has_expected_configs():
    assert set(ABLATIONS) >= {"baseline", "cr", "cr_raptor", "cr_raptor_crag", "full"}
    # baseline = no flags on
    assert ABLATIONS["baseline"] == {}
    # full turns on every workstream
    full = ABLATIONS["full"]
    for k in ("CONTEXTUAL_RETRIEVAL_ENABLED", "RAPTOR_ENABLED", "CRAG_ENABLED",
              "SEMANTIC_CACHE_ENABLED", "AGENT_MULTIHOP_ENABLED"):
        assert full.get(k) == "true"


def test_build_env_merges_overrides_onto_base():
    base = {"PATH": "/usr/bin", "FOO": "bar"}
    env = build_env(base, {"CONTEXTUAL_RETRIEVAL_ENABLED": "true"})
    assert env["PATH"] == "/usr/bin"            # base preserved
    assert env["FOO"] == "bar"
    assert env["CONTEXTUAL_RETRIEVAL_ENABLED"] == "true"
    # ingest must be synchronous so the index is fully built before scoring
    assert env["STRUCTMEM_INGEST_MODE"] == "sync"
    # must not mutate the input base dict
    assert "CONTEXTUAL_RETRIEVAL_ENABLED" not in base


def test_build_cmd_targets_run_benchmark_with_args():
    cmd = build_cmd(suite="both", n=20, judge_provider="deepseek", out_path="data/eval/_abl_full.json")
    joined = " ".join(cmd)
    assert "run_benchmark.py" in joined
    assert "--suite" in cmd and "both" in cmd
    assert "--n" in cmd and "20" in cmd
    assert "--judge-provider" in cmd and "deepseek" in cmd
    assert "--out" in cmd and "data/eval/_abl_full.json" in cmd
