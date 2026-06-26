import json

from src.agentrag.eval.benchmark_datasets import load_local_jsonl, EvalExample


def test_load_local_jsonl_roundtrip(tmp_path):
    p = tmp_path / "eval.jsonl"
    rows = [
        {"id": "prod_corpus-0", "question": "Q0?", "reference_answer": "A0",
         "gold_contexts": ["chunk 0"], "lang": "vi", "source": "prod_corpus"},
        {"id": "prod_corpus-1", "question": "Q1?", "reference_answer": "A1",
         "gold_contexts": ["chunk 1"], "lang": "en", "source": "prod_corpus"},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    out = load_local_jsonl(str(p))
    assert len(out) == 2
    assert isinstance(out[0], EvalExample)
    assert out[0].id == "prod_corpus-0"
    assert out[0].question == "Q0?"
    assert out[0].reference_answer == "A0"
    assert out[0].gold_contexts == ["chunk 0"]
    assert out[0].lang == "vi"


def test_load_local_jsonl_skips_blank_and_invalid(tmp_path):
    p = tmp_path / "eval.jsonl"
    p.write_text(
        '{"id":"a","question":"Q?","reference_answer":"A","gold_contexts":["c"]}\n'
        "\n"  # blank line
        '{"question":"","gold_contexts":[]}\n',  # no question/contexts → skipped
        encoding="utf-8",
    )
    out = load_local_jsonl(str(p))
    assert len(out) == 1
    assert out[0].id == "a"


def test_load_local_jsonl_n_cap(tmp_path):
    p = tmp_path / "eval.jsonl"
    p.write_text("\n".join(
        json.dumps({"id": f"r{i}", "question": f"Q{i}?", "reference_answer": "A",
                    "gold_contexts": ["c"]}) for i in range(5)
    ), encoding="utf-8")
    assert len(load_local_jsonl(str(p), n=2)) == 2
