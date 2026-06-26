# tests/eval/test_build_prod_evalset.py
import pytest

from scripts.eval.build_prod_evalset import (
    detect_lang, build_eval_row, synth_question, gen_gold_answer,
)


class FakeGateway:
    def __init__(self, json_payloads=None, text="gold answer here"):
        self._json = list(json_payloads or [])
        self._text = text
        self.calls = []

    async def json_response(self, system_prompt, user_prompt, task="general"):
        self.calls.append({"task": task, "user": user_prompt})
        return self._json.pop(0), 1.0

    async def text_response(self, system_prompt, user_prompt, task="general"):
        self.calls.append({"task": task, "user": user_prompt})
        return self._text


def test_detect_lang_vietnamese():
    assert detect_lang("Bệnh nhân được chẩn đoán xơ gan giai đoạn cuối") == "vi"


def test_detect_lang_english():
    assert detect_lang("The patient was diagnosed with end-stage cirrhosis") == "en"


def test_build_eval_row_shape():
    row = build_eval_row(3, "What is X?", "X is a thing.", "X is a thing defined in the source chunk.")
    assert row["id"] == "prod_corpus-3"
    assert row["question"] == "What is X?"
    assert row["reference_answer"] == "X is a thing."
    assert row["gold_contexts"] == ["X is a thing defined in the source chunk."]
    assert row["source"] == "prod_corpus"
    assert row["lang"] in ("vi", "en")


@pytest.mark.asyncio
async def test_synth_question_takes_first():
    gw = FakeGateway(json_payloads=[{"questions": ["Q1?", "Q2?"]}])
    q = await synth_question("a chunk of text long enough to matter", gw)
    assert q == "Q1?"
    assert gw.calls[0]["task"] == "schema_discovery"


@pytest.mark.asyncio
async def test_synth_question_none_when_empty():
    gw = FakeGateway(json_payloads=[{"questions": []}])
    assert await synth_question("chunk", gw) is None


@pytest.mark.asyncio
async def test_gen_gold_answer_uses_chunk_and_task():
    gw = FakeGateway(text="A grounded gold answer.")
    out = await gen_gold_answer("What is X?", "X is defined here in the chunk.", gw)
    assert out == "A grounded gold answer."
    assert gw.calls[0]["task"] == "gold_gen"
    assert "X is defined here in the chunk." in gw.calls[0]["user"]
