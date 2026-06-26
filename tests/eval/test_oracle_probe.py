import pytest

from scripts.eval.oracle_probe import (
    generate_oracle_answer, summarize_probe, pearson, ProbeRow,
)


class FakeTextGateway:
    def __init__(self, text):
        self._text = text
        self.calls = []

    async def text_response(self, system_prompt, user_prompt, task="general"):
        self.calls.append({"system": system_prompt, "user": user_prompt})
        return self._text


@pytest.mark.asyncio
async def test_generate_oracle_answer_uses_gold_context():
    gw = FakeTextGateway("oracle answer")
    out = await generate_oracle_answer("what is X?", "X is defined here.", gw)
    assert out == "oracle answer"
    assert "X is defined here." in gw.calls[0]["user"]


def test_pearson_perfect_correlation():
    assert round(pearson([1, 2, 3], [2, 4, 6]), 6) == 1.0


def test_pearson_degenerate_returns_zero():
    # zero variance → defined as 0.0, not a crash
    assert pearson([1, 1, 1], [2, 3, 4]) == 0.0


def test_summarize_probe_reports_gap_and_noise():
    rows = [
        ProbeRow("q1", system_mean=0.70, oracle_mean=0.74, judge2_mean=0.69),
        ProbeRow("q2", system_mean=0.80, oracle_mean=0.82, judge2_mean=0.81),
    ]
    s = summarize_probe(rows)
    assert s["n"] == 2
    assert s["system_avg"] == 0.75
    assert s["oracle_avg"] == 0.78
    assert abs(s["oracle_minus_system"] - 0.03) < 1e-9
    assert abs(s["judge_noise_pearson"] - 1.0) < 1e-9  # system vs judge2 perfectly correlated here
