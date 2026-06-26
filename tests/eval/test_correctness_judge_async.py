import pytest

from src.agentrag.eval.correctness_judge import (
    extract_nuggets, score_nuggets, score_rubric, score_correctness,
)


class FakeGateway:
    """Returns scripted json payloads in order; records calls."""
    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = []

    async def json_response(self, system_prompt, user_prompt, task="general"):
        self.calls.append({"task": task, "system": system_prompt, "user": user_prompt})
        return self._payloads.pop(0), 1.0


@pytest.mark.asyncio
async def test_extract_nuggets_calls_gateway_and_parses():
    gw = FakeGateway([{"nuggets": ["A", "B"]}])
    out = await extract_nuggets("gold text", gw)
    assert out == ["A", "B"]
    assert gw.calls[0]["task"] == "eval_judge"
    assert "gold text" in gw.calls[0]["user"]


@pytest.mark.asyncio
async def test_score_nuggets_aggregates_labels():
    gw = FakeGateway([{"labels": ["covered", "absent"]}])
    s = await score_nuggets("q", "ans", ["A", "B"], gw)
    assert s.n_total == 2
    assert s.recall == 0.5
    assert s.score == 0.5


@pytest.mark.asyncio
async def test_score_nuggets_empty_skips_gateway():
    gw = FakeGateway([])  # would IndexError if called
    s = await score_nuggets("q", "ans", [], gw)
    assert s.score == 0.0
    assert gw.calls == []


@pytest.mark.asyncio
async def test_score_rubric_clamps():
    gw = FakeGateway([{"score": 1.4}])
    r = await score_rubric("q", "ans", "gold", "ctx", gw)
    assert r == 1.0


@pytest.mark.asyncio
async def test_score_correctness_full_flow():
    # 1) extract nuggets, 2) label them, 3) rubric score
    gw = FakeGateway([
        {"nuggets": ["A", "B", "C", "D"]},
        {"labels": ["covered", "covered", "contradicted", "absent"]},
        {"score": 0.7},
    ])
    e = await score_correctness("q", "ans", "gold", "ctx", gw)
    assert e.nugget == 0.25     # recall 0.5 - penalty 0.25
    assert e.rubric == 0.7
    assert e.mean == 0.475
    assert e.low_confidence is True  # |0.25-0.7| = 0.45 > 0.2
    assert len(gw.calls) == 3
