"""A decide call whose tool can never run is not made.

With AGENT_MAX_STEPS=3 the graph ran bootstrap → decide → tool → decide →
assemble: the second decide's tool choice was always discarded (~2 s of the
local 3B per question, measured 2026-11-28). A duplicate tool call also looped
back to decide for one more discarded call.
"""
from __future__ import annotations

import pytest

from src.agentrag.agent import graph_service as gs


def _walk(decisions: list[dict]) -> tuple[int, int]:
    """Drive the routers the way the compiled graph does. Returns
    (decide calls, tool calls)."""
    state: dict = {"step_count": 0}
    decides = tools = 0
    node = gs._route_after_bootstrap(state)
    while node != "assemble":
        if node == "decide":
            state["decide_decision"] = decisions[decides] if decides < len(decisions) else {"done": False}
            decides += 1
            state["step_count"] += 1
            node = gs._route_decide(state)
        elif node == "tool_exec":
            tools += 1
            node = gs._route_after_tool(state)
        else:  # pragma: no cover
            raise AssertionError(node)
    return decides, tools


@pytest.mark.parametrize(
    "max_steps, expected",
    [
        (1, (0, 0)),  # limit 1: any decide result was discarded
        (2, (0, 0)),
        (3, (1, 1)),  # was (2, 1): second decide discarded
        (4, (2, 2)),  # was (3, 2)
    ],
)
def test_every_decide_call_can_act(monkeypatch, max_steps, expected):
    monkeypatch.setattr(gs.settings, "AGENT_MAX_STEPS", max_steps)
    assert _walk([]) == expected


def test_done_still_stops_early(monkeypatch):
    monkeypatch.setattr(gs.settings, "AGENT_MAX_STEPS", 4)
    assert _walk([{"done": True}]) == (1, 0)


def test_duplicate_tool_call_still_gets_another_decide_when_steps_remain(monkeypatch):
    # tool_exec marks a duplicate as done; with steps left the next decide can
    # pick a refined query whose tool then runs (review of this change).
    monkeypatch.setattr(gs.settings, "AGENT_MAX_STEPS", 5)
    assert gs._route_after_tool({"step_count": 1, "decide_decision": {"done": True}}) == "decide"
    # ...and at the limit it goes to assemble like any other step.
    assert gs._route_after_tool({"step_count": 3, "decide_decision": {"done": True}}) == "assemble"


def test_graph_is_wired_through_the_routers():
    edges = gs._GRAPH.get_graph().edges
    plain = {(e.source, e.target) for e in edges if not e.conditional}
    assert ("bootstrap", "decide") not in plain
    assert ("tool_exec", "decide") not in plain
    conditional = {(e.source, e.target) for e in edges if e.conditional}
    assert {("bootstrap", "decide"), ("bootstrap", "assemble"),
            ("tool_exec", "decide"), ("tool_exec", "assemble")} <= conditional


def test_compiled_graph_makes_one_decide_per_runnable_tool(monkeypatch):
    """Run a real compiled graph (stub nodes, real routers and edges) and
    count the decide / tool_exec executions."""
    import asyncio

    calls = {"decide": 0, "tool_exec": 0}

    async def passthrough(state):
        return {}

    async def chitchat_check(state):
        return {"is_chitchat": False}

    async def bootstrap(state):
        return {"tool_trace": [], "step_count": 0}

    async def decide(state):
        calls["decide"] += 1
        return {"decide_decision": {"done": False}, "step_count": state.get("step_count", 0) + 1}

    async def tool_exec(state):
        calls["tool_exec"] += 1
        return {}

    for name, fn in {
        "validate": passthrough, "memory": passthrough, "chitchat_check": chitchat_check,
        "semantic_plan": passthrough, "bootstrap": bootstrap, "decide": decide,
        "tool_exec": tool_exec, "assemble": passthrough, "answer_node": passthrough,
        "critique": passthrough, "ground": passthrough,
    }.items():
        monkeypatch.setattr(gs, name, fn)
    monkeypatch.setattr(gs.settings, "CRAG_ENABLED", False)

    for max_steps, expected in [(2, (0, 0)), (3, (1, 1)), (4, (2, 2))]:
        monkeypatch.setattr(gs.settings, "AGENT_MAX_STEPS", max_steps)
        calls.update(decide=0, tool_exec=0)
        asyncio.run(gs._build_graph().ainvoke({"question": "q"}))
        assert (calls["decide"], calls["tool_exec"]) == expected, max_steps


def test_default_runs_bootstrap_only(monkeypatch):
    # 2026-11-28 A/B: the decide round added no gold passage on c3 and cost
    # ~3 s/answer, so the default is bootstrap-only (no decide LLM call).
    from src.agentrag.config import Settings

    default = Settings.model_fields["AGENT_MAX_STEPS"].default
    assert default == 2
    monkeypatch.setattr(gs.settings, "AGENT_MAX_STEPS", default)
    assert _walk([]) == (0, 0)
