from src.agentrag.agent import graph_service as gs


def test_route_critique_to_corrective_when_ungrounded(monkeypatch):
    monkeypatch.setattr(gs.settings, "CRAG_ENABLED", True)
    monkeypatch.setattr(gs.settings, "AGENT_CRITIQUE_MAX_RETRIES", 1)
    state = {"critique_decision": {"grounded": False, "reason": "no_citations"},
             "critique_retries": 0}
    assert gs._route_critique(state) == "corrective_retrieve"


def test_route_critique_to_end_when_grounded(monkeypatch):
    monkeypatch.setattr(gs.settings, "CRAG_ENABLED", True)
    state = {"critique_decision": {"grounded": True}, "critique_retries": 0}
    assert gs._route_critique(state) == "ground"


def test_route_critique_stops_after_max_retries(monkeypatch):
    monkeypatch.setattr(gs.settings, "CRAG_ENABLED", True)
    monkeypatch.setattr(gs.settings, "AGENT_CRITIQUE_MAX_RETRIES", 1)
    state = {"critique_decision": {"grounded": False}, "critique_retries": 1}
    assert gs._route_critique(state) == "ground"  # give up, return best effort


def test_route_critique_disabled_goes_straight_to_ground(monkeypatch):
    monkeypatch.setattr(gs.settings, "CRAG_ENABLED", False)
    state = {"critique_decision": {"grounded": False}, "critique_retries": 0}
    assert gs._route_critique(state) == "ground"
