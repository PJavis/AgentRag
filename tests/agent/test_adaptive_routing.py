from src.agentrag.agent import graph_service as gs
from src.agentrag.structured.query_classifier import ClassifierOutput


def _co(intent="semantic", complexity="simple", single_domain=True, conf=0.95):
    return ClassifierOutput(intent=intent, query_type=None, confidence=conf,
                            reasoning="", method="rule", complexity=complexity,
                            single_domain=single_domain)


def test_route_takes_fast_path_for_simple_single_domain(monkeypatch):
    monkeypatch.setattr(gs.settings, "ADAPTIVE_ROUTING_ENABLED", True)
    monkeypatch.setattr(gs.settings, "ADAPTIVE_FASTPATH_MIN_CONFIDENCE", 0.85)
    state = {"intent": "semantic", "classifier_output": _co()}
    assert gs._route_intent(state) == "fast_answer"


def test_route_takes_full_path_for_complex(monkeypatch):
    monkeypatch.setattr(gs.settings, "ADAPTIVE_ROUTING_ENABLED", True)
    state = {"intent": "semantic", "classifier_output": _co(complexity="complex")}
    assert gs._route_intent(state) == "semantic_plan"


def test_route_takes_full_path_when_flag_off(monkeypatch):
    monkeypatch.setattr(gs.settings, "ADAPTIVE_ROUTING_ENABLED", False)
    state = {"intent": "semantic", "classifier_output": _co()}
    assert gs._route_intent(state) == "semantic_plan"


def test_structured_intent_still_routes_structured(monkeypatch):
    monkeypatch.setattr(gs.settings, "ADAPTIVE_ROUTING_ENABLED", True)
    state = {"intent": "structured", "classifier_output": _co(intent="structured", complexity="complex")}
    assert gs._route_intent(state) == "structured_run"
