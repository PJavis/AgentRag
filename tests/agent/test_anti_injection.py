"""Prompt-injection defense: the answer system prompt must instruct the model to treat
retrieved document content as untrusted DATA, never as instructions."""
from src.agentrag.agent.service import ANTI_INJECTION_RULE, _answer_system_prompt


def _ctx():
    return [{"document_title": "Doc", "content": "x", "source": 1, "rerank_score": 0.9}]


def test_anti_injection_rule_nontrivial():
    low = ANTI_INJECTION_RULE.lower()
    assert "instruction" in low and ("untrusted" in low or "data" in low)


def test_answer_prompt_includes_anti_injection():
    prompt = _answer_system_prompt("Câu hỏi?", verbose=False, packed_context=_ctx())
    assert ANTI_INJECTION_RULE in prompt
