"""Lenient JSON extraction for noisy LLM output (gemini-2.5-flash-lite emits
valid JSON followed by trailing prose → json.loads 'Extra data')."""
from src.agentrag.agent.llm import _loads_lenient


def test_trailing_data_after_object():
    # The dominant flash-lite failure: a valid object then extra text.
    assert _loads_lenient('{"done": false, "reflection": "x"}\nMore notes here.') == {
        "done": False, "reflection": "x"
    }


def test_leading_prose_before_object():
    assert _loads_lenient('Here is the result: {"done": true} thanks') == {"done": True}


def test_code_fenced_object():
    assert _loads_lenient('```json\n{"x": 2}\n```') == {"x": 2}


def test_nested_object_with_trailing_junk():
    assert _loads_lenient('{"a": {"b": 1}, "c": [1,2]} trailing') == {"a": {"b": 1}, "c": [1, 2]}


def test_clean_object():
    assert _loads_lenient('{"a": 1, "b": 2}') == {"a": 1, "b": 2}


def test_no_json_returns_none():
    assert _loads_lenient("no json here at all") is None


def test_brace_inside_string_not_miscounted():
    assert _loads_lenient('{"note": "use { and } carefully"} tail') == {"note": "use { and } carefully"}
