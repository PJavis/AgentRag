"""A 200 from /chat is not a success.

The first version of the baseline runner checked only for transport errors, so
nineteen replies of `{"error": "question is required"}` were reported as
"19 answered, 0 failed" — a false green that would have become a baseline.
"""
import pytest

from scripts.eval.table_arm_baseline_run import ChatError, ask_text, validate_reply


def test_an_error_payload_with_status_200_is_a_failure():
    with pytest.raises(ChatError, match="question is required"):
        validate_reply({"error": "question is required"})


def test_an_empty_answer_is_a_failure():
    with pytest.raises(ChatError, match="empty answer"):
        validate_reply({"answer": "   "})
    with pytest.raises(ChatError, match="empty answer"):
        validate_reply({})


def test_a_non_object_reply_is_a_failure():
    with pytest.raises(ChatError):
        validate_reply(["not", "an", "object"])


def test_a_real_answer_passes():
    validate_reply({"answer": "Liều dùng là 500 mg.", "citations": []})


def test_question_text_names_the_row_and_the_column():
    text = ask_text("Paracetamol", "Liều")
    assert "Paracetamol" in text and "Liều" in text
