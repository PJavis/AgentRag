"""B-acceptance — adapter models accept the new fields."""
from src.agentrag.adapter.models import ChatMessage


def test_chat_message_accepts_follow_ups():
    m = ChatMessage(
        type="ai", role="assistant", content="hi",
        follow_ups=["A?", "B?"],
    )
    d = m.model_dump()
    assert d["follow_ups"] == ["A?", "B?"]


def test_chat_message_follow_ups_optional():
    m = ChatMessage(type="ai", role="assistant", content="hi")
    assert m.follow_ups is None


def test_chat_message_accepts_extended_citations():
    cite = {
        "document_title": "Lec 10.pdf",
        "section_path": "Ch. 3",
        "page": 47,
        "page_label": "p.47",
        "excerpt": "Van hai lá…",
        "content_hash": "abc",
        "mime": "application/pdf",
        "segment_type": "text",
    }
    m = ChatMessage(
        type="ai", role="assistant", content="ans",
        citations=[cite],
    )
    assert m.citations is not None
    assert m.citations[0]["page_label"] == "p.47"
