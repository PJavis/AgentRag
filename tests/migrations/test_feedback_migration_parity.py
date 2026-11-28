from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

from src.agentrag.adapter.db import AdapterChatFeedback

_MIGRATION = Path("migrations/versions/2026062501_add_adapter_chat_feedback.py")


def test_migration_covers_every_orm_column():
    """Every AdapterChatFeedback ORM column must appear in the migration (drift guard)."""
    src = _MIGRATION.read_text(encoding="utf-8")
    for col in AdapterChatFeedback.__table__.columns:
        assert f'"{col.name}"' in src, f"migration missing column {col.name}"


def test_single_head_is_new_revision():
    """The revision graph stays linear: one head, and the feedback revision
    sits in it (2026112801 source_bytes_sha was added on top)."""
    script = ScriptDirectory.from_config(Config("alembic.ini"))
    assert script.get_heads() == ["2026112801"]  # get_heads() returns a list
    assert script.get_revision("2026112801").down_revision == "2026062501"
    rev = script.get_revision("2026062501")
    assert rev.down_revision == "2026060501"
