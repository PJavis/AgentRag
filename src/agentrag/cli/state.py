from __future__ import annotations

import json
from pathlib import Path

_STATE_DIR = Path.home() / ".agentrag"
_STATE_FILE = _STATE_DIR / "state.json"


def load_state() -> dict:
    if not _STATE_FILE.exists():
        return {}
    try:
        return json.loads(_STATE_FILE.read_text())
    except Exception:
        return {}


def save_state(state: dict) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, indent=2))


def get_active_conversation() -> tuple[str | None, str | None]:
    state = load_state()
    return state.get("active_conversation_id"), state.get("active_conversation_title")


def set_active_conversation(conversation_id: str, title: str | None) -> None:
    state = load_state()
    state["active_conversation_id"] = conversation_id
    state["active_conversation_title"] = title
    save_state(state)


def clear_active_conversation() -> None:
    state = load_state()
    state.pop("active_conversation_id", None)
    state.pop("active_conversation_title", None)
    save_state(state)
