"""Chat memory processing — called by the ARQ worker (chat_memory function)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMemoryJob:
    conversation_id: str
    user_message: str
    assistant_message: str
    turn_id: str
    turn_timestamp: str
