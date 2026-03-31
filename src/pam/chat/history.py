from __future__ import annotations

import json
import uuid
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import select

from src.pam.config import settings
from src.pam.database import AsyncSessionLocal
from src.pam.database.models import ChatMessage, Conversation


class ConversationStore:
    def __init__(self):
        self._redis: Redis | None = None
        if settings.REDIS_URL:
            self._redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)

    @staticmethod
    def _messages_cache_key(conversation_id: str) -> str:
        return f"pam:conversation:{conversation_id}:messages:v1"

    async def _read_messages_cache(self, conversation_id: str) -> list[dict[str, Any]] | None:
        if self._redis is None:
            return None
        try:
            raw = await self._redis.get(self._messages_cache_key(conversation_id))
        except RedisError:
            self._redis = None
            return None
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except Exception:
            return None
        if not isinstance(payload, list):
            return None
        return payload

    async def _write_messages_cache(self, conversation_id: str, messages: list[dict[str, Any]]) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.set(
                self._messages_cache_key(conversation_id),
                json.dumps(messages, ensure_ascii=False),
                ex=settings.CHAT_REDIS_TTL_SECONDS,
            )
        except RedisError:
            self._redis = None

    async def _delete_messages_cache(self, conversation_id: str) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.delete(self._messages_cache_key(conversation_id))
        except RedisError:
            self._redis = None

    async def create_conversation(
        self,
        title: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with AsyncSessionLocal() as session:
            conversation = Conversation(
                title=title,
                extra_metadata=extra_metadata,
            )
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)
            return {
                "conversation_id": str(conversation.id),
                "title": conversation.title,
                "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
            }

    async def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        try:
            conversation_uuid = uuid.UUID(conversation_id)
        except ValueError:
            return None
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Conversation).where(Conversation.id == conversation_uuid)
            )
            conversation = result.scalar_one_or_none()
            if conversation is None:
                return None
            return {
                "conversation_id": str(conversation.id),
                "title": conversation.title,
                "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
            }

    async def get_or_create_conversation(
        self,
        conversation_id: str | None,
        title: str | None = None,
    ) -> dict[str, Any]:
        if not conversation_id:
            return await self.create_conversation(title=title)
        conversation = await self.get_conversation(conversation_id)
        if conversation is None:
            return await self.create_conversation(title=title)
        return conversation

    async def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        tool_trace: list[dict[str, Any]] | None = None,
        timings_ms: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with AsyncSessionLocal() as session:
            message = ChatMessage(
                conversation_id=uuid.UUID(conversation_id),
                role=role,
                content=content,
                citations=citations,
                tool_trace=tool_trace,
                timings_ms=timings_ms,
                extra_metadata=extra_metadata,
            )
            session.add(message)
            await session.commit()
            await session.refresh(message)

        await self._delete_messages_cache(conversation_id)
        return {
            "message_id": str(message.id),
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "citations": citations or [],
            "created_at": message.created_at.isoformat() if message.created_at else None,
        }

    async def list_messages(
        self,
        conversation_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        try:
            conversation_uuid = uuid.UUID(conversation_id)
        except ValueError:
            return []

        cached = await self._read_messages_cache(conversation_id)
        if cached is not None:
            return cached[-limit:]

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ChatMessage)
                .where(ChatMessage.conversation_id == conversation_uuid)
                .order_by(ChatMessage.created_at.asc())
            )
            messages = result.scalars().all()

        payload = [
            {
                "message_id": str(item.id),
                "conversation_id": conversation_id,
                "role": item.role,
                "content": item.content,
                "citations": item.citations or [],
                "tool_trace": item.tool_trace or [],
                "timings_ms": item.timings_ms or {},
                "created_at": item.created_at.isoformat() if item.created_at else None,
            }
            for item in messages
        ]
        await self._write_messages_cache(conversation_id, payload)
        return payload[-limit:]

    async def list_conversations(self, limit: int = 20) -> list[dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Conversation)
                .order_by(Conversation.created_at.desc())
                .limit(limit)
            )
            rows = result.scalars().all()
        return [
            {
                "conversation_id": str(item.id),
                "title": item.title,
                "created_at": item.created_at.isoformat() if item.created_at else None,
            }
            for item in rows
        ]
