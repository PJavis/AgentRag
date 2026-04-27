# src/pam/database/__init__.py
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession  # <-- thêm AsyncSession vào đây
from src.agentrag.config import settings

# Engine async
engine = create_async_engine(settings.DATABASE_URL, echo=False)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

# Export Base và models
from .base import Base
from .models import Project, Document, Segment, SyncLog, Conversation, ChatMessage
