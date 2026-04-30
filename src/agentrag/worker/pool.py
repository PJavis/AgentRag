"""ARQ Redis pool singleton — shared across the FastAPI app process."""
from __future__ import annotations

from arq import ArqRedis, create_pool
from arq.connections import RedisSettings

_pool: ArqRedis | None = None


async def init_pool(redis_url: str) -> ArqRedis:
    global _pool
    _pool = await create_pool(RedisSettings.from_dsn(redis_url))
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None


def get_pool() -> ArqRedis:
    if _pool is None:
        raise RuntimeError("ARQ pool not initialized — call init_pool() during app startup")
    return _pool
