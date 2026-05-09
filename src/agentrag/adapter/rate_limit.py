"""Per-user rate limiter backed by Redis (sliding-window counter).

We classify endpoints into buckets:
  - "upload"  → POST /api/sources, /api/sources/json   (RATE_LIMIT_UPLOAD_PER_MIN)
  - "default" → everything else                        (RATE_LIMIT_PER_MIN_DEFAULT)

Rate limit keys are scoped to the authenticated user (or remote IP for
unauthenticated public routes). On any Redis error we fail open — better
to over-serve than to block legitimate traffic.
"""
from __future__ import annotations

import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.agentrag.config import settings

_UPLOAD_PATHS = (
    "/api/sources",
    "/api/sources/json",
)


def _bucket_for(request: Request) -> str:
    if request.method == "POST" and any(
        request.url.path.endswith(p) for p in _UPLOAD_PATHS
    ):
        return "upload"
    return "default"


def _limit_for(bucket: str) -> int:
    if bucket == "upload":
        return settings.RATE_LIMIT_UPLOAD_PER_MIN
    return settings.RATE_LIMIT_PER_MIN_DEFAULT


def _actor_key(request: Request) -> str:
    identity = getattr(request.state, "auth_identity", None)
    if identity and identity.user_id:
        return f"user:{identity.user_id}"
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return f"ip:{fwd.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._redis = None
        self._redis_init_attempted = False

    async def _get_redis(self):
        if self._redis is not None:
            return self._redis
        if self._redis_init_attempted:
            return None
        self._redis_init_attempted = True
        if not settings.REDIS_URL:
            return None
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
            return self._redis
        except Exception:
            return None

    async def dispatch(self, request: Request, call_next):
        if not settings.RATE_LIMIT_ENABLED or request.method == "OPTIONS":
            return await call_next(request)
        # Don't rate-limit health / config probes
        path = request.url.path
        if any(path.endswith(p) for p in ("/health", "/api/config", "/api/auth/status")):
            return await call_next(request)

        redis = await self._get_redis()
        if redis is None:
            return await call_next(request)

        bucket = _bucket_for(request)
        limit = _limit_for(bucket)
        actor = _actor_key(request)
        window = int(time.time()) // 60  # 1-minute fixed window
        key = f"rl:{bucket}:{actor}:{window}"

        try:
            count = await redis.incr(key)
            if count == 1:
                await redis.expire(key, 65)
        except Exception:
            return await call_next(request)

        if count > limit:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "bucket": bucket,
                    "limit": limit,
                    "retry_after_seconds": 60 - (int(time.time()) % 60),
                },
                headers={"Retry-After": str(60 - (int(time.time()) % 60))},
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(limit - count, 0))
        return response
