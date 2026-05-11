"""Authentication middleware: JWT user tokens + legacy shared password.

The open-notebook frontend sends `Authorization: Bearer <token>`. We accept
either a JWT (issued by /api/auth/login or /api/auth/google/callback) or
the legacy `OPEN_NOTEBOOK_PASSWORD` for backwards compatibility.
"""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.agentrag.adapter.auth_service import (
    AuthIdentity,
    LEGACY_PASSWORD_USER_ID,
    decode_token,
)
from src.agentrag.config import settings

_PUBLIC_PREFIXES = (
    "/api/config",
    "/api/auth/status",
    "/api/auth/login",
    "/api/auth/signup",
    "/api/auth/google/start",
    "/api/auth/google/callback",
    "/api/health",
    "/api/images",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/admin",
)


def _is_public(path: str) -> bool:
    if path == "/":
        return True
    for p in _PUBLIC_PREFIXES:
        if path == p or path.startswith(p + "/") or path.endswith(p) or (p + "/") in path:
            return True
    return False


def _identity_from_token(token: str) -> AuthIdentity | None:
    """Resolve a bearer token to an identity (JWT first, then legacy password)."""
    if not token:
        return None
    payload = decode_token(token)
    if payload and payload.get("sub"):
        return AuthIdentity(
            user_id=str(payload["sub"]),
            email=payload.get("email"),
            name=payload.get("name"),
            is_admin=bool(payload.get("admin")),
        )
    if settings.OPEN_NOTEBOOK_PASSWORD and token == settings.OPEN_NOTEBOOK_PASSWORD:
        return AuthIdentity(
            user_id=LEGACY_PASSWORD_USER_ID,
            email=None,
            name="Legacy",
            is_admin=True,
            is_legacy=True,
        )
    return None


class OpenNotebookAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Always allow CORS preflight
        if request.method == "OPTIONS":
            return await call_next(request)

        # Auth disabled entirely → set anonymous identity and continue
        if not settings.AUTH_ENABLED:
            request.state.auth_identity = AuthIdentity(
                user_id="anonymous",
                email=None,
                name="Anonymous",
                is_admin=False,
                is_legacy=True,
            )
            return await call_next(request)

        # Public endpoints: still try to attach identity if a token is present
        if _is_public(request.url.path):
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                identity = _identity_from_token(auth[7:])
                if identity:
                    request.state.auth_identity = identity
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        identity = _identity_from_token(auth[7:])
        if identity is None:
            return JSONResponse(status_code=401, content={"detail": "Invalid token"})

        request.state.auth_identity = identity
        return await call_next(request)


def is_admin(request: Request) -> bool:
    """True if request carries the legacy admin token OR the user is an admin."""
    if settings.ADAPTER_ADMIN_TOKEN:
        admin_header = request.headers.get("X-Admin-Token", "")
        if admin_header == settings.ADAPTER_ADMIN_TOKEN:
            return True
    identity = getattr(request.state, "auth_identity", None)
    return bool(identity and identity.is_admin)


def get_identity(request: Request) -> AuthIdentity | None:
    """Return the authenticated identity attached by the middleware."""
    return getattr(request.state, "auth_identity", None)
