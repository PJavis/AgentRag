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
    ensure_user_row,
)
from src.agentrag.config import settings

_PUBLIC_PREFIXES = (
    "/api/config",
    "/api/auth/status",
    "/api/auth/login",
    "/api/auth/signup",
    "/api/auth/google/start",
    "/api/auth/google/callback",
    "/api/images",
    "/api/ontology",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/admin",
    "/images",
)

# Liveness probes: public as exact paths only — /health/providers (root) exposes
# provider URLs, models and config errors and stays private.
_PUBLIC_EXACT = ("/health", "/api/health")

# The adapter is mounted at /on, and a mounted app's middleware sees the full path.
_ADAPTER_MOUNT = "/on"


def _strip_adapter_mount(path: str) -> str:
    if path == _ADAPTER_MOUNT:
        return "/"
    if path.startswith(_ADAPTER_MOUNT + "/"):
        return path[len(_ADAPTER_MOUNT):]
    return path


def _is_public(path: str) -> bool:
    # Exact prefix match only. The old endswith/substring match made any path
    # ending in /health or /docs, or containing /admin/, public.
    path = _strip_adapter_mount(path)
    if path == "/" or path in _PUBLIC_EXACT:
        return True
    return any(path == p or path.startswith(p + "/") for p in _PUBLIC_PREFIXES)


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

        # Self-heal: JWT survived a `make reset-data` that wiped the users
        # table. Recreate the user row idempotently so FK constraints
        # (documents.user_id, conversations.user_id) keep working.
        if not identity.is_legacy and identity.user_id != "anonymous":
            await ensure_user_row(
                user_id=identity.user_id,
                email=identity.email,
                name=identity.name,
                is_admin=identity.is_admin,
            )

        request.state.auth_identity = identity
        return await call_next(request)


class RootAuthMiddleware(OpenNotebookAuthMiddleware):
    """Same auth for the root app's own routes (main.py, /mcp). Requests under the
    adapter mount are skipped here because the adapter authenticates them itself."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path == _ADAPTER_MOUNT or path.startswith(_ADAPTER_MOUNT + "/"):
            return await call_next(request)
        return await super().dispatch(request, call_next)


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
