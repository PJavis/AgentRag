"""Auth endpoints: signup / login / me / Google OAuth.

Mounted at /on/api/auth — open-notebook frontend hits /api/auth/status to
discover whether auth is required, then GETs /api/notebooks with
`Authorization: Bearer <token>`. We issue JWTs as that bearer.
"""
from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, Field

from src.agentrag.adapter import auth_service as auth_svc
from src.agentrag.config import settings

router = APIRouter(prefix="/auth")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class SignupRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=6, max_length=128)
    name: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, Any]


def _user_to_dict(user) -> dict[str, Any]:
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.name,
        "avatar_url": user.avatar_url,
        "is_admin": user.is_admin,
        "google_linked": bool(user.google_id),
    }


@router.get("/status")
async def auth_status():
    """Tell the frontend whether auth is required + which providers are available."""
    return {
        "auth_enabled": bool(settings.AUTH_ENABLED),
        "allow_signup": bool(settings.AUTH_ALLOW_SIGNUP),
        "providers": {
            "password": True,
            "google": bool(settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET),
        },
        "message": "Authentication is required" if settings.AUTH_ENABLED else "Authentication is disabled",
    }


@router.post("/signup", response_model=TokenResponse)
async def signup(body: SignupRequest):
    if not settings.AUTH_ALLOW_SIGNUP:
        raise HTTPException(403, "Signup is disabled")
    if not _EMAIL_RE.match(body.email):
        raise HTTPException(400, "Invalid email")
    existing = await auth_svc.get_user_by_email(body.email)
    if existing is not None:
        raise HTTPException(409, "Account with this email already exists")
    user = await auth_svc.create_user(
        email=body.email, password=body.password, name=body.name
    )
    token = auth_svc.issue_token(user, ttl_seconds=settings.JWT_TTL_DAYS * 86400)
    return TokenResponse(access_token=token, user=_user_to_dict(user))


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    user = await auth_svc.authenticate(body.email, body.password)
    if user is None:
        raise HTTPException(401, "Invalid email or password")
    await auth_svc.update_last_login(user.id)
    token = auth_svc.issue_token(user, ttl_seconds=settings.JWT_TTL_DAYS * 86400)
    return TokenResponse(access_token=token, user=_user_to_dict(user))


@router.get("/me")
async def me(request: Request):
    identity = getattr(request.state, "auth_identity", None)
    if identity is None or identity.is_legacy:
        raise HTTPException(401, "Not authenticated")
    user = await auth_svc.get_user_by_id(identity.user_id)
    if user is None:
        raise HTTPException(401, "User no longer exists")
    return {"user": _user_to_dict(user)}


# ── Google OAuth ──────────────────────────────────────────────────────────────


def _google_redirect_uri(request: Request) -> str:
    if settings.GOOGLE_REDIRECT_URI:
        return settings.GOOGLE_REDIRECT_URI
    # Build from request: scheme://host/on/api/auth/google/callback
    base = str(request.base_url).rstrip("/")
    return f"{base}/on/api/auth/google/callback"


@router.get("/google/start")
async def google_start(request: Request):
    """Redirect user to Google's consent screen."""
    if not (settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET):
        raise HTTPException(400, "Google sign-in is not configured")
    state = auth_svc.random_state()
    redirect_uri = _google_redirect_uri(request)
    url = auth_svc.google_auth_url(state, redirect_uri)
    if not url:
        raise HTTPException(500, "Google OAuth misconfigured")
    response = RedirectResponse(url, status_code=302)
    # 5-min state cookie; verified on callback
    response.set_cookie(
        "agentrag_oauth_state",
        state,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=300,
        path="/",
    )
    return response


@router.get("/google/callback")
async def google_callback(request: Request, code: str | None = None, state: str | None = None):
    """Handle Google's redirect; exchange code → JWT and bounce to frontend."""
    if not code:
        raise HTTPException(400, "Missing 'code' query param")
    cookie_state = request.cookies.get("agentrag_oauth_state")
    if not state or not cookie_state or state != cookie_state:
        raise HTTPException(400, "Invalid OAuth state")
    redirect_uri = _google_redirect_uri(request)
    profile = await auth_svc.google_exchange(code, redirect_uri)
    if not profile:
        raise HTTPException(401, "Google authentication failed")
    user = await auth_svc.upsert_google_user(profile)
    await auth_svc.update_last_login(user.id)
    token = auth_svc.issue_token(user, ttl_seconds=settings.JWT_TTL_DAYS * 86400)
    # Bounce to frontend with token in URL fragment so it doesn't hit logs
    target = f"{settings.FRONTEND_URL.rstrip('/')}/login#token={token}"
    response = RedirectResponse(target, status_code=302)
    response.delete_cookie("agentrag_oauth_state", path="/")
    return response


@router.post("/logout")
async def logout():
    """Stateless JWT — logout is client-side. Just acknowledge."""
    return {"message": "Logged out"}
