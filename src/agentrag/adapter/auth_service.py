"""User auth: signup/login + JWT + Google OAuth.

Replaces the shared-password model with per-user accounts. JWTs are
emitted as plain bearer tokens so the existing open-notebook frontend's
`Authorization: Bearer <token>` flow keeps working unchanged.
"""
from __future__ import annotations

import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Any

import bcrypt
import httpx
import jwt
from sqlalchemy import select

from src.agentrag.config import settings
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import User

JWT_ALGORITHM = "HS256"
JWT_DEFAULT_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days
LEGACY_PASSWORD_USER_ID = "legacy-password"


@dataclass
class AuthIdentity:
    """Resolved request actor — either a real user or the legacy shared password."""

    user_id: str
    email: str | None
    name: str | None
    is_admin: bool
    is_legacy: bool = False


def _jwt_secret() -> str:
    """Return JWT signing secret. Falls back to a derived value but warns once."""
    secret = settings.JWT_SECRET
    if secret:
        return secret
    # Derive a deterministic secret from PG creds so tokens survive restarts
    # in single-instance dev. NOT safe for multi-host deployment.
    return f"agentrag-dev-{settings.POSTGRES_PASSWORD}-{settings.POSTGRES_DB}"


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str | None) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def issue_token(user: User, ttl_seconds: int = JWT_DEFAULT_TTL_SECONDS) -> str:
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "name": user.name,
        "admin": bool(user.is_admin),
        "iat": int(time.time()),
        "exp": int(time.time()) + ttl_seconds,
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any] | None:
    try:
        return jwt.decode(token, _jwt_secret(), algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


async def get_user_by_email(email: str) -> User | None:
    async with AsyncSessionLocal() as session:
        row = await session.execute(select(User).where(User.email == email.lower()))
        return row.scalar_one_or_none()


async def get_user_by_id(user_id: str) -> User | None:
    try:
        uid = uuid.UUID(user_id)
    except (ValueError, TypeError):
        return None
    async with AsyncSessionLocal() as session:
        return await session.get(User, uid)


def _is_admin_email(email: str) -> bool:
    """Check ADMIN_EMAILS whitelist (case-insensitive comma-separated list)."""
    from src.agentrag.config import settings as _settings
    raw = (_settings.ADMIN_EMAILS or "").strip()
    if not raw:
        return False
    target = email.lower().strip()
    return any(target == e.strip().lower() for e in raw.split(",") if e.strip())


async def create_user(
    email: str,
    password: str | None = None,
    name: str | None = None,
    google_id: str | None = None,
    avatar_url: str | None = None,
) -> User:
    async with AsyncSessionLocal() as session:
        user = User(
            email=email.lower(),
            password_hash=hash_password(password) if password else None,
            name=name,
            google_id=google_id,
            avatar_url=avatar_url,
            is_admin=_is_admin_email(email),  # bootstrap from ADMIN_EMAILS env
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


async def update_last_login(user_id: uuid.UUID) -> None:
    from datetime import datetime, timezone

    async with AsyncSessionLocal() as session:
        u = await session.get(User, user_id)
        if u:
            u.last_login_at = datetime.now(timezone.utc)
            await session.commit()


async def authenticate(email: str, password: str) -> User | None:
    user = await get_user_by_email(email)
    if user is None:
        return None
    if not verify_password(password, user.password_hash):
        return None
    # Auto-promote on login when email is on the ADMIN_EMAILS whitelist
    # (one-way: never demotes an existing admin).
    if not user.is_admin and _is_admin_email(user.email):
        async with AsyncSessionLocal() as session:
            u = await session.get(User, user.id)
            if u and not u.is_admin:
                u.is_admin = True
                await session.commit()
                user.is_admin = True
    return user


# ── Google OAuth ──────────────────────────────────────────────────────────────

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


def google_auth_url(state: str, redirect_uri: str) -> str | None:
    if not settings.GOOGLE_CLIENT_ID:
        return None
    from urllib.parse import urlencode

    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


async def google_exchange(code: str, redirect_uri: str) -> dict[str, Any] | None:
    if not (settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET):
        return None
    async with httpx.AsyncClient(timeout=10.0) as client:
        token_resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        if token_resp.status_code >= 400:
            return None
        access_token = token_resp.json().get("access_token")
        if not access_token:
            return None
        userinfo = await client.get(
            GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"}
        )
        if userinfo.status_code >= 400:
            return None
        return userinfo.json()


async def upsert_google_user(profile: dict[str, Any]) -> User:
    """Create or update a user from Google userinfo payload."""
    google_id = str(profile.get("sub") or "")
    email = (profile.get("email") or "").lower()
    if not email or not google_id:
        raise ValueError("Google profile missing email or sub")

    user = await get_user_by_email(email)
    if user is None:
        return await create_user(
            email=email,
            password=None,
            name=profile.get("name"),
            google_id=google_id,
            avatar_url=profile.get("picture"),
        )
    # Backfill google_id / avatar if missing
    async with AsyncSessionLocal() as session:
        u = await session.get(User, user.id)
        if not u.google_id:
            u.google_id = google_id
        if profile.get("picture") and not u.avatar_url:
            u.avatar_url = profile["picture"]
        if profile.get("name") and not u.name:
            u.name = profile["name"]
        await session.commit()
        await session.refresh(u)
        return u


# ── Token helpers ─────────────────────────────────────────────────────────────


def random_state() -> str:
    return secrets.token_urlsafe(24)
