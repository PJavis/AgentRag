from __future__ import annotations

import socket
from urllib.parse import urlparse

from src.pam.config import Settings
from src.pam.config_validation import validate_settings


def collect_provider_health(settings: Settings) -> dict:
    validation_error: str | None = None
    try:
        validate_settings(settings)
    except Exception as exc:
        validation_error = str(exc)

    providers = {
        "embedding": _provider_status(
            role="embedding",
            provider=settings.EMBEDDING_PROVIDER,
            model=settings.EMBEDDING_MODEL,
            base_url=_resolve_base_url(
                provider=settings.EMBEDDING_PROVIDER,
                explicit_base_url=settings.EMBEDDING_BASE_URL,
                settings=settings,
            ),
            settings=settings,
        ),
        "extraction": _provider_status(
            role="extraction",
            provider=settings.EXTRACTION_PROVIDER,
            model=settings.EXTRACTION_MODEL,
            base_url=_resolve_base_url(
                provider=settings.EXTRACTION_PROVIDER,
                explicit_base_url=settings.EXTRACTION_BASE_URL,
                settings=settings,
            ),
            settings=settings,
        ),
        "graph_embedding": _provider_status(
            role="graph_embedding",
            provider=settings.GRAPH_EMBEDDING_PROVIDER,
            model=settings.GRAPH_EMBEDDING_MODEL,
            base_url=_resolve_base_url(
                provider=settings.GRAPH_EMBEDDING_PROVIDER,
                explicit_base_url=settings.GRAPH_EMBEDDING_BASE_URL,
                settings=settings,
            ),
            settings=settings,
        ),
    }

    neo4j = _socket_status(settings.NEO4J_URI)

    return {
        "ok": validation_error is None,
        "validation_error": validation_error,
        "providers": providers,
        "neo4j": {
            "uri": settings.NEO4J_URI,
            **neo4j,
        },
    }


def _provider_status(
    role: str,
    provider: str,
    model: str,
    base_url: str | None,
    settings: Settings,
) -> dict:
    token_present = _token_present(provider=provider, settings=settings)
    status = {
        "role": role,
        "provider": provider,
        "model": model,
        "token_present": token_present,
        "base_url": base_url,
        "reachable": None,
        "reachability_error": None,
    }

    if provider in {"ollama"} and base_url:
        reachability = _socket_status(base_url)
        status["reachable"] = reachability["reachable"]
        status["reachability_error"] = reachability["error"]
    elif provider == "hf_inference" and role == "extraction":
        status["reachable"] = None
    elif provider in {"openai", "gemini", "hf_inference"}:
        status["reachable"] = None

    return status


def _resolve_base_url(
    provider: str,
    explicit_base_url: str | None,
    settings: Settings,
) -> str | None:
    if explicit_base_url:
        return explicit_base_url
    if provider == "ollama":
        return settings.OLLAMA_BASE_URL
    if provider == "hf_inference":
        return settings.HF_OPENAI_BASE_URL
    if provider == "gemini":
        return "https://generativelanguage.googleapis.com/v1beta/openai/"
    return None


def _token_present(provider: str, settings: Settings) -> bool:
    if provider == "openai":
        return bool(settings.OPENAI_API_KEY)
    if provider == "gemini":
        return bool(settings.GEMINI_API_KEY)
    if provider == "hf_inference":
        return bool(settings.HF_TOKEN)
    if provider == "ollama":
        return bool(settings.OLLAMA_API_KEY)
    return False


def _socket_status(url_or_uri: str) -> dict[str, str | bool | None]:
    parsed = urlparse(url_or_uri)
    host = parsed.hostname or "localhost"
    port = parsed.port

    if port is None:
        if parsed.scheme in {"http", "https"}:
            port = 443 if parsed.scheme == "https" else 80
        elif parsed.scheme in {"bolt", "neo4j"}:
            port = 7687
        else:
            return {"reachable": None, "error": "could not infer port"}

    try:
        with socket.create_connection((host, port), timeout=1.5):
            return {"reachable": True, "error": None}
    except OSError as exc:
        return {"reachable": False, "error": str(exc)}
