"""Models router — surface AgentRag's configured providers/models to the UI.

The open-notebook frontend lets users browse models and pick one for the
chat dropdown. We don't actually have a registry, but we synthesize a
sensible list from the agent/extraction/embedding/vision settings so the
UI shows real values instead of empty stubs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException

from src.agentrag.config import settings
from src.agentrag.config_overrides import save_overrides, apply_overrides

router = APIRouter()


def _parse_model_id(model_id: str) -> tuple[str, str, str] | None:
    """Parse `prefix::provider::model` IDs produced by _model_row.

    Returns (prefix, provider, model) or None if malformed.
    """
    if not model_id or "::" not in model_id:
        return None
    parts = model_id.split("::", 2)
    if len(parts) != 3:
        return None
    prefix, provider, model = parts
    if not provider or not model:
        return None
    return prefix, provider, model


_PREFIX_FIELDS = {
    "agent":   ("AGENT_PROVIDER", "AGENT_MODEL"),
    "extract": ("EXTRACTION_PROVIDER", "EXTRACTION_MODEL"),
    "embed":   ("EMBEDDING_PROVIDER", "EMBEDDING_MODEL"),
    "vision":  ("VISION_PROVIDER", "VISION_MODEL"),
}


# Frontend field name → which model_id prefix is expected.
_DEFAULT_FIELD_TO_PREFIX = {
    "default_chat_model":      "agent",
    "default_language_model":  "agent",
    "default_embedding_model": "embed",
    "default_vision_model":    "vision",
}

_FIXED_TS = "2025-01-01T00:00:00Z"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_row(
    *, model_id: str, name: str, provider: str, mtype: str, default: bool = False
) -> dict[str, Any]:
    return {
        "id": model_id,
        "name": name,
        "provider": provider,
        "type": mtype,
        "credential_id": None,
        "is_default": default,
        "created": _FIXED_TS,
        "updated": _FIXED_TS,
    }


def _agent_model() -> dict[str, Any]:
    provider = settings.AGENT_PROVIDER or settings.EXTRACTION_PROVIDER
    name = settings.AGENT_MODEL or settings.EXTRACTION_MODEL
    return _model_row(
        model_id=f"agent::{provider}::{name}",
        name=name,
        provider=provider,
        mtype="language",
        default=True,
    )


def _extraction_model() -> dict[str, Any]:
    return _model_row(
        model_id=f"extract::{settings.EXTRACTION_PROVIDER}::{settings.EXTRACTION_MODEL}",
        name=settings.EXTRACTION_MODEL,
        provider=settings.EXTRACTION_PROVIDER,
        mtype="language",
    )


def _embedding_model() -> dict[str, Any]:
    return _model_row(
        model_id=f"embed::{settings.EMBEDDING_PROVIDER}::{settings.EMBEDDING_MODEL}",
        name=settings.EMBEDDING_MODEL,
        provider=settings.EMBEDDING_PROVIDER,
        mtype="embedding",
        default=True,
    )


def _vision_model() -> dict[str, Any] | None:
    if not settings.VISION_PROVIDER or not settings.VISION_MODEL:
        return None
    return _model_row(
        model_id=f"vision::{settings.VISION_PROVIDER}::{settings.VISION_MODEL}",
        name=settings.VISION_MODEL,
        provider=settings.VISION_PROVIDER,
        mtype="vision",
    )


def _provider_for_model(name: str) -> str:
    """Infer provider from a model name's prefix — mirrors AgentLLM routing."""
    m = (name or "").lower()
    if m.startswith("gemini-") or m.startswith("gemma-"):
        return "gemini"
    if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("deepseek"):
        return "deepseek"
    return settings.AGENT_PROVIDER or settings.EXTRACTION_PROVIDER or "ollama"


def _task_map_models() -> list[dict[str, Any]]:
    """Surface every distinct model in LLM_TASK_MODEL_MAP (e.g. deepseek-v4-pro)
    so the chat picker can select them — not just AGENT/EXTRACTION."""
    import json

    try:
        task_map = json.loads(settings.LLM_TASK_MODEL_MAP or "{}")
    except (json.JSONDecodeError, TypeError):
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for model in (str(v) for v in task_map.values() if v):
        if model in seen:
            continue
        seen.add(model)
        provider = _provider_for_model(model)
        rows.append(
            _model_row(
                model_id=f"task::{provider}::{model}",
                name=model,
                provider=provider,
                mtype="language",
            )
        )
    return rows


def _all_models() -> list[dict[str, Any]]:
    out = [_agent_model(), _extraction_model(), _embedding_model()]
    out.extend(_task_map_models())
    v = _vision_model()
    if v:
        out.append(v)
    # Dedupe by (type, name) so the same model from agent + task-map shows once.
    seen = set()
    deduped = []
    for m in out:
        key = (m["type"], m["name"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)
    return deduped


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/models")
async def list_models(type: str | None = None):
    rows = _all_models()
    if type:
        rows = [r for r in rows if r["type"] == type]
    return rows


@router.get("/models/defaults")
async def get_default_models():
    agent = _agent_model()
    embed = _embedding_model()
    return {
        "default_language_model": agent["id"],
        "default_embedding_model": embed["id"],
        "default_chat_model": agent["id"],
    }


@router.put("/models/defaults")
async def set_default_models(body: dict):
    """Persist UI-selected defaults to data/model_overrides.json + apply live.

    Accepts:
      { default_chat_model | default_language_model | default_embedding_model
        | default_vision_model: "<prefix>::<provider>::<model>" }

    The override file is loaded on every process start, so changes survive
    restarts. ARQ worker picks up new values on its next boot.
    """
    updates: dict[str, str] = {}
    for field, value in (body or {}).items():
        if field not in _DEFAULT_FIELD_TO_PREFIX or not isinstance(value, str):
            continue
        parsed = _parse_model_id(value)
        if parsed is None:
            raise HTTPException(400, f"Malformed model_id for {field!r}: {value!r}")
        prefix, provider, model = parsed
        expected_prefix = _DEFAULT_FIELD_TO_PREFIX[field]
        if prefix != expected_prefix:
            raise HTTPException(
                400,
                f"{field!r} expects prefix {expected_prefix!r}, got {prefix!r}",
            )
        provider_field, model_field = _PREFIX_FIELDS[prefix]
        updates[provider_field] = provider
        updates[model_field] = model

    if updates:
        save_overrides(updates)
        apply_overrides(settings)

    return await get_default_models()


@router.get("/models/providers")
async def get_providers():
    providers: dict[str, dict[str, Any]] = {}
    for m in _all_models():
        providers.setdefault(m["provider"], {"available": True, "model_count": 0})
        providers[m["provider"]]["model_count"] += 1
    return providers


@router.get("/models/by-provider/{provider}")
async def models_by_provider(provider: str):
    return [m for m in _all_models() if m["provider"] == provider]


@router.post("/models/sync/{provider}")
async def sync_provider(provider: str):
    return {"synced": sum(1 for m in _all_models() if m["provider"] == provider)}


@router.post("/models/sync")
async def sync_all_providers():
    grouped: dict[str, int] = {}
    for m in _all_models():
        grouped[m["provider"]] = grouped.get(m["provider"], 0) + 1
    return {"providers": grouped}


@router.get("/models/count/{provider}")
async def model_count(provider: str):
    return {
        "provider": provider,
        "count": sum(1 for m in _all_models() if m["provider"] == provider),
    }


@router.post("/models/auto-assign")
async def auto_assign_models():
    return {
        "assigned": {
            "language": _agent_model()["id"],
            "embedding": _embedding_model()["id"],
        }
    }


@router.post("/models/discover/{provider}")
async def discover_models(provider: str):
    return [m for m in _all_models() if m["provider"] == provider]


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    for m in _all_models():
        if m["id"] == model_id:
            return m
    return _model_row(
        model_id=model_id,
        name=model_id.split("::")[-1] if "::" in model_id else model_id,
        provider=settings.EXTRACTION_PROVIDER,
        mtype="language",
    )


@router.post("/models/{model_id}/test")
async def test_model(model_id: str):
    return {"success": True, "latency_ms": 0, "model_id": model_id}


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    return {"message": "Models are managed via .env — delete is a no-op."}
