"""Startup validation for vision settings.

Both rules guard silent-success traps: ingest "succeeds" while scanned pages are
described by a TEXT model and come out as ~40 chars, then get filtered away.
"""
from __future__ import annotations

import copy

import pytest

from src.agentrag.config import settings
from src.agentrag.config_validation import _validate_vision_settings, validate_settings


def _vision(**overrides):
    s = copy.copy(settings)
    s.VISION_PROVIDER = "gemini"
    s.VISION_MODEL = "gemini-2.5-flash"
    s.VISION_BASE_URL = None
    s.GEMINI_API_KEY = "test-key"
    # Pin every env-dependent input: these tests must not pass or fail on the
    # developer's .env (an unset OPENAI_API_KEY used to break the openai cases).
    s.OPENAI_API_KEY = "test-key"
    s.OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1/"
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


def test_vision_disabled_passes():
    _validate_vision_settings(_vision(VISION_PROVIDER=None, VISION_MODEL=None))


def test_gemini_with_default_endpoint_passes():
    _validate_vision_settings(_vision())


def test_gemini_with_gemini_endpoint_passes():
    _validate_vision_settings(
        _vision(VISION_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/")
    )


def test_provider_without_model_fails():
    # llm_gateway falls back to EXTRACTION_MODEL (a text model) when VISION_MODEL is empty.
    with pytest.raises(ValueError, match="VISION_MODEL"):
        _validate_vision_settings(_vision(VISION_MODEL=None))


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434/v1/",
        "http://host.docker.internal:11434/v1",
        "http://ollama:11434/v1/",
    ],
)
def test_cloud_provider_pointed_at_ollama_fails(url):
    with pytest.raises(ValueError, match="VISION_BASE_URL"):
        _validate_vision_settings(_vision(VISION_BASE_URL=url))


def test_cloud_provider_pointed_at_ollama_service_without_port_fails():
    with pytest.raises(ValueError, match="VISION_BASE_URL"):
        _validate_vision_settings(_vision(VISION_PROVIDER="openai", VISION_BASE_URL="http://ollama/v1"))


def test_openai_compatible_client_on_ollama_with_local_model_passes():
    _validate_vision_settings(
        _vision(VISION_PROVIDER="openai", VISION_MODEL="llava:13b", VISION_BASE_URL="http://127.0.0.1:11434/v1/")
    )


def test_ollama_provider_with_ollama_endpoint_passes():
    _validate_vision_settings(
        _vision(VISION_PROVIDER="ollama", VISION_MODEL="llava:13b", VISION_BASE_URL="http://127.0.0.1:11434/v1/")
    )


def test_gemini_provider_requires_key():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        _validate_vision_settings(_vision(GEMINI_API_KEY=None))


def test_validate_settings_runs_vision_rules():
    with pytest.raises(ValueError, match="VISION_BASE_URL"):
        validate_settings(_vision(VISION_BASE_URL="http://127.0.0.1:11434/v1/"))


def test_cloud_provider_on_remapped_ollama_port_fails():
    # Ollama behind a proxy / remapped port: caught by comparing with OLLAMA_BASE_URL.
    with pytest.raises(ValueError, match="VISION_BASE_URL"):
        _validate_vision_settings(
            _vision(OLLAMA_BASE_URL="http://gpu-box:8434/v1", VISION_BASE_URL="http://gpu-box:8434/v1/")
        )


def test_openai_o4_model_on_ollama_fails():
    with pytest.raises(ValueError, match="VISION_BASE_URL"):
        _validate_vision_settings(
            _vision(VISION_PROVIDER="openai", VISION_MODEL="o4-mini", VISION_BASE_URL="http://127.0.0.1:11434/v1/")
        )


def test_ollama_provider_left_on_gemini_endpoint_fails():
    with pytest.raises(ValueError, match="not an Ollama"):
        _validate_vision_settings(
            _vision(
                VISION_PROVIDER="ollama", VISION_MODEL="llava:13b",
                VISION_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        )


def test_ollama_provider_with_hosted_model_name_fails():
    with pytest.raises(ValueError, match="hosted-API model"):
        _validate_vision_settings(_vision(VISION_PROVIDER="ollama", VISION_MODEL="gemini-2.5-flash"))


def test_ollama_provider_without_base_url_passes():
    _validate_vision_settings(_vision(VISION_PROVIDER="ollama", VISION_MODEL="qwen2.5vl:7b"))


async def test_ui_model_defaults_reject_bad_vision_combo_without_saving(monkeypatch):
    # PUT /models/defaults used to persist + apply without validation; a bad
    # combo then stopped the api from booting on the next restart.
    from fastapi import HTTPException

    from src.agentrag.adapter.routers import models as models_router

    saved = []
    monkeypatch.setattr(models_router, "save_overrides", lambda u: saved.append(u))
    monkeypatch.setattr(models_router, "apply_overrides", lambda s: None)
    monkeypatch.setattr(settings, "VISION_BASE_URL", "http://127.0.0.1:11434/v1/")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "k")
    with pytest.raises(HTTPException) as exc:
        await models_router.set_default_models(
            {"default_vision_model": "vision::gemini::gemini-2.5-flash"}
        )
    assert exc.value.status_code == 400
    assert "VISION_BASE_URL" in exc.value.detail
    assert saved == []
