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
