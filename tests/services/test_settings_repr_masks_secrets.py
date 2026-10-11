"""repr(settings) — which pydantic also embeds in AttributeError for a missing
setting — must not print API keys, passwords, tokens or secrets."""
from __future__ import annotations

import pytest

from src.agentrag.config import Settings


def _settings():
    return Settings(
        OPENAI_API_KEY="sk-live-openai-xyz",
        DEEPSEEK_API_KEY="sk-live-deepseek-xyz",
        GOOGLE_CLIENT_SECRET="GOCSPX-live-xyz",
        JWT_SECRET="jwt-live-xyz",
        POSTGRES_PASSWORD="pg-live-xyz",
        HF_TOKEN="hf_live_xyz",
    )


SECRETS = ["sk-live-openai-xyz", "sk-live-deepseek-xyz", "GOCSPX-live-xyz",
           "jwt-live-xyz", "pg-live-xyz", "hf_live_xyz"]


def test_repr_and_str_mask_secrets():
    s = _settings()
    for text in (repr(s), str(s)):
        for secret in SECRETS:
            assert secret not in text
        assert "OPENAI_API_KEY='***'" in text


def test_missing_attribute_error_does_not_leak():
    s = _settings()
    with pytest.raises(AttributeError) as exc:
        s.NOT_A_SETTING  # noqa: B018
    for secret in SECRETS:
        assert secret not in str(exc.value)


def test_values_still_readable():
    s = _settings()
    assert s.DEEPSEEK_API_KEY == "sk-live-deepseek-xyz"
    assert "RETRIEVAL_TOP_K=" in repr(s)  # non-secret fields still shown
