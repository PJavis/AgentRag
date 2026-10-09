from src.agentrag.config import Settings


def test_new_enhancement_flags_have_safe_defaults(monkeypatch):
    # All new features default OFF so production behavior is unchanged until enabled.
    # Check the CODE defaults: ignore the local .env and any exported overrides.
    for name in ("CONTEXTUAL_RETRIEVAL_ENABLED", "RAPTOR_ENABLED", "CRAG_ENABLED",
                 "SEMANTIC_CACHE_ENABLED", "RAPTOR_MIN_LEAVES", "SEMANTIC_CACHE_THRESHOLD",
                 "CONTEXTUAL_RETRIEVAL_TASK"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None, POSTGRES_USER="u", POSTGRES_PASSWORD="p", POSTGRES_DB="d")
    assert settings.CONTEXTUAL_RETRIEVAL_ENABLED is False
    assert settings.RAPTOR_ENABLED is False
    assert settings.CRAG_ENABLED is False
    assert settings.SEMANTIC_CACHE_ENABLED is False
    # Sensible numeric defaults.
    assert settings.RAPTOR_MIN_LEAVES == 8
    assert settings.SEMANTIC_CACHE_THRESHOLD == 0.97
    assert settings.CONTEXTUAL_RETRIEVAL_TASK == "contextualize"
