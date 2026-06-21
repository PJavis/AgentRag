from src.agentrag.config import settings


def test_new_enhancement_flags_have_safe_defaults():
    # All new features default OFF so production behavior is unchanged until enabled.
    assert settings.CONTEXTUAL_RETRIEVAL_ENABLED is False
    assert settings.RAPTOR_ENABLED is False
    assert settings.CRAG_ENABLED is False
    assert settings.SEMANTIC_CACHE_ENABLED is False
    # Sensible numeric defaults.
    assert settings.RAPTOR_MIN_LEAVES == 8
    assert settings.SEMANTIC_CACHE_THRESHOLD == 0.97
    assert settings.CONTEXTUAL_RETRIEVAL_TASK == "contextualize"
