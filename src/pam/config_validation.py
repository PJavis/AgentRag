from __future__ import annotations

from src.pam.config import Settings


def validate_settings(settings: Settings) -> None:
    _validate_embedding_settings(settings)
    _validate_extraction_settings(settings)
    _validate_graph_embedding_settings(settings)
    _validate_general_settings(settings)


def _validate_embedding_settings(settings: Settings) -> None:
    provider = settings.EMBEDDING_PROVIDER
    if not settings.EMBEDDING_MODEL:
        raise ValueError("EMBEDDING_MODEL is required")

    if provider == "openai" and not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
    if provider == "gemini" and not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
    if provider == "hf_inference" and not settings.HF_TOKEN:
        raise ValueError("HF_TOKEN is required when EMBEDDING_PROVIDER=hf_inference")
    if provider == "ollama" and not (settings.EMBEDDING_BASE_URL or settings.OLLAMA_BASE_URL):
        raise ValueError(
            "EMBEDDING_BASE_URL or OLLAMA_BASE_URL is required when EMBEDDING_PROVIDER=ollama"
        )


def _validate_extraction_settings(settings: Settings) -> None:
    provider = settings.EXTRACTION_PROVIDER
    if not settings.EXTRACTION_MODEL:
        raise ValueError("EXTRACTION_MODEL is required")

    if provider == "openai" and not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required when EXTRACTION_PROVIDER=openai")
    if provider == "gemini" and not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required when EXTRACTION_PROVIDER=gemini")
    if provider == "hf_inference":
        if not settings.HF_TOKEN:
            raise ValueError("HF_TOKEN is required when EXTRACTION_PROVIDER=hf_inference")
        if not (settings.EXTRACTION_BASE_URL or settings.HF_OPENAI_BASE_URL):
            raise ValueError(
                "EXTRACTION_BASE_URL or HF_OPENAI_BASE_URL is required when EXTRACTION_PROVIDER=hf_inference"
            )
    if provider == "ollama" and not (settings.EXTRACTION_BASE_URL or settings.OLLAMA_BASE_URL):
        raise ValueError(
            "EXTRACTION_BASE_URL or OLLAMA_BASE_URL is required when EXTRACTION_PROVIDER=ollama"
        )


def _validate_graph_embedding_settings(settings: Settings) -> None:
    provider = settings.GRAPH_EMBEDDING_PROVIDER
    if not settings.GRAPH_EMBEDDING_MODEL:
        raise ValueError("GRAPH_EMBEDDING_MODEL is required")

    if provider == "openai" and not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is required when GRAPH_EMBEDDING_PROVIDER=openai"
        )
    if provider == "gemini" and not settings.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is required when GRAPH_EMBEDDING_PROVIDER=gemini"
        )
    if provider == "hf_inference" and not settings.HF_TOKEN:
        raise ValueError(
            "HF_TOKEN is required when GRAPH_EMBEDDING_PROVIDER=hf_inference"
        )
    if provider == "ollama" and not (
        settings.GRAPH_EMBEDDING_BASE_URL or settings.OLLAMA_BASE_URL
    ):
        raise ValueError(
            "GRAPH_EMBEDDING_BASE_URL or OLLAMA_BASE_URL is required when GRAPH_EMBEDDING_PROVIDER=ollama"
        )


def _validate_general_settings(settings: Settings) -> None:
    if settings.EMBEDDING_BATCH_SIZE <= 0:
        raise ValueError("EMBEDDING_BATCH_SIZE must be > 0")
    if settings.SEARCH_CHUNK_MAX_TOKENS <= 0:
        raise ValueError("SEARCH_CHUNK_MAX_TOKENS must be > 0")
    if settings.GRAPH_CHUNK_MAX_TOKENS <= 0:
        raise ValueError("GRAPH_CHUNK_MAX_TOKENS must be > 0")
    if settings.GRAPH_MAX_CONCURRENCY <= 0:
        raise ValueError("GRAPH_MAX_CONCURRENCY must be > 0")
    if settings.GRAPH_CHUNK_TIMEOUT_SECONDS <= 0:
        raise ValueError("GRAPH_CHUNK_TIMEOUT_SECONDS must be > 0")
    if settings.GRAPH_CHUNK_RETRIES < 0:
        raise ValueError("GRAPH_CHUNK_RETRIES must be >= 0")
