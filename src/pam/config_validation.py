from __future__ import annotations

from src.pam.config import Settings


def validate_settings(settings: Settings) -> None:
    _validate_embedding_settings(settings)
    _validate_extraction_settings(settings)
    _validate_agent_settings(settings)
    _validate_retrieval_reranker_settings(settings)
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


def _validate_agent_settings(settings: Settings) -> None:
    provider = settings.AGENT_PROVIDER or settings.EXTRACTION_PROVIDER
    model = settings.AGENT_MODEL or settings.EXTRACTION_MODEL

    if settings.AGENT_PROVIDER and not settings.AGENT_MODEL:
        raise ValueError("AGENT_MODEL is required when AGENT_PROVIDER is set")
    if not model:
        raise ValueError("AGENT_MODEL or EXTRACTION_MODEL is required for agent runtime")

    if provider == "openai" and not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required when AGENT provider resolves to openai")
    if provider == "gemini" and not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required when AGENT provider resolves to gemini")
    if provider == "hf_inference":
        if not settings.HF_TOKEN:
            raise ValueError("HF_TOKEN is required when AGENT provider resolves to hf_inference")
        if not (settings.AGENT_BASE_URL or settings.EXTRACTION_BASE_URL or settings.HF_OPENAI_BASE_URL):
            raise ValueError(
                "AGENT_BASE_URL or EXTRACTION_BASE_URL or HF_OPENAI_BASE_URL is required "
                "when AGENT provider resolves to hf_inference"
            )
    if provider == "ollama" and not (
        settings.AGENT_BASE_URL or settings.EXTRACTION_BASE_URL or settings.OLLAMA_BASE_URL
    ):
        raise ValueError(
            "AGENT_BASE_URL or EXTRACTION_BASE_URL or OLLAMA_BASE_URL is required "
            "when AGENT provider resolves to ollama"
        )


def _validate_retrieval_reranker_settings(settings: Settings) -> None:
    if not settings.RETRIEVAL_RERANK_ENABLED:
        return

    if settings.RETRIEVAL_RERANK_BACKEND == "local_cross_encoder":
        if not settings.RETRIEVAL_RERANK_MODEL:
            raise ValueError(
                "RETRIEVAL_RERANK_MODEL is required when RETRIEVAL_RERANK_BACKEND=local_cross_encoder"
            )
        return

    provider = (
        settings.RETRIEVAL_RERANK_PROVIDER
        or settings.AGENT_PROVIDER
        or settings.EXTRACTION_PROVIDER
    )
    model = (
        settings.RETRIEVAL_RERANK_MODEL
        or settings.AGENT_MODEL
        or settings.EXTRACTION_MODEL
    )
    if not model:
        raise ValueError(
            "RETRIEVAL_RERANK_MODEL or AGENT_MODEL or EXTRACTION_MODEL is required when RETRIEVAL_RERANK_ENABLED=true"
        )

    if provider == "openai" and not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required when reranker provider resolves to openai")
    if provider == "gemini" and not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required when reranker provider resolves to gemini")
    if provider == "hf_inference" and not settings.HF_TOKEN:
        raise ValueError("HF_TOKEN is required when reranker provider resolves to hf_inference")
    if provider == "ollama" and not (
        settings.RETRIEVAL_RERANK_BASE_URL
        or settings.AGENT_BASE_URL
        or settings.EXTRACTION_BASE_URL
        or settings.OLLAMA_BASE_URL
    ):
        raise ValueError(
            "RETRIEVAL_RERANK_BASE_URL or AGENT_BASE_URL or EXTRACTION_BASE_URL or OLLAMA_BASE_URL "
            "is required when reranker provider resolves to ollama"
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
    if settings.RETRIEVAL_TOP_K <= 0:
        raise ValueError("RETRIEVAL_TOP_K must be > 0")
    if settings.RETRIEVAL_NUM_CANDIDATES <= 0:
        raise ValueError("RETRIEVAL_NUM_CANDIDATES must be > 0")
    if settings.RETRIEVAL_RRF_K <= 0:
        raise ValueError("RETRIEVAL_RRF_K must be > 0")
    if settings.RETRIEVAL_RERANK_TOP_N <= 0:
        raise ValueError("RETRIEVAL_RERANK_TOP_N must be > 0")
    if settings.AGENT_MAX_STEPS <= 0:
        raise ValueError("AGENT_MAX_STEPS must be > 0")
    if settings.AGENT_TOOL_TOP_K <= 0:
        raise ValueError("AGENT_TOOL_TOP_K must be > 0")
    if settings.AGENT_MAX_CONTEXT_CHUNKS <= 0:
        raise ValueError("AGENT_MAX_CONTEXT_CHUNKS must be > 0")
