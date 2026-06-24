from __future__ import annotations

from src.agentrag.config import Settings


def validate_settings(settings: Settings) -> None:
    _validate_embedding_settings(settings)
    _validate_extraction_settings(settings)
    _validate_agent_settings(settings)
    _validate_retrieval_reranker_settings(settings)
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


_API_MODEL_MARKERS = (
    "gemini", "gpt-", "gpt4", "gpt-4", "claude", "deepseek", "o1-", "o3-",
    "text-embedding", "flash", "turbo",
)


def _looks_like_api_model(model: str) -> bool:
    """True when the model name looks like a hosted API/chat model rather than a
    HuggingFace cross-encoder repo id (which is always 'org/name')."""
    m = (model or "").lower()
    return any(marker in m for marker in _API_MODEL_MARKERS)


def _validate_retrieval_reranker_settings(settings: Settings) -> None:
    if not settings.RETRIEVAL_RERANK_ENABLED:
        return

    if settings.RETRIEVAL_RERANK_BACKEND == "local_cross_encoder":
        # Model optional — LLMReranker defaults to dengcao/bge-reranker-v2-m3.
        # Guard the silent trap: an API/chat model name here makes CrossEncoder
        # try to load it from HuggingFace → OSError → swallowed → rerank inert.
        model = settings.RETRIEVAL_RERANK_MODEL
        if model and _looks_like_api_model(model):
            raise ValueError(
                f"RETRIEVAL_RERANK_MODEL={model!r} looks like an API/chat model, but "
                "RETRIEVAL_RERANK_BACKEND=local_cross_encoder loads a HuggingFace "
                "cross-encoder (e.g. dengcao/bge-reranker-v2-m3). Set a cross-encoder "
                "repo id, or leave RETRIEVAL_RERANK_MODEL empty for the default."
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


def _validate_general_settings(settings: Settings) -> None:
    if settings.EMBEDDING_BATCH_SIZE <= 0:
        raise ValueError("EMBEDDING_BATCH_SIZE must be > 0")
    if settings.SEARCH_CHUNK_MAX_TOKENS <= 0:
        raise ValueError("SEARCH_CHUNK_MAX_TOKENS must be > 0")
    if settings.STRUCTMEM_CHUNK_MAX_TOKENS <= 0:
        raise ValueError("STRUCTMEM_CHUNK_MAX_TOKENS must be > 0")
    if settings.STRUCTMEM_MAX_CONCURRENCY <= 0:
        raise ValueError("STRUCTMEM_MAX_CONCURRENCY must be > 0")
    if settings.STRUCTMEM_CHUNK_TIMEOUT_SECONDS <= 0:
        raise ValueError("STRUCTMEM_CHUNK_TIMEOUT_SECONDS must be > 0")
    if settings.STRUCTMEM_CHUNK_RETRIES < 0:
        raise ValueError("STRUCTMEM_CHUNK_RETRIES must be >= 0")
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
    if settings.CHAT_HISTORY_WINDOW <= 0:
        raise ValueError("CHAT_HISTORY_WINDOW must be > 0")
    if settings.CHAT_REDIS_TTL_SECONDS <= 0:
        raise ValueError("CHAT_REDIS_TTL_SECONDS must be > 0")
