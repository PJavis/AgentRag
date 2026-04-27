from __future__ import annotations

from src.agentrag.config import Settings

from .base import BaseEmbeddingProvider
from .gemini_embedder import GeminiEmbeddingProvider
from .hf_inference_embedder import HFInferenceEmbeddingProvider
from .openai_embedder import OpenAIEmbeddingProvider


def build_embedding_provider(settings: Settings) -> BaseEmbeddingProvider:
    provider = settings.EMBEDDING_PROVIDER
    model = settings.EMBEDDING_MODEL

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        return OpenAIEmbeddingProvider(
            model=model,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )

    if provider == "ollama":
        return OpenAIEmbeddingProvider(
            model=model,
            api_key=settings.OLLAMA_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL or settings.OLLAMA_BASE_URL,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )

    if provider == "gemini":
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
        return GeminiEmbeddingProvider(
            model=model,
            api_key=settings.GEMINI_API_KEY,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )

    if provider == "hf_inference":
        if not settings.HF_TOKEN:
            raise ValueError(
                "HF_TOKEN is required when EMBEDDING_PROVIDER=hf_inference"
            )
        return HFInferenceEmbeddingProvider(
            model=model,
            api_key=settings.HF_TOKEN,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")
