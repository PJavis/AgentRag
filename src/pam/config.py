# src/pam/config.py
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path.cwd() / ".env",  # dùng thư mục hiện tại (gốc dự án) khi chạy lệnh
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str

    POSTGRES_HOST: str = "127.0.0.1"
    POSTGRES_PORT: int = 5433

    HF_TOKEN: str | None = None
    HF_OPENAI_BASE_URL: str = "https://router.huggingface.co/v1"
    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None

    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434/v1/"
    OLLAMA_API_KEY: str = "ollama"

    NEO4J_PASSWORD: str = "neo4j123456"   # default từ docker-compose
    NEO4J_USER: str = "neo4j"
    NEO4J_URI: str = "bolt://localhost:7687"

    #: sync: Graphiti chạy ngay trong ingest. async: đưa vào hàng đợi (cần worker, xem main.py lifespan).
    GRAPH_INGEST_MODE: Literal["sync", "async"] = "async"
    ENABLE_DOCLING_PARSE: bool = False

    EMBEDDING_PROVIDER: Literal["openai", "gemini", "hf_inference", "ollama"] = "hf_inference"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large-instruct"
    EMBEDDING_BASE_URL: str | None = None
    EMBEDDING_BATCH_SIZE: int = 32

    EXTRACTION_PROVIDER: Literal["openai", "gemini", "hf_inference", "ollama"] = "ollama"
    EXTRACTION_MODEL: str = "llama3.1:8b-instruct-q5_K_M"
    EXTRACTION_BASE_URL: str | None = None
    EXTRACTION_TEMPERATURE: float = 0.0

    GRAPH_EMBEDDING_PROVIDER: Literal["openai", "gemini", "hf_inference", "ollama"] = "ollama"
    GRAPH_EMBEDDING_MODEL: str = "nomic-embed-text"
    GRAPH_EMBEDDING_BASE_URL: str | None = None

    #: Chunk cho Postgres / Elasticsearch / embed
    SEARCH_CHUNK_MAX_TOKENS: int = 512
    SEARCH_CHUNK_OVERLAP_TOKENS: int = 64
    #: Chunk riêng cho Graphiti (thường lớn hơn → ít episode, ít vòng LLM)
    GRAPH_CHUNK_MAX_TOKENS: int = 1536
    GRAPH_CHUNK_OVERLAP_TOKENS: int = 128
    CHUNK_TOKENIZER_MODEL: str = "text-embedding-3-large"

    GRAPH_MAX_CONCURRENCY: int = 3
    GRAPH_CHUNK_TIMEOUT_SECONDS: int = 180
    GRAPH_CHUNK_RETRIES: int = 2
    GRAPH_ENABLE_CACHE: bool = True
    GRAPH_CACHE_DIR: str = ".cache/pam/graph"

    @property
    def DATABASE_URL(self) -> str:
        password = quote_plus(self.POSTGRES_PASSWORD)
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{password}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()
