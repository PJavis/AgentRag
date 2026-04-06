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

    # Optional agent-specific runtime model config.
    # If omitted, agent falls back to EXTRACTION_* settings.
    AGENT_PROVIDER: Literal["openai", "gemini", "hf_inference", "ollama"] | None = None
    AGENT_MODEL: str | None = None
    AGENT_BASE_URL: str | None = None
    AGENT_TEMPERATURE: float | None = None

    GRAPH_EMBEDDING_PROVIDER: Literal["openai", "gemini", "hf_inference", "ollama"] = "ollama"
    GRAPH_EMBEDDING_MODEL: str = "nomic-embed-text"
    GRAPH_EMBEDDING_BASE_URL: str | None = None

    #: Chunk cho Postgres / Elasticsearch / embed
    SEARCH_CHUNK_MAX_TOKENS: int = 512
    SEARCH_CHUNK_OVERLAP_TOKENS: int = 64
    SEARCH_CHUNK_BY_PARAGRAPH: bool = True
    #: Chunk riêng cho Graphiti (thường lớn hơn → ít episode, ít vòng LLM)
    GRAPH_CHUNK_MAX_TOKENS: int = 1536
    GRAPH_CHUNK_OVERLAP_TOKENS: int = 128
    CHUNK_TOKENIZER_MODEL: str = "text-embedding-3-large"

    GRAPH_MAX_CONCURRENCY: int = 1
    GRAPH_CHUNK_TIMEOUT_SECONDS: int = 300
    GRAPH_CHUNK_RETRIES: int = 3
    GRAPH_ENABLE_CACHE: bool = True
    GRAPH_CACHE_DIR: str = ".cache/pam/graph"

    ELASTICSEARCH_URL: str = "http://localhost:9200"
    ELASTICSEARCH_INDEX_NAME: str = "pam_segments"
    ELASTICSEARCH_ENTITY_INDEX_NAME: str = "pam_entities"
    ELASTICSEARCH_RELATIONSHIP_INDEX_NAME: str = "pam_relationships"
    REDIS_URL: str | None = "redis://127.0.0.1:6379/0"
    RETRIEVAL_TOP_K: int = 10
    RETRIEVAL_NUM_CANDIDATES: int = 50
    RETRIEVAL_RRF_K: int = 60
    RETRIEVAL_RERANK_ENABLED: bool = False
    RETRIEVAL_RERANK_TOP_N: int = 20
    RETRIEVAL_RERANK_BACKEND: Literal["llm_chat", "local_cross_encoder"] = "llm_chat"
    RETRIEVAL_RERANK_PROVIDER: Literal["openai", "gemini", "hf_inference", "ollama"] | None = None
    RETRIEVAL_RERANK_MODEL: str | None = None
    RETRIEVAL_RERANK_BASE_URL: str | None = None
    RETRIEVAL_RERANK_TEMPERATURE: float = 0.0

    AGENT_MAX_STEPS: int = 4
    AGENT_TOOL_TOP_K: int = 5
    AGENT_MAX_CONTEXT_CHUNKS: int = 8
    CHAT_HISTORY_WINDOW: int = 10
    CHAT_REDIS_TTL_SECONDS: int = 300

    # Structured SQL Reasoning (ADR 0002)
    STRUCTURED_REASONING_ENABLED: bool = True
    STRUCTURED_CLASSIFIER_METHOD: Literal["rule", "llm", "rule+llm"] = "rule+llm"
    STRUCTURED_MAX_CHUNKS_FOR_SCHEMA: int = 10
    STRUCTURED_MAX_CHUNKS_FOR_EXTRACT: int = 20
    STRUCTURED_SQL_MAX_RETRIES: int = 2
    STRUCTURED_CONFIDENCE_THRESHOLD: float = 0.7

    # LLM Routing (ADR 0001 Phase C)
    LLM_ROUTING_ENABLED: bool = False
    LLM_TASK_MODEL_MAP: str = "{}"
    LLM_COST_TRACKING_ENABLED: bool = False

    # Multimodal / Vision (image description at ingest)
    VISION_ENABLED: bool = False
    VISION_PROVIDER: Literal["openai", "gemini", "ollama"] = "gemini"
    VISION_MODEL: str = "gemini-2.0-flash"
    VISION_BASE_URL: str | None = None
    VISION_MAX_IMAGES_PER_DOC: int = 20

    # Excel ingest strategy
    EXCEL_INGEST_MODE: Literal["markdown", "sql"] = "markdown"
    # markdown: sheet → markdown table → chunk như text thường
    # sql:      sheet → SQLite, query trực tiếp qua structured pipeline

    # Observability (ADR 0001 Phase B)
    OBSERVABILITY_TRACE_ENABLED: bool = True

    @property
    def DATABASE_URL(self) -> str:
        password = quote_plus(self.POSTGRES_PASSWORD)
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{password}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()
