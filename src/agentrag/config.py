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
        env_ignore_empty=True,  # treat empty env vars as missing (use defaults)
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

    #: sync: StructMem chạy ngay trong ingest. async: đưa vào hàng đợi (cần worker, xem main.py lifespan).
    STRUCTMEM_INGEST_MODE: Literal["sync", "async"] = "async"

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

    #: Chunk cho Postgres / Elasticsearch / embed
    SEARCH_CHUNK_MAX_TOKENS: int = 512
    SEARCH_CHUNK_OVERLAP_TOKENS: int = 64
    SEARCH_CHUNK_BY_PARAGRAPH: bool = True
    #: Chunk riêng cho Graphiti (thường lớn hơn → ít episode, ít vòng LLM)
    STRUCTMEM_CHUNK_MAX_TOKENS: int = 1536
    STRUCTMEM_CHUNK_OVERLAP_TOKENS: int = 128
    CHUNK_TOKENIZER_MODEL: str = "text-embedding-3-large"

    # Number of chunks extracted in parallel within a single document job
    STRUCTMEM_MAX_CONCURRENCY: int = 1
    # Number of documents a single ARQ worker processes simultaneously.
    # Keep at 1 for local Ollama to avoid GPU thrashing; raise for cloud APIs.
    STRUCTMEM_WORKER_MAX_JOBS: int = 1
    STRUCTMEM_CHUNK_TIMEOUT_SECONDS: int = 300
    STRUCTMEM_CHUNK_RETRIES: int = 3
    # Total timeout per ARQ job (seconds) — must be >> CHUNK_TIMEOUT * MAX_CONCURRENCY
    STRUCTMEM_JOB_TIMEOUT_SECONDS: int = 3600
    STRUCTMEM_ENABLE_CACHE: bool = True
    STRUCTMEM_CACHE_DIR: str = ".cache/agentrag/extract"

    ELASTICSEARCH_URL: str = "http://localhost:9200"
    ELASTICSEARCH_INDEX_NAME: str = "agentrag_segments"
    ELASTICSEARCH_ENTITY_INDEX_NAME: str = "agentrag_entities"
    ELASTICSEARCH_RELATIONSHIP_INDEX_NAME: str = "agentrag_relationships"

    # StructMem — thay thế Graphiti graph extraction
    STRUCTMEM_ENABLED: bool = True
    STRUCTMEM_ENTRIES_INDEX_NAME: str = "agentrag_entries"
    STRUCTMEM_SYNTHESIS_INDEX_NAME: str = "agentrag_synthesis"
    # Số chunks/group tích luỹ trước khi trigger cross-chunk consolidation
    STRUCTMEM_CONSOLIDATION_THRESHOLD: int = 20
    # Top-K historical entries làm seed trong consolidation
    STRUCTMEM_CONSOLIDATION_HISTORY_TOP_K: int = 15
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

    # Chat StructMem — áp dụng dual-perspective memory cho lịch sử hội thoại
    CHAT_STRUCTMEM_ENABLED: bool = False
    CHAT_MEMORY_INDEX: str = "agentrag_chat_entries"
    CHAT_MEMORY_SYNTHESIS_INDEX: str = "agentrag_chat_synthesis"
    CHAT_MEMORY_CONSOLIDATION_THRESHOLD: int = 10  # số turns tích luỹ trước khi consolidate
    CHAT_MEMORY_TOP_K: int = 8  # số entries retrieve mỗi lượt

    # LLM Routing (ADR 0001 Phase C)
    LLM_ROUTING_ENABLED: bool = False
    LLM_TASK_MODEL_MAP: str = "{}"
    LLM_COST_TRACKING_ENABLED: bool = False

    # PDF parser backend: pymupdf (page-aware, recommended) or markitdown (legacy)
    PDF_PARSER_BACKEND: Literal["pymupdf", "markitdown"] = "pymupdf"

    # Vision LLM — for describing images extracted from PDFs and standalone image files.
    # If VISION_PROVIDER is None, image extraction is skipped (text-only ingestion).
    VISION_PROVIDER: Literal["openai", "gemini", "ollama"] | None = None
    VISION_MODEL: str | None = None          # e.g. gpt-4o | gemini-1.5-flash | llava:13b
    VISION_BASE_URL: str | None = None
    IMAGE_STORAGE_DIR: str = "data/images"
    IMAGE_MIN_SIZE_BYTES: int = 5000         # skip icons / decorative bullets
    VISION_TIMEOUT_SECONDS: int = 180        # bump cao cho llava cold-start
    # sync: describe images inline during ingest (slow, blocks pipeline).
    # async: skip describe; queue ARQ vision_extract job → text retrieval ready ngay.
    VISION_INGEST_MODE: Literal["sync", "async"] = "async"
    # Max concurrent describe calls in vision_extract worker job.
    # Cloud APIs (Gemini/OpenAI): 4-8. Local Ollama: keep at 1 to avoid GPU thrashing.
    VISION_MAX_CONCURRENCY: int = 4
    # Max requests-per-minute against the vision provider. Token-bucket smoothing.
    # Gemini free tier: 10 RPM for 2.5-flash. Tier 1 paid: ~1000 RPM.
    # Set to 0 to disable RPM cap (still bounded by VISION_MAX_CONCURRENCY).
    VISION_MAX_RPM: int = 10
    # Retry transient 429/5xx per image before giving up. 0 = no retry.
    VISION_PER_IMAGE_RETRIES: int = 3
    # Flush described chunks → PG + ES every N images so progress is visible
    # while the job runs (instead of one big bulk commit at the end).
    VISION_FLUSH_BATCH_SIZE: int = 10

    # Excel ingest strategy
    EXCEL_INGEST_MODE: Literal["markdown", "sql"] = "markdown"
    # markdown: sheet → markdown table → chunk như text thường
    # sql:      sheet → SQLite, query trực tiếp qua structured pipeline

    # Query Rewriting — HyDE + decomposition (requires one extra LLM call per query)
    QUERY_REWRITE_ENABLED: bool = False
    # HyDE: generate hypothetical answer and augment query for better kNN match
    QUERY_REWRITE_HYDE: bool = True
    # Decompose: split complex questions into sub-queries for multi-hop retrieval
    QUERY_REWRITE_DECOMPOSE: bool = False

    # Observability (ADR 0001 Phase B)
    OBSERVABILITY_TRACE_ENABLED: bool = True

    # ── Open-Notebook adapter ──────────────────────────────────────────────────
    OPEN_NOTEBOOK_PASSWORD: str | None = None   # legacy shared password (still accepted as bearer)
    ADAPTER_ADMIN_TOKEN: str | None = None      # admin token for reasoning view
    ADAPTER_VERSION: str = "0.7.0"              # version reported to open-notebook frontend

    # ── User auth (JWT) ────────────────────────────────────────────────────────
    AUTH_ENABLED: bool = True                   # require login if true (else open access)
    AUTH_ALLOW_SIGNUP: bool = True              # allow public signup
    JWT_SECRET: str | None = None               # signing secret; auto-derived in dev
    JWT_TTL_DAYS: int = 7

    # Google OAuth — leave blank to disable Google sign-in
    GOOGLE_CLIENT_ID: str | None = None
    GOOGLE_CLIENT_SECRET: str | None = None
    GOOGLE_REDIRECT_URI: str | None = None      # e.g. http://localhost:8000/on/api/auth/google/callback
    FRONTEND_URL: str = "http://localhost:3000"

    # ── Rate limit & dedupe ────────────────────────────────────────────────────
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MIN_DEFAULT: int = 120       # per-user per-min for chat/search
    RATE_LIMIT_UPLOAD_PER_MIN: int = 20         # per-user per-min for uploads
    UPLOAD_MAX_BYTES: int = 100 * 1024 * 1024   # 100 MB
    UPLOAD_DEDUPE_BY_HASH: bool = True

    # Worker concurrency (for uvicorn / gunicorn)
    UVICORN_WORKERS: int = 1

    @property
    def DATABASE_URL(self) -> str:
        password = quote_plus(self.POSTGRES_PASSWORD)
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{password}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()

# Apply persisted UI overrides (PUT /on/api/models/defaults) on top of .env.
from src.agentrag.config_overrides import apply_overrides  # noqa: E402

apply_overrides(settings)
