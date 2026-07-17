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
    #: DeepSeek (OpenAI-compatible). Routed per-task via a "deepseek-*" model name
    #: in LLM_TASK_MODEL_MAP. Falls back to OPENAI_API_KEY if unset.
    DEEPSEEK_API_KEY: str | None = None
    #: Anthropic / Claude — routed per-task via a "claude-*" model name in
    #: LLM_TASK_MODEL_MAP, through Anthropic's OpenAI-compatible endpoint
    #: (https://api.anthropic.com/v1/). Primary use: an INDEPENDENT cross-provider
    #: eval judge (eval_judge/eval_judge2) so the score isn't self-graded by the
    #: answer model's provider. Judge-only path — no adaptive-thinking on the compat API.
    ANTHROPIC_API_KEY: str | None = None

    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434/v1/"
    OLLAMA_API_KEY: str = "ollama"

    #: sync: StructMem chạy ngay trong ingest. async: đưa vào hàng đợi (cần worker, xem main.py lifespan).
    STRUCTMEM_INGEST_MODE: Literal["sync", "async"] = "async"

    EMBEDDING_PROVIDER: Literal["openai", "gemini", "hf_inference", "ollama"] = "hf_inference"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large-instruct"
    EMBEDDING_BASE_URL: str | None = None
    EMBEDDING_BATCH_SIZE: int = 32
    # Output vector dim — only honored by providers that support truncation
    # (e.g. Gemini gemini-embedding-001 supports 768 / 1536 / 3072 via
    # Matryoshka Representation Learning). Leave None to use model default.
    EMBEDDING_OUTPUT_DIM: int | None = None

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
    # S5 — domain tagging during ingest (SectionTagger). Adds system_tag /
    # specialty_tag / canonical_terms to every chunk.
    TAGGING_ENABLED: bool = True
    # S5 — federated retrieval (DomainRouter + FederatedRetriever).
    DOMAIN_FILTER_ENABLED: bool = True
    DOMAIN_ROUTER_CONFIDENCE_THRESHOLD: float = 0.7
    DOMAIN_ROUTER_TOP_K: int = 3
    # Unified doc memory index (kind ∈ {"entry","synthesis"}). Was previously
    # agentrag_entries + agentrag_synthesis; collapsed in R4 to halve the
    # mapping surface. Legacy *_INDEX_NAME settings retained below as aliases
    # so external configs keep loading; both resolve to the unified index.
    STRUCTMEM_INDEX: str = "agentrag_memory_doc"
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
    # local_cross_encoder: bge-reranker-v2-m3 on CPU/GPU (free, no API). Default.
    # llm_chat: rank via OpenAI-compat chat (slow, any provider).
    RETRIEVAL_RERANK_BACKEND: Literal["llm_chat", "local_cross_encoder"] = "local_cross_encoder"
    RETRIEVAL_RERANK_PROVIDER: Literal["openai", "gemini", "hf_inference", "ollama"] | None = None
    RETRIEVAL_RERANK_MODEL: str | None = None
    RETRIEVAL_RERANK_BASE_URL: str | None = None
    RETRIEVAL_RERANK_TEMPERATURE: float = 0.0
    #: Relevance-floor gate — drop retrieved candidates whose reranker relevance
    #: (sigmoid of the bge cross-encoder score, 0–1) is below the floor BEFORE the
    #: answer sees them. Prunes distractors; an out-of-corpus query whose top hit
    #: is below floor → empty context → clean abstain (no distractor citations).
    #: Only effective with RETRIEVAL_RERANK_BACKEND=local_cross_encoder (the only
    #: backend that emits a score). Default OFF (changes answer behavior).
    RETRIEVAL_RELEVANCE_GATE_ENABLED: bool = False
    #: bge-reranker-v2-m3 (sigmoid) scores off-corpus content ~0.50 (logit≈0). 2026-06-19
    #: calibration put relevant top-chunks ~0.73 and set the floor at 0.6; the 2026-06-26
    #: prod-corpus probe (docs/eval/eval_fidelity_probe_prod_2026-06-26.md) found that
    #: PARAPHRASED relevant VN chunks score as low as ~0.61 and jitter under 0.6 across the
    #: agent's query-rewrites → flaky FALSE-abstention with the gold chunk at rank 0. Lowered
    #: to 0.55 — mid-band between OOC ~0.50 and low-relevant ~0.61, giving ~0.05 margin both
    #: sides (this realises the "margin/anti-knife-edge" intent without a separate constant
    #: that would double-loosen). Re-measure + re-validate OOC abstention per corpus.
    RETRIEVAL_RELEVANCE_FLOOR: float = 0.55
    #: Always merge a plain-hybrid retrieval on the RAW question into the rerank
    #: candidate pool (ContextAssembler). The agent's decide-step rewrites/sub-queries
    #: (hybrid_kg + variants) can retrieve worse chunks than the raw question and drop the
    #: packed-context max rerank below the floor → flaky false-abstain (2026-06-26 row21:
    #: raw hybrid = 0.716, agent pool topped at <floor). Injecting the raw hits guarantees
    #: rewrites only ADD, never degrade. Doesn't loosen OOC (raw OOC query still ~0.50).
    RETRIEVAL_INCLUDE_RAW_QUERY: bool = True
    #: 2026-07-16: 8→50. Gold ranks deep (fusion rank 8–17) for factual queries;
    #: the unfiltered raw-query backstop must reach it. Safe now that the pool is
    #: reranked BEFORE the trim (context.assemble), so extra recall doesn't flood
    #: the packed answer context — low-rerank distractors are trimmed.
    RETRIEVAL_RAW_QUERY_TOP_K: int = 50
    #: When best rerank score is in [floor, floor+GRAY_MARGIN), treat the
    #: context as uncertain and force the strong-abstain prompt (out-of-corpus
    #: distractors that score just over the floor → confident-hallucination fix).
    ANSWERABILITY_GRAY_MARGIN: float = 0.13
    #: Master switch for the gray-band abstain. Default OFF — enable after the
    #: refusal-set re-eval beats baseline (docs/eval/benchmark_abstain_ab_*).
    ANSWERABILITY_GATE_ENABLED: bool = False
    #: Abstain-on-thin-context (answer-layer, the corrected version of the
    #: relevance gate). When the best retrieved chunk's rerank relevance is below
    #: RETRIEVAL_RELEVANCE_FLOOR, KEEP the context but instruct the model to abstain
    #: ("no relevant info; don't answer from background knowledge; don't cite") AND
    #: drop distractor citations from the resulting abstention. ON since 2026-06-19 A/B
    #: at floor 0.6: out-of-corpus refusal_rate 0→0.467, hedged_cited 0.533→0, in-corpus flat.
    ANSWER_ABSTAIN_ON_THIN_CONTEXT: bool = True

    AGENT_MAX_STEPS: int = 3   # serial decide→tool round-trips; lower = faster p50
    #: Per-request timeout (seconds) for every AsyncOpenAI client (make_async_openai).
    #: Guards against a stalled provider connection hanging a call indefinitely — a
    #: 42-min agent.chat hang was observed under gemini high-demand (2026-06-26). On
    #: timeout the openai SDK retries rather than blocking forever.
    LLM_REQUEST_TIMEOUT_S: float = 60.0
    #: Total wall-clock budget (seconds) for one agent.chat graph run. The per-call
    #: LLM_REQUEST_TIMEOUT_S does NOT bound the whole multi-step/multi-rewrite loop
    #: (decide + N tool searches + rerank + answer, each retryable) — under a gemini
    #: 503 storm the total can exceed 120s (2026-06-26). On exceeding this budget the
    #: chat returns a graceful "busy, retry" response (timed_out=True) instead of hanging.
    AGENT_TOTAL_TIMEOUT_S: float = 90.0
    AGENT_TOOL_TOP_K: int = 30  # 2026-07-16: 5→30, deep gold needs a wide pool (rerank-before-trim keeps precision)
    AGENT_MAX_CONTEXT_CHUNKS: int = 8
    CHAT_HISTORY_WINDOW: int = 10
    CHAT_REDIS_TTL_SECONDS: int = 300

    # Chat StructMem — áp dụng dual-perspective memory cho lịch sử hội thoại.
    # Single index, rows discriminated by `kind` ∈ {"entry","synthesis"}.
    CHAT_STRUCTMEM_ENABLED: bool = True
    CHAT_MEMORY_INDEX: str = "agentrag_memory_chat"
    CHAT_MEMORY_CONSOLIDATION_THRESHOLD: int = 10  # số turns tích luỹ trước khi consolidate
    CHAT_MEMORY_TOP_K: int = 8  # số entries retrieve mỗi lượt

    # LLM Routing (ADR 0001 Phase C)
    LLM_ROUTING_ENABLED: bool = False
    LLM_TASK_MODEL_MAP: str = "{}"
    LLM_COST_TRACKING_ENABLED: bool = False

    # Vision LLM — for describing images extracted from PDFs and standalone image files.
    # If VISION_PROVIDER is None, image extraction is skipped (text-only ingestion).
    VISION_PROVIDER: Literal["openai", "gemini", "ollama"] | None = None
    VISION_MODEL: str | None = None          # e.g. gpt-4o | gemini-1.5-flash | llava:13b
    VISION_BASE_URL: str | None = None
    IMAGE_STORAGE_DIR: str = "data/images"
    # Persist original uploaded bytes (PDF/DOCX/XLSX/MD) so users can re-download
    # or open in a native viewer (browser PDF.js, Word, Excel). One file per
    # document_id, named `<doc_id><ext>`. Empty/None → originals discarded after
    # ingest (old behavior).
    ORIGINALS_DIR: str = "data/originals"
    IMAGE_MIN_SIZE_BYTES: int = 5000         # skip icons / decorative bullets
    VISION_TIMEOUT_SECONDS: int = 180        # bump cao cho llava cold-start
    # sync: describe images inline during ingest (slow, blocks pipeline).
    # async: skip describe; queue ARQ vision_extract job → text retrieval ready ngay.
    VISION_INGEST_MODE: Literal["sync", "async"] = "async"

    # Auto-route to large-context model khi prompt > threshold tokens.
    # Inspired by open-notebook provision_langchain_model.
    # None = disabled (use default model regardless of size).
    LLM_LARGE_CONTEXT_MODEL: str | None = None         # ví dụ gemini-2.5-pro, qwen2.5:32b
    LLM_LARGE_CONTEXT_THRESHOLD: int = 100_000         # tokens
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
    # Group N images per vision LLM call to cut RPM cost ~N×. Set to 1 to
    # disable. Default 4 fits free-tier Gemini 12 RPM with doc-many-figures.
    VISION_DESCRIBE_BATCH: int = 4

    # ── Audio transcription (Whisper) ─────────────────────────────────────────
    # Provider: faster_whisper (local, free) | openai (cloud, paid).
    AUDIO_TRANSCRIBE_PROVIDER: Literal["faster_whisper", "openai"] = "faster_whisper"
    # faster-whisper model size: tiny | base | small | medium | large-v3.
    # `small` ~150MB ~5x realtime on GPU; `medium` ~1.5GB ~2x realtime; `large-v3` best quality.
    AUDIO_WHISPER_MODEL: str = "small"
    AUDIO_WHISPER_DEVICE: str = "auto"          # auto | cpu | cuda
    AUDIO_WHISPER_COMPUTE_TYPE: str = "auto"    # auto | int8 | float16 | int8_float16
    AUDIO_WHISPER_LANGUAGE: str | None = None   # None = auto-detect; "vi" for VN
    AUDIO_WHISPER_BEAM_SIZE: int = 5
    AUDIO_OPENAI_MODEL: str = "whisper-1"

    # ── Visual (CLIP) embedding ──────────────────────────────────────────────
    # Cross-modal text↔image embedding for visual retrieval. Default
    # `sentence-transformers/clip-ViT-B-32-multilingual-v1` (512-dim,
    # VN-capable). Lazy-loaded on first image ingest.
    VISUAL_EMBEDDING_ENABLED: bool = True
    VISUAL_EMBEDDING_MODEL: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    VISUAL_EMBEDDING_DIMS: int = 512
    VISUAL_EMBEDDING_DEVICE: str = "auto"  # auto | cpu | cuda

    # Token-aware context budget. When >0, replaces AGENT_MAX_CONTEXT_CHUNKS
    # as the trim signal: keep adding ranked chunks until total tokens exceed
    # this budget. Keeps chunk count flexible (short chunks → more, long → fewer).
    AGENT_MAX_CONTEXT_TOKENS: int = 6000
    # S8 — per-call output ceiling. 128K = let model self-cap. Lower
    # (e.g. 8192) for fast local Ollama on small models.
    AGENT_MAX_OUTPUT_TOKENS: int = 131072
    # S10 — cap image-segment fraction in top_k. Vision-described images
    # often dominate dense retrieval on scanned PDFs (their English-rich
    # descriptions match better than original Vietnamese text). Force a
    # mix by demoting excess images when text candidates exist.
    RETRIEVAL_MAX_IMAGE_RATIO: float = 0.3
    # PDF OCR fallback (scanned / image-only PDFs).
    # When PyMuPDF.get_text returns < PDF_OCR_MIN_TEXT_CHARS, render the
    # page → Tesseract (lang=PDF_OCR_LANG). If Tesseract output is still
    # too short and PDF_OCR_VISION_FALLBACK is on, send the page image
    # through VISION_PROVIDER as a final fallback.
    PDF_OCR_FALLBACK_ENABLED: bool = True
    PDF_OCR_LANG: str = "vie+eng"
    PDF_OCR_MIN_TEXT_CHARS: int = 50
    PDF_OCR_DPI: int = 300
    PDF_OCR_VISION_FALLBACK: bool = True
    PDF_OCR_VISION_THRESHOLD: int = 30
    # PDF parser backend escalation when PyMuPDF text-layer is thin:
    #   hybrid (default) → Tesseract → vision LLM
    PDF_PARSER_BACKEND: str = "hybrid"
    # Lost-in-the-middle: reorder packed chunks so the top-ranked entries sit
    # at the start AND end of the prompt, weaker middle. Set false for plain
    # rank-order packing.
    AGENT_LOST_IN_MIDDLE_REORDER: bool = True

    # Plan-then-execute: before retrieval, decompose complex / multi-hop
    # questions into a list of sub-queries, retrieve evidence for each in
    # parallel, then answer once with consolidated context. Trades 1 extra
    # planner-LLM call for fewer reactive decide-loop iterations. Skip when
    # the question is short / single-clause (no benefit).
    AGENT_PLAN_THEN_EXECUTE_ENABLED: bool = True
    # Minimum char length before plan-then-execute kicks in. Below this, plain
    # reactive loop is used (greetings, lookups don't need a planner pass).
    AGENT_PLAN_TRIGGER_MIN_CHARS: int = 60
    # Max sub-queries per plan. Cap to avoid runaway fan-out under bad planner.
    AGENT_PLAN_MAX_SUBQUERIES: int = 4

    # Excel ingest strategy
    EXCEL_INGEST_MODE: Literal["markdown", "sql"] = "markdown"
    # markdown: sheet → markdown table → chunk như text thường
    # sql:      sheet → SQLite, query trực tiếp qua structured pipeline

    # Query Rewriting — HyDE + decomposition (requires one extra LLM call per query)
    QUERY_REWRITE_ENABLED: bool = True   # 2026-07-17: HyDE A/B on clean_v2 — on 0.904 vs off 0.864 (−0.039, 2 regressions); ship on (matches validated eval; OOC 15/15 safe)
    # HyDE: generate hypothetical answer and augment query for better kNN match
    QUERY_REWRITE_HYDE: bool = True
    # Decompose: split complex questions into sub-queries for multi-hop retrieval
    QUERY_REWRITE_DECOMPOSE: bool = False

    # ── WS1: Contextual Retrieval (Anthropic) ────────────────────────────────
    #: When on, each chunk gets a 50–100 token LLM-generated context prepended
    #: to the text that is embedded + BM25-indexed (original `content` still cited).
    CONTEXTUAL_RETRIEVAL_ENABLED: bool = False
    #: LLMGateway task name → routable to DeepSeek via LLM_TASK_MODEL_MAP.
    CONTEXTUAL_RETRIEVAL_TASK: str = "contextualize"
    #: Cap the document text sent as the cached situating prefix (chars).
    CONTEXTUAL_MAX_DOC_CHARS: int = 48000
    CONTEXTUAL_CACHE_DIR: str = ".cache/agentrag/context"

    # ── WS2: RAPTOR summary layer ────────────────────────────────────────────
    RAPTOR_ENABLED: bool = False
    RAPTOR_MAX_LEVELS: int = 3
    #: Documents with fewer leaf chunks than this skip RAPTOR (no value).
    RAPTOR_MIN_LEAVES: int = 8
    #: Target average cluster size when picking the number of GMM components.
    RAPTOR_CLUSTER_SIZE: int = 5
    RAPTOR_SUMMARY_TASK: str = "raptor_summary"
    #: Max share of a result set that may be RAPTOR summary nodes.
    RAPTOR_SUMMARY_MAX_RATIO: float = 0.4

    # ── WS3: CRAG critique + multi-hop ───────────────────────────────────────
    CRAG_ENABLED: bool = False
    #: Min retrieved-hit count below which retrieval is judged "incorrect".
    CRAG_MIN_HITS: int = 1
    CRAG_GROUNDING_ENABLED: bool = True
    AGENT_CRITIQUE_MAX_RETRIES: int = 1
    AGENT_MULTIHOP_ENABLED: bool = False

    # ── WS5: Semantic retrieval cache ────────────────────────────────────────
    SEMANTIC_CACHE_ENABLED: bool = False
    SEMANTIC_CACHE_THRESHOLD: float = 0.97
    SEMANTIC_CACHE_TTL_SECONDS: int = 120
    SEMANTIC_CACHE_MAX_ITEMS: int = 256

    # Observability (ADR 0001 Phase B)
    OBSERVABILITY_TRACE_ENABLED: bool = True

    # ── Langfuse LLM tracing ───────────────────────────────────────────────────
    # When enabled, every LLM call (agent / vision / reranker) is traced via the
    # langfuse.openai drop-in wrapper. Keys come from the self-hosted (or cloud)
    # Langfuse project. Cost ledger (observability/cost.py) stays independent.
    LANGFUSE_ENABLED: bool = False
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_HOST: str = "http://localhost:3000"   # self-host default (docker-compose profile=observability)
    #: Privacy gate: when False (default) the question/answer/comment TEXT is NOT sent to
    #: the trace store — only latency/token/structure/session. Medical PHI must not leave
    #: the app unless this is explicitly enabled (dev/debug on non-PHI data).
    OBSERVABILITY_CAPTURE_CONTENT: bool = False

    # ── Arize Phoenix tracing (alongside Langfuse) ─────────────────────────────
    # Local OTEL trace + eval UI. When enabled, OpenInference instruments the
    # OpenAI client so every LLM call is traced to the Phoenix collector.
    PHOENIX_ENABLED: bool = False
    PHOENIX_COLLECTOR_ENDPOINT: str = "http://localhost:6006/v1/traces"
    PHOENIX_PROJECT: str = "agentrag"
    # NOTE: RAGAS judge/embedding config lives in scripts/eval/score_ragas.py
    # (CLI flags + env keys), not here — RAGAS runs in an isolated venv that
    # cannot import this settings module. See scripts/eval/run_ragas.py.

    # ── Open-Notebook adapter ──────────────────────────────────────────────────
    OPEN_NOTEBOOK_PASSWORD: str | None = None   # legacy shared password (still accepted as bearer)
    ADAPTER_ADMIN_TOKEN: str | None = None      # admin token for reasoning view
    ADAPTER_VERSION: str = "0.7.0"              # version reported to open-notebook frontend

    # ── User auth (JWT) ────────────────────────────────────────────────────────
    AUTH_ENABLED: bool = True                   # require login if true (else open access)
    AUTH_ALLOW_SIGNUP: bool = True              # allow public signup
    ADMIN_EMAILS: str = ""                      # comma-separated; matching emails auto-promoted on signup/login
    # Fallback model when AGENT_MODEL / EXTRACTION_MODEL returns "model not
    # found" at runtime (e.g. qwen-agentrag absent because finetune never
    # ran). Set blank to disable. Default = qwen2.5:7b-instruct, the base
    # the project finetunes from.
    LLM_FALLBACK_MODEL: str = "qwen2.5:7b-instruct"
    # Ollama num_ctx override (default 8192 truncates long prompts silently).
    # 32768 = 32k tokens, fits qwen2.5 7B Q4 within 16GB VRAM.
    LLM_OLLAMA_NUM_CTX: int = 32768
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
