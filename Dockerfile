# syntax=docker/dockerfile:1.7
# AgentRag API + worker image — multi-stage: builder makes the venv with uv,
# runtime is python-slim + venv + source. CUDA torch kept deliberately
# (GPU-ready image; api/worker containers currently run torch on CPU).

FROM python:3.11-slim AS builder
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy UV_COMPILE_BYTECODE=0

# Build deps for psycopg (libpq), pillow (jpeg/zlib) source builds if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project || uv sync --no-install-project

COPY . .
RUN uv sync --frozen || uv sync

FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Runtime libs only: curl for compose healthchecks, libpq5 for psycopg,
# jpeg/zlib for pillow. No compilers, no headers, no uv.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libpq5 libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# venv first (stable ~8GB layer), source after (small, changes often)
COPY --from=builder /app/.venv /app/.venv
COPY . .

EXPOSE 8000
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "4", "-b", "0.0.0.0:8000", "main:app", "--timeout", "120"]
