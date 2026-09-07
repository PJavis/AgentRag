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
#
# tesseract is NOT optional: PDFParser sends any page whose text layer is thin
# to OCR, and ~25% of the medical corpus is scanned. Without the binary the OCR
# call raises, the page yields nothing, chunks under 80 chars are filtered, and
# the page disappears from the corpus — an ingest that reports success while
# silently dropping a quarter of the pages. Measured 2026-09-07: a 20-page
# scanned document produced 1 segment in-container versus 29 on a host that had
# tesseract. vie+eng matches PDF_OCR_LANG.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libpq5 libjpeg62-turbo zlib1g \
    tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# venv first (stable ~8GB layer), source after (small, changes often)
COPY --from=builder /app/.venv /app/.venv
COPY . .

# --- Build provenance -------------------------------------------------------
# Every result set must carry the identity of the code that produced it.
# docker-compose.deploy.yml and docker-compose.fullstack.yml are both invoked
# with -f, which disables auto-merge of docker-compose.override.yml and so of
# the optional ./src bind mount. A stale image on those paths does not error —
# it silently produces plausible results from the wrong code.
ARG GIT_SHA=unknown
ARG BUILD_ID=unknown
ARG BUILT_AT=unknown
ENV AGENTRAG_GIT_SHA=$GIT_SHA \
    AGENTRAG_BUILD_ID=$BUILD_ID \
    AGENTRAG_BUILT_AT=$BUILT_AT
LABEL org.opencontainers.image.revision=$GIT_SHA \
      org.opencontainers.image.version=$BUILD_ID \
      org.opencontainers.image.created=$BUILT_AT

# Hash of the source this image ships, written OUTSIDE /app/src so the ./src
# bind mount cannot overwrite it. At startup the app re-hashes what it is
# actually importing and compares: a mismatch means a mount is shadowing the
# image, or the image is stale. Needs no build args, so a plain
# `docker compose up --build` is covered too.
RUN python -c "import sys; sys.path.insert(0, '/app'); \
from src.agentrag.common.build_info import source_sha; \
open('/app/.build-source-sha', 'w').write(source_sha())"

EXPOSE 8000
# --timeout 600, not 120: /ingest/folder parses a whole corpus in-request, and
# PDF parsing is CPU-bound, so it blocks the event loop long enough for gunicorn
# to kill the worker mid-ingest. Observed 2026-09-07 as
# "WORKER TIMEOUT (pid:10)" after 23 minutes, leaving the corpus half-migrated.
# 600 covers a large document; a whole-corpus ingest should still be run
# out-of-band (see docs/eval/table_arm_b_rollout_2026-09-06.md) rather than
# over HTTP.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "4", "-b", "0.0.0.0:8000", "main:app", "--timeout", "600"]
