# Design: agentrag-api image diet (multi-stage, CUDA kept)

Date: 2026-07-05
Status: approved

## Goal

Shrink `licht693/agentrag-api` from 25.7GB uncompressed to ~9GB without
behavior change. CUDA torch stays (user decision: image remains GPU-ready
even though api/worker containers currently run torch on CPU).

## Why it is fat

- venv is 7.8GB (torch 1.7GB + nvidia CUDA libs 4.1GB — floor we keep).
- Single-stage Dockerfile runs `uv sync` twice (deps layer + project layer),
  duplicating the venv across layers.
- `UV_COMPILE_BYTECODE=1` adds ~1.5–2GB of .pyc.
- build-essential, libpq-dev/libjpeg-dev/zlib1g-dev headers, uv binary and
  caches all ship in the runtime image.

## Change: multi-stage `Dockerfile`

**builder** (`python:3.11-slim`):
- apt: `build-essential libpq-dev libjpeg-dev zlib1g-dev` (+curl/ca-certs for uv fetch if needed)
- uv from `ghcr.io/astral-sh/uv:0.5` as today
- `UV_COMPILE_BYTECODE=0`
- `COPY pyproject.toml uv.lock* ./` → `uv sync --frozen --no-install-project || uv sync --no-install-project`
- `COPY . .` → `uv sync --frozen || uv sync`

**runtime** (`python:3.11-slim`):
- apt runtime libs only: `curl ca-certificates libpq5 libjpeg62-turbo zlib1g`
  (curl needed by compose healthcheck; libpq5 for psycopg; jpeg/zlib for pillow)
- `COPY --from=builder /app /app` (source + `.venv` in one layer)
- Same env (`PYTHONUNBUFFERED`, `PATH=/app/.venv/bin:$PATH`), `EXPOSE 8000`,
  same gunicorn CMD. `uv` binary not present in runtime — CMD must not call
  `uv run`; invoke `gunicorn`/`arq` from the venv PATH directly. The compose
  `worker` service command (`uv run arq ...`) must change to `arq ...`.

## Non-goals

- No dependency moves out of `[project.dependencies]` (CUDA torch stays).
- No tesseract binary added (OCR fallback is dead in-container today —
  pre-existing, out of scope).
- No CPU-torch variant tag.

## Verification

- Local: `docker build -t agentrag-api:diet .` → `docker images` shows ≤ ~10GB.
- Smoke: `docker run --rm agentrag-api:diet python -c "import torch, sentence_transformers, faster_whisper; print(torch.__version__)"`.
- `docker run --rm agentrag-api:diet gunicorn --version` (venv PATH works, no uv).
- Full-stack health: `docker compose --profile app up -d --build api` →
  api healthcheck green (`curl :8000/on/health`), worker starts (arq command).
- CI: PR run green (build-only); post-merge master run pushes; pull + size
  check on the pushed tag.

## Risks

- First-boot import compile (no .pyc): a few extra seconds per fresh
  container — acceptable.
- `COPY --from=builder /app /app` keeps venv+source one layer: any source
  change re-pushes the whole ~8GB layer. Mitigation: copy `.venv` and source
  as two separate COPY steps (venv layer stable while code iterates).
  The plan should do that: `COPY --from=builder /app/.venv /app/.venv` then
  `COPY --from=builder /app /app` ordering issue — instead copy source from
  build context directly (`COPY . .`, dockerignore already tight) after the
  venv copy.
