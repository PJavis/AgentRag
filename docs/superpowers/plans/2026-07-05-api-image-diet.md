# API Image Diet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shrink `licht693/agentrag-api` 25.7GB → ~9GB via multi-stage Dockerfile; zero behavior change, CUDA torch kept.

**Architecture:** Builder stage owns compilers/uv/caches and produces `.venv`; slim runtime stage gets the venv (own layer) + source from build context + runtime libs only. No `uv` binary in runtime → CMD and compose worker command invoke venv binaries directly.

**Tech Stack:** Docker multi-stage, uv, python:3.11-slim, docker compose.

**Spec:** `docs/superpowers/specs/2026-07-05-api-image-diet-design.md`

## Global Constraints

- CUDA torch STAYS — no dependency changes in `pyproject.toml`, no CPU-torch pinning.
- Runtime image must NOT contain: build-essential, `-dev` headers, the `uv` binary, uv/pip caches, `.pyc` bytecode (`UV_COMPILE_BYTECODE=0`).
- Runtime apt libs exactly: `curl ca-certificates libpq5 libjpeg62-turbo zlib1g` (curl = compose healthcheck dependency).
- `.venv` copied from builder as its OWN layer, source copied from build context AFTER it (code changes must not re-push the venv layer).
- Same runtime behavior: gunicorn on :8000 with UvicornWorker, 4 workers, timeout 120; worker runs arq on `src.agentrag.worker.settings.WorkerSettings`.
- No tesseract binary added (pre-existing gap, out of scope).
- Target: final image ≤ ~10GB uncompressed.

---

### Task 1: Multi-stage Dockerfile + compose worker command

**Files:**
- Modify: `Dockerfile` (full rewrite, currently 27 lines, single-stage)
- Modify: `docker-compose.yml:198` (worker `command`)

**Interfaces:**
- Consumes: existing `.dockerignore` (context is 6.6M, already hardened).
- Produces: image where `PATH=/app/.venv/bin:$PATH` and no `uv` exists — anything exec'ing into api/worker containers must call venv binaries directly. Task 2 verifies.

- [ ] **Step 1: Rewrite `Dockerfile`**

Replace the entire file content with:

```dockerfile
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
```

- [ ] **Step 2: Update worker command in `docker-compose.yml`**

Line 198 (worker service), change:

```yaml
    command: ["uv", "run", "arq", "src.agentrag.worker.settings.WorkerSettings"]
```

to:

```yaml
    command: ["arq", "src.agentrag.worker.settings.WorkerSettings"]
```

(`arq` resolves via the image's `PATH=/app/.venv/bin`; the runtime stage has no `uv`.)

- [ ] **Step 3: Grep for other in-container `uv run` callers**

```bash
grep -rn "uv run" docker-compose.yml docker-compose.fullstack.yml deploy/ .github/
```

Expected: no remaining hits that execute INSIDE the api/worker image (Makefile/docs run on host — out of scope; do not change them).

- [ ] **Step 4: Build and measure**

```bash
docker build -t agentrag-api:diet . 2>&1 | tail -3
docker images agentrag-api:diet --format "{{.Size}}"
```

Expected: build succeeds; size ≤ ~10GB (was 25.7GB). Build takes ~10min on cache-cold torch download.

- [ ] **Step 5: Import + entrypoint smoke (no stack needed)**

```bash
docker run --rm agentrag-api:diet python -c "import torch, sentence_transformers, faster_whisper, umap; print('imports OK', torch.__version__)"
docker run --rm agentrag-api:diet gunicorn --version
docker run --rm agentrag-api:diet arq --help >/dev/null && echo "arq OK"
docker run --rm agentrag-api:diet sh -c "command -v uv || echo 'no uv (expected)'; ls /app/.venv/bin/python >/dev/null && echo 'venv OK'"
docker run --rm agentrag-api:diet sh -c "find /app/.venv -name '*.pyc' | head -1 | grep -q . && echo 'PYC PRESENT (fail)' || echo 'no pyc OK'"
```

Expected: imports OK + torch version; gunicorn version prints; `arq OK`; `no uv (expected)` + `venv OK`; `no pyc OK`.

- [ ] **Step 6: Full-stack health smoke**

```bash
docker compose --profile app up -d --build api worker 2>&1 | tail -2
sleep 45 && curl -sf http://127.0.0.1:8000/on/health && echo " api HEALTHY"
docker logs agentrag-worker 2>&1 | tail -3
```

Expected: api returns health 200; worker log shows arq started (no `uv: not found`). Requires the stack's postgres/es/valkey healthy (they are on this box). If TEI/ollama not up, api may still be healthy — health endpoint doesn't call them.

- [ ] **Step 7: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "build: multi-stage Dockerfile — api image 25.7GB -> ~9GB

Builder stage keeps compilers/uv/caches; runtime = slim + venv layer +
source. Bytecode compilation off. Worker compose command calls arq
directly (no uv in runtime image). CUDA torch kept (GPU-ready image)."
```

---

### Task 2: CI + registry verification

**Files:** none (operations)

**Interfaces:**
- Consumes: Task 1's image; existing `docker-publish.yml` workflow (unchanged).

- [ ] **Step 1: Push branch, open PR**

```bash
git push
gh pr create --base master --title "build: api image diet — multi-stage Dockerfile (25.7GB -> ~9GB)" --body "$(cat <<'EOF'
Multi-stage Dockerfile per docs/superpowers/specs/2026-07-05-api-image-diet-design.md. Builder owns uv/compilers/caches; runtime = slim + venv + source. No pyc, no uv binary in runtime; worker compose command calls arq directly. CUDA torch kept. Local verified: image ~9GB, imports/gunicorn/arq/health all green.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Watch PR build-only run**

```bash
RUN_ID=$(gh run list --workflow docker-publish -R PJavis/AgentRag --limit 1 --json databaseId -q '.[0].databaseId')
gh run watch --exit-status $RUN_ID -R PJavis/AgentRag && echo CI_GREEN
```

Expected: both matrix jobs green (frontend unaffected, cache hit).

- [ ] **Step 3: USER GATE — merge PR** (agent cannot merge; ask the user)

- [ ] **Step 4: Watch master push run; verify pushed size**

```bash
RUN_ID=$(gh run list --workflow docker-publish -R PJavis/AgentRag --branch master --limit 1 --json databaseId -q '.[0].databaseId')
gh run watch --exit-status $RUN_ID -R PJavis/AgentRag && echo PUSH_GREEN
docker manifest inspect licht693/agentrag-api:latest | grep -c '"size"' && \
docker manifest inspect -v licht693/agentrag-api:latest 2>/dev/null | uv run python -c "import json,sys; d=json.load(sys.stdin); layers=d['SchemaV2Manifest']['layers'] if 'SchemaV2Manifest' in d else d.get('OCIManifest',{}).get('layers',[]); print('compressed total MB:', sum(l['size'] for l in layers)//1048576)"
```

Expected: PUSH_GREEN; compressed total ~3500-4500 MB (was ~8-10GB compressed).

- [ ] **Step 5: Local re-pull sanity + cleanup**

```bash
docker pull licht693/agentrag-api:latest 2>&1 | tail -1
docker images licht693/agentrag-api:latest --format "{{.Size}}"
docker image prune -f | tail -1
```

Expected: pulled size ≤ ~10GB. (Keep or remove the local tag per disk needs.)
