# DockerHub CI Publish + HF Hub Model Hosting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CI-built app images on DockerHub and the finetuned embedding model on HF Hub, so any machine deploys with `docker compose pull` — no local builds, no manual model copying.

**Architecture:** A single GitHub Actions workflow matrix-builds the two app images (`agentrag-api` from root `Dockerfile`, `agentrag-frontend` from `frontend/Dockerfile`) and pushes to DockerHub on master/tag pushes. Compose services gain `image:` names so `pull` works. TEI serves the finetuned model directly from HF Hub instead of a local bind mount.

**Tech Stack:** GitHub Actions (`docker/build-push-action`, `docker/metadata-action`), DockerHub, HuggingFace Hub (`hf` CLI), docker compose.

**Spec:** `docs/superpowers/specs/2026-07-05-dockerhub-hf-publish-design.md`

## Global Constraints

- DockerHub namespace: `pjavis` → images `pjavis/agentrag-api`, `pjavis/agentrag-frontend`, both public.
- HF model repo: `pjavis/agentrag-embed-v1`, public.
- Platforms: `linux/amd64` only.
- CI pushes only on `master` pushes and `v*` tags; PRs build without pushing.
- Images must never contain `.env` (both `.dockerignore` files already exclude it — do not remove those entries).
- Default branch is `master`; current work happens on `feat/ragas-langfuse-reranker`.

---

### Task 1: Harden root `.dockerignore`

Local builds currently bake `models/` (3.3GB) and `checkpoints/` into the api image. CI checkouts don't have them (gitignored), but local builds must match CI output.

**Files:**
- Modify: `.dockerignore` (repo root)

**Interfaces:**
- Produces: a build context that contains no model weights; later tasks assume the api image is code-only.

- [ ] **Step 1: Measure current build context (failing test)**

```bash
cd /home/nguyenquocdung/work/AgentRag
printf 'FROM busybox\nCOPY . /ctx\nRUN du -sh /ctx\n' | docker build --no-cache --progress=plain -f- . 2>&1 | grep -E '^\#[0-9]+ [0-9.]+ .*G|/ctx'
```

Expected: `du` output shows several GB (models/ included) — confirms the leak.

- [ ] **Step 2: Add exclusions**

Append to `.dockerignore` (keep every existing line):

```
models/
checkpoints/
docs/
frontend/
*.docx
image.png
```

- [ ] **Step 3: Re-measure context (test passes)**

```bash
printf 'FROM busybox\nCOPY . /ctx\nRUN du -sh /ctx\n' | docker build --no-cache --progress=plain -f- . 2>&1 | grep '/ctx'
```

Expected: total well under 100MB, no GB-scale entries.

- [ ] **Step 4: Commit**

```bash
git add .dockerignore
git commit -m "build: exclude models/checkpoints/docs from docker build context"
```

---

### Task 2: GitHub Actions publish workflow

**Files:**
- Create: `.github/workflows/docker-publish.yml`

**Interfaces:**
- Consumes: repo secrets `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` (Task 6 sets them up — workflow merges safely before secrets exist because PR events never push).
- Produces: DockerHub tags `latest` (master), `sha-<7char>`, `X.Y.Z` (from `v*` tags) for both images. Task 3 references `pjavis/agentrag-api:latest` and `pjavis/agentrag-frontend:latest`.

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/docker-publish.yml`:

```yaml
name: docker-publish

on:
  push:
    branches: [master]
    tags: ["v*"]
  pull_request:
    branches: [master]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        include:
          - image: agentrag-api
            context: .
          - image: agentrag-frontend
            context: ./frontend
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Docker Hub
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Docker metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: pjavis/${{ matrix.image }}
          tags: |
            type=raw,value=latest,enable={{is_default_branch}}
            type=sha,prefix=sha-
            type=semver,pattern={{version}}

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: ${{ matrix.context }}
          platforms: linux/amd64
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha,scope=${{ matrix.image }}
          cache-to: type=gha,mode=max,scope=${{ matrix.image }}
```

- [ ] **Step 2: Validate YAML parses**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/docker-publish.yml')); print('OK')"
```

Expected: `OK`. (If `yaml` missing, use `python3` with pyyaml from anywhere — parse check only.)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/docker-publish.yml
git commit -m "ci: build and push api+frontend images to DockerHub"
```

Real execution test happens in Task 6 (PR triggers build-only run).

---

### Task 3: Compose services get registry image names

**Files:**
- Modify: `docker-compose.yml` (services `api` ~line 138, `worker` ~line 176, `frontend` ~line 200)

**Interfaces:**
- Consumes: image names published by Task 2.
- Produces: `docker compose pull` works on machines with no build context.

- [ ] **Step 1: Add `image:` to the three services**

In `docker-compose.yml`, directly under each service name, add the `image:` line while keeping the existing `build:` block:

```yaml
  api:
    image: pjavis/agentrag-api:latest
    build:
      context: .
```

```yaml
  worker:
    image: pjavis/agentrag-api:latest
    build:
      context: .
```

```yaml
  frontend:
    image: pjavis/agentrag-frontend:latest
    build:
      context: ./frontend
```

(`worker` reuses the api image — same Dockerfile, different `command`. With `image:` + `build:` together, compose tags local builds with the registry name and pulls when not building.)

- [ ] **Step 2: Validate compose config**

```bash
docker compose --profile app config 2>/dev/null | grep -E 'image: pjavis'
```

Expected output (3 lines):

```
    image: pjavis/agentrag-api:latest
    image: pjavis/agentrag-api:latest
    image: pjavis/agentrag-frontend:latest
```

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "build(compose): name app images for DockerHub pull-based deploys"
```

---

### Task 4: TEI serves models from HF Hub

**Files:**
- Modify: `deploy/tei.compose.yml`

**Interfaces:**
- Consumes: HF repo `pjavis/agentrag-embed-v1` (uploaded in Task 5 — until then GPU profile falls back to the documented local-mount variant).
- Produces: TEI that self-downloads models; no `../models` needed on deploy boxes.

- [ ] **Step 1: Rewrite `deploy/tei.compose.yml`**

Replace the full file content with:

```yaml
# Text Embeddings Inference (TEI) — OpenAI-compatible embedding server.
# Serves the finetuned embedding model so AgentRag can hit it via
# EMBEDDING_PROVIDER=openai + EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1/.
#
# Models are pulled from HF Hub on first start into the tei_hf_cache volume
# (one-time ~1.1GB download; survives restarts). For offline/local work,
# swap the volume for a bind mount and use --model-id=/data:
#   - ../models/agentrag-embed-v1:/data   (+ HF_HUB_OFFLINE=1)
#
# Start (GPU, 16 GB box):
#   docker compose -f deploy/tei.compose.yml --profile gpu up -d
#
# Start (CPU fallback, no GPU):
#   docker compose -f deploy/tei.compose.yml --profile cpu up -d
#
# Stop:
#   docker compose -f deploy/tei.compose.yml down

services:
  tei-gpu:
    profiles: ["gpu"]
    # cuda-latest = all-arch PTX build; required for Blackwell sm_120 (RTX 50xx).
    # The pinned :1.5 / :1.7 images are sm_80 only and fail on this GPU.
    image: ghcr.io/huggingface/text-embeddings-inference:cuda-latest
    container_name: agentrag-tei
    ports:
      - "8080:80"
    volumes:
      # HF cache — hub downloads land here (TEI sets HUGGINGFACE_HUB_CACHE=/data).
      - tei_hf_cache:/data
    command:
      # Finetuned e5 (agentrag-embed-v1): MEAN pooling, 768-dim.
      - --model-id=pjavis/agentrag-embed-v1
      - --pooling=mean
      - --max-batch-tokens=16384
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: ["gpu"]
    restart: unless-stopped

  tei-cpu:
    profiles: ["cpu"]
    image: ghcr.io/huggingface/text-embeddings-inference:cpu-1.5
    container_name: agentrag-tei
    ports:
      - "8080:80"
    volumes:
      - tei_hf_cache:/data
    command:
      - --model-id=BAAI/bge-m3
      - --pooling=cls
    restart: unless-stopped

volumes:
  tei_hf_cache:
```

- [ ] **Step 2: Validate compose config**

```bash
docker compose -f deploy/tei.compose.yml --profile gpu config 2>/dev/null | grep -E 'model-id|tei_hf_cache' | head -5
```

Expected: shows `--model-id=pjavis/agentrag-embed-v1` and the `tei_hf_cache` volume mount.

- [ ] **Step 3: Commit**

```bash
git add deploy/tei.compose.yml
git commit -m "feat(deploy): TEI pulls embedding models from HF Hub"
```

Runtime verification deferred to Task 6 (needs the model uploaded first).

---

### Task 5: Model card + upload `agentrag-embed-v1` to HF Hub

**Files:**
- Replace: `models/agentrag-embed-v1/README.md` (auto-generated by sentence-transformers; lives inside the gitignored model dir — uploaded to HF, not committed to git)

**Interfaces:**
- Consumes: local weights at `models/agentrag-embed-v1/`.
- Produces: public HF repo `pjavis/agentrag-embed-v1` that Task 4's TEI config pulls.

**⚠ Data-leak guard:** the existing auto-generated README embeds raw training samples (Vietnamese medical textbook passages and exam questions) in its `widget:` frontmatter and "Training Dataset" samples table. These MUST NOT go public — the replacement below drops all sample content. Before upload, verify no other file in the dir contains training data (weights + tokenizer + config files only).

- [ ] **Step 1: Replace the model card**

Overwrite `models/agentrag-embed-v1/README.md` entirely with:

```markdown
---
license: mit
base_model: intfloat/multilingual-e5-base
language:
  - vi
library_name: sentence-transformers
tags:
  - sentence-transformers
  - sentence-similarity
  - feature-extraction
  - text-embeddings-inference
  - medical
pipeline_tag: sentence-similarity
---

# agentrag-embed-v1

Vietnamese-medical fine-tune of [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) for RAG retrieval in the AgentRag project.

- **Pooling:** mean (+ L2 normalize)
- **Dimensions:** 768
- **Max sequence length:** 512
- **Domain:** Vietnamese medical documents
- **Training:** 5.3k (query, positive, negative) triplets, MultipleNegativesRankingLoss, 2 epochs
- **Eval:** recall@10 +0.20 over base e5 on the project's retrieval benchmark (C1)

## Serving with TEI

```yaml
services:
  tei:
    image: ghcr.io/huggingface/text-embeddings-inference:cuda-latest
    command:
      - --model-id=pjavis/agentrag-embed-v1
      - --pooling=mean
```

Query prefix conventions follow e5: `query: ...` / `passage: ...`.
```

- [ ] **Step 2: Check HF auth**

```bash
uv run hf auth whoami || uv run huggingface-cli whoami
```

Expected: prints HF username. If "Not logged in": **stop and ask the user** to run `! uv run hf auth login` (needs a write token from https://huggingface.co/settings/tokens) — do not proceed until logged in.

- [ ] **Step 3: Upload**

```bash
uv run hf upload pjavis/agentrag-embed-v1 ./models/agentrag-embed-v1 . --repo-type model
```

(Older CLI fallback: `uv run huggingface-cli upload pjavis/agentrag-embed-v1 ./models/agentrag-embed-v1 . --repo-type model`.)

Expected: upload completes, prints repo URL.

- [ ] **Step 4: Verify public pull works without auth**

```bash
curl -s https://huggingface.co/api/models/pjavis/agentrag-embed-v1 | head -c 300
```

Expected: JSON with `"id":"pjavis/agentrag-embed-v1"`, no auth error. If the repo defaulted to private, make it public: `uv run python -c "from huggingface_hub import update_repo_settings; update_repo_settings('pjavis/agentrag-embed-v1', private=False)"`.

- [ ] **Step 5: Nothing to commit** — model dir is gitignored by design. Confirm: `git status --short` shows no `models/` entries.

---

### Task 6: End-to-end verification (secrets, PR, runtime)

**Files:** none (operations only)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: User sets up DockerHub secrets (manual gate)**

Ask the user to do (cannot be automated):
1. DockerHub → Account Settings → Personal access tokens → Generate (Read & Write).
2. GitHub `PJavis/AgentRag` → Settings → Secrets and variables → Actions → New repository secret: `DOCKERHUB_USERNAME` = `pjavis`, `DOCKERHUB_TOKEN` = the token.

- [ ] **Step 2: Push branch, open PR**

```bash
git push -u origin feat/ragas-langfuse-reranker
gh pr create --base master --title "feat: DockerHub CI publish + HF Hub model hosting" --body "$(cat <<'EOF'
CI builds api+frontend images and pushes to DockerHub (master/tags); TEI serves the finetuned embedding from HF Hub. Spec: docs/superpowers/specs/2026-07-05-dockerhub-hf-publish-design.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Watch the PR build-only run**

```bash
gh run watch --exit-status $(gh run list --workflow docker-publish --limit 1 --json databaseId -q '.[0].databaseId')
```

Expected: both matrix jobs green, no push (PR event).

- [ ] **Step 4: TEI runtime check from Hub**

```bash
docker compose -f deploy/tei.compose.yml --profile gpu up -d
sleep 90 && curl -sf http://127.0.0.1:8080/health && \
curl -s http://127.0.0.1:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input":"query: đau đầu","model":"tei"}' | uv run python -c "import json,sys; print(len(json.load(sys.stdin)['data'][0]['embedding']))"
```

Expected: health 200, prints `768`. (First start downloads 1.1GB — if health not ready, wait and retry.)

- [ ] **Step 5: After merge to master (user decides when): verify images public**

```bash
docker manifest inspect pjavis/agentrag-api:latest >/dev/null && echo api-ok
docker manifest inspect pjavis/agentrag-frontend:latest >/dev/null && echo frontend-ok
```

Expected: `api-ok`, `frontend-ok` — from any machine, no login.
