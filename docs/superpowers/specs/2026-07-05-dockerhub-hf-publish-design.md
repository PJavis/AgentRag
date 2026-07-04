# Design: Publish images to DockerHub (CI) + finetuned model to HF Hub

Date: 2026-07-05
Status: approved

## Goal

Make AgentRag deployable anywhere without building on the target machine:
code on GitHub, app images on DockerHub (built by CI), finetuned embedding
model on HuggingFace Hub. A deploy box needs only the compose files and a
`.env` — `docker compose pull && docker compose --profile app up -d` and
everything downloads itself.

Non-goal: freeing local disk via the registry (a push is a copy, not a move —
local cleanup is `docker builder prune` / `docker image prune`, out of scope).

## Decisions (locked with user)

| Decision | Choice |
|---|---|
| Build method | GitHub Actions CI builds + pushes; no mandatory local builds |
| DockerHub namespace | `pjavis` → `pjavis/agentrag-api`, `pjavis/agentrag-frontend` |
| Image visibility | Public |
| CI triggers | Push to `master` → `latest` + `sha-<short>`; tags `v*` → semver; PRs → build-only smoke check (no push) |
| Platforms | `linux/amd64` only (no ARM target today; add later if needed) |
| HF model repo | `pjavis/agentrag-embed-v1`, public |

## Components

### 1. GitHub Actions workflow — `.github/workflows/docker-publish.yml`

- Triggers: `push` to `master`, `push` tags `v*`, `pull_request` to `master`.
- Job matrix over two images:
  - `agentrag-api`: context `.`, root `Dockerfile` (api + worker share the
    image; worker just overrides the command in compose).
  - `agentrag-frontend`: context `./frontend`.
- `docker/login-action` using repo secrets `DOCKERHUB_USERNAME` /
  `DOCKERHUB_TOKEN`; login and push are skipped on PR events.
- `docker/metadata-action` generates tags: `latest` (master), `sha-<short>`,
  `X.Y.Z` from `v*` tags.
- `docker/build-push-action` with `cache-from`/`cache-to: type=gha` so repeat
  CI builds reuse layers.

### 2. Compose changes

- `docker-compose.yml`: add `image: pjavis/agentrag-api:latest` to `api` and
  `worker`, `image: pjavis/agentrag-frontend:latest` to `frontend`, keeping the
  existing `build:` blocks. Local dev unchanged (`--build` still builds); a
  deploy box runs `docker compose pull` and never builds.
- `deploy/tei.compose.yml`: serve the finetuned model from the Hub instead of
  a bind mount — `--model-id=pjavis/agentrag-embed-v1`, drop
  `HF_HUB_OFFLINE=1`, add a named volume for the HF cache so the 1.1GB
  download happens once and survives restarts. Local `models/` dir stays for
  training/offline work.

### 3. HF Hub model upload (one-time, manual)

- `hf auth login` with a write token, then
  `hf upload pjavis/agentrag-embed-v1 ./models/agentrag-embed-v1`.
- Model card README: base `intfloat/multilingual-e5-base`, mean pooling,
  768-dim, VN-medical finetune, TEI serving snippet.
- HF free tier covers this (public repos free; 1.1GB trivial).

### 4. `.dockerignore` hardening

Root `.dockerignore` currently misses `models/` (3.3GB) and `checkpoints/` —
CI checkouts don't contain them (gitignored) but local builds would bake them
into the image. Add: `models/`, `checkpoints/`, `docs/`, `frontend/`,
`*.docx`, `image.png`.

## Manual steps (user, one-time)

1. DockerHub → Account Settings → Personal access token (read/write scope).
2. GitHub repo → Settings → Secrets and variables → Actions → add
   `DOCKERHUB_USERNAME=pjavis`, `DOCKERHUB_TOKEN=<token>`.
3. HF: create write token, `hf auth login`, run the upload command.

## Gotchas / constraints

- The workflow only triggers on master pushes once the workflow file exists on
  master — merge the current feature branch (or cherry-pick the workflow) to
  activate it.
- Public images bake source code but never `.env` (verified dockerignored in
  both contexts).
- DockerHub free tier: public repos unlimited; anonymous pull rate limits
  apply on deploy boxes (fine at this scale).
- Assumes HF username `pjavis`; adjust repo id if the actual handle differs.

## Testing

- PR opened with the workflow → both matrix jobs build green (no push).
- After merge to master: `docker pull pjavis/agentrag-api:latest` and
  `docker pull pjavis/agentrag-frontend:latest` succeed from a clean machine.
- TEI: `docker compose -f deploy/tei.compose.yml up -d` on a box without
  `models/` → downloads from Hub, `GET :8080/health` OK, embedding dim 768.
- End-to-end deploy smoke: `docker compose pull && docker compose --profile
  app up -d` on a machine without the repo's build context → API healthy.
