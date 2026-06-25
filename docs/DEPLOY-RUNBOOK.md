# VITAL — Deploy & Ops Runbook

Production/self-host runbook. Closes roadmap **P2.12** (ops hardening: auth/rate-limit
verified, deploy steps, worker autoscaling).

## 1. Services & compose profiles

Infra (always on): `postgres` (+pgvector), `elasticsearch`, `valkey`, `tei` (embeddings).
Profiled services (`docker compose --profile <p> up -d`):

| Profile | Services | Use |
|---|---|---|
| `app` | `api`, `worker`, `frontend` | the application |
| `edge` | `nginx` | same-origin reverse proxy (browser → API) |
| `local-llm` | `ollama` | self-hosted LLM/embeddings |
| `observability` | `langfuse`, `langfuse-db`, `phoenix` | tracing / quality monitor |

A full self-host: `--profile app --profile edge --profile local-llm` (+ `observability`).

## 2. Prerequisites

- Docker + Compose; `uv`; Node (build the frontend image).
- LLM key: `DEEPSEEK_API_KEY` (committed default stack) or `GEMINI_API_KEY` (cloud alt).
- Ollama models if self-hosting: `ollama pull llama3.2:3b` (orchestration) + the embedding
  model. **Reranker gotcha:** `RETRIEVAL_RERANK_BACKEND=local_cross_encoder` +
  `RETRIEVAL_RERANK_MODEL=dengcao/bge-reranker-v2-m3` — the ONLY backend that emits the
  `rerank_score` powering abstain/floor safety; startup guards an API model-name here.

## 3. First deploy

```bash
cp .env.example .env && $EDITOR .env     # set keys + review the section-2 gotchas
make docker-up                            # infra (pg/es/valkey/tei)
make migrate                              # alembic upgrade head (incl. adapter_chat_feedback)
make seed-ontology                        # REQUIRED — ontology resolver + section tagger
docker compose --profile app --profile edge up -d   # api + worker + frontend + nginx
make health                               # pg/es/valkey/ollama + providers reachable
```

Upgrades: `git pull && make migrate && docker compose --profile app up -d --build`.
Migrations are the schema source of truth (never rely on the `create_all` startup net).

## 4. Security — verified config

| Control | Setting (`.env`) | Code | Verify |
|---|---|---|---|
| **Auth** (token + signup) | `AUTH_ENABLED=true`, `AUTH_ALLOW_SIGNUP` | `adapter/auth.py`, `adapter/auth_service.py` | a request without a valid Bearer token → `401` |
| **Rate limit** (per-user/min) | `RATE_LIMIT_ENABLED=true`, `RATE_LIMIT_PER_MIN_DEFAULT=120`, `RATE_LIMIT_UPLOAD_PER_MIN=120` | `adapter/rate_limit.py` | exceed 120 chat calls/min → `429` |
| Doc-scope filter | `DOMAIN_FILTER_ENABLED` | retrieval | answers stay within allowed docs |
| **PHI trace gate** | `OBSERVABILITY_CAPTURE_CONTENT=false` (default) | `common/langfuse_client.py` | with it off, Langfuse traces carry structure/latency only — no question/answer text |
| **Prompt-injection defense** | (always on) | `ANTI_INJECTION_RULE` in `agent/service.py` | retrieved doc content is treated as data, not instructions |
| **Right-to-delete** | (always on, auth-gated) | `DELETE /chat/account` → `adapter/account_deletion.py` | authed user wipes all their data; anonymous/legacy → `403` |

**Hardening checklist before exposing publicly:**
- Keep `AUTH_ENABLED=true`; set `AUTH_ALLOW_SIGNUP=false` after creating accounts if closed.
- Keep `OBSERVABILITY_CAPTURE_CONTENT=false` for PHI — only enable on non-PHI/dev data.
- ⚠️ **Multi-tenancy / IDOR:** notebooks/sources/notes/insights/transformations endpoints
  have NO per-user ownership check (`docs/security/authz-audit-2026-06-25.md`). Fine for a
  single-user/per-clinic deployment; **a launch blocker if multi-tenant** — add the shared
  ownership dependency first.
- Change Langfuse defaults (`LANGFUSE_NEXTAUTH_SECRET`, `SALT`, `LANGFUSE_INIT_USER_PASSWORD`,
  and the dev keys `pk-/sk-lf-agentrag-dev`) — the compose defaults are dev-only.
- Put TLS at `nginx` (`edge` profile); never expose Postgres/ES ports publicly.
- Rotate `POSTGRES_PASSWORD` from the `postgres/postgres` default.

## 5. Worker autoscaling (`scaler.py`)

ARQ background jobs (StructMem extraction, vision, consolidation, chat memory) run on
`worker`. `scaler.py` polls Redis queue depth and spawns/kills worker processes:

```bash
python scaler.py            # one scaler manages N arq worker processes
```

| Env | Default | Meaning |
|---|---|---|
| `SCALER_MIN_WORKERS` | 1 | floor |
| `SCALER_MAX_WORKERS` | 4 | ceiling |
| `SCALER_SCALE_UP_AT` | 5 | queue depth per extra worker |
| `SCALER_POLL_SECONDS` | 5 | poll interval |
| `SCALER_COOLDOWN_SECONDS` | 30 | min seconds between rescales |

SIGTERM/Ctrl-C drains workers gracefully. Tune `MAX_WORKERS` to LLM/embedding throughput
(cloud providers scale wider than a single local GPU). StructMem ingest is the heaviest
job (~2h/100 docs of graph extraction) — size the ceiling for ingest bursts.

## 6. Observability (optional)

`docker compose --profile observability up -d` → Langfuse at `:3002`, Phoenix at its port.
Set in `.env`: `LANGFUSE_ENABLED=true`, `LANGFUSE_HOST=http://localhost:3002`, the project
keys. Each `/chat` turn becomes one trace (grouped by conversation); thumbs feedback lands
as a `user_feedback` score on the turn's trace. See `docs/eval/langfuse-online-2026-06-25.md`.

## 7. Verify a deploy works

1. `make health` → all infra + providers reachable.
2. `make test-fast` → green backend gate.
3. **Chat works**: ask an in-corpus question → answer with `[n]` citations.
4. **Safety fires**: ask an out-of-corpus question (a made-up drug) → the system refuses
   ("Tài liệu hiện có không có thông tin…") and cites nothing. If it answers confidently,
   the rerank backend is misconfigured (section 2).
5. Benchmarks now run a preflight (`scripts/eval/run_benchmark.py`) that fails fast if
   ES/embedding/judge are not ready.

## 8. Backups & recovery

- **Postgres is the source of truth** (documents, segments, conversations, feedback).
  Back up `postgres_data`. Elasticsearch is a rebuildable projection of PG.
- Volumes: `postgres_data`, `es_data`, `valkey_data`, `ollama_data`, `images_data`,
  `langfuse_db_data`. Snapshot `postgres_data` + `images_data` at minimum.
