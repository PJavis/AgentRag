# AgentRag — one-stop developer Makefile
#
# QUICK START
#   make install            docker up + uv sync + npm install + alembic migrate
#   make seed-ontology      seed medical taxonomy + backfill ES tags  (S5)
#   make dev                api + worker + frontend  (foreground, Ctrl+C all)
#   make health             curl /config/validate + /on/api/auth/status
#
# RUN A SINGLE COMPONENT
#   make api                FastAPI dev server   (reload)
#   make api-prod           gunicorn + uvicorn workers
#   make frontend           Next.js dev server
#   make worker             single ARQ worker
#   make scaler             auto-scaling ARQ worker pool
#   make cli                interactive chat CLI
#
# BACKGROUND
#   make up-bg              start api + worker + frontend in background
#   make logs               tail .run/*.log
#   make stop               kill background servers
#
# S1 — COST & TOKEN DASHBOARD
#   make cost-reset         clear in-memory LLM cost ledger
#   make dashboard-open     open http://localhost:3000/cost in browser
#
# S5 — ONTOLOGY / DOMAIN PARTITION
#   make seed-ontology      run scripts/seed_ontology.py + backfill_tags
#   make backfill-tags      re-tag ES segments only (no seed)
#
# OPS
#   make migrate            alembic upgrade head
#   make reset              nuke + rebuild docker volumes (keeps .env)
#   make reset-data         + wipes extracted images and ingested data
#   make nuke               + wipes deps (.venv, node_modules, .next, etc)
#   make clean              remove cached artifacts only (no docker)
#
# TESTS
#   make test               full pytest suite (needs Postgres + ES up)
#   make test-fast          unit-ish suite — skips infra-dependent tests
#
# All commands assume this repo's root as cwd.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

ROOT := $(shell pwd)
FRONTEND_DIR := $(ROOT)/frontend
LOG_DIR := $(ROOT)/.run
PIDS := api frontend worker scaler

API_PORT ?= 8000
FRONTEND_PORT ?= 3000
UVICORN_HOST ?= 0.0.0.0
UVICORN_RELOAD ?= --reload
UVICORN_WORKERS ?= 1

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@awk '/^# / { sub(/^# ?/,"",$$0); print } /^[a-zA-Z][a-zA-Z0-9_-]*:.*/ && !/^\.PHONY/ && !/^help:/ { sub(/:.*/,"",$$1); printf "  \033[36m%-18s\033[0m\n", $$1 }' $(MAKEFILE_LIST) | head -60

# ── Install ───────────────────────────────────────────────────────────────────

.PHONY: install
install: docker-up env uv-sync frontend-install migrate
	@echo
	@echo "✅ Install complete. Run \`make dev\` to start everything."

.PHONY: env
env:
	@if [ ! -f .env ]; then \
	  cp .env.example .env && \
	  echo "📝 Created .env from .env.example — review the values before running."; \
	else \
	  echo "✓ .env already exists"; \
	fi
	@if [ ! -f $(FRONTEND_DIR)/.env.local ]; then \
	  cp $(FRONTEND_DIR)/.env.local.example $(FRONTEND_DIR)/.env.local 2>/dev/null \
	    || printf "API_URL=http://localhost:%s/on\nINTERNAL_API_URL=http://localhost:%s/on\n" $(API_PORT) $(API_PORT) > $(FRONTEND_DIR)/.env.local; \
	  echo "📝 Created frontend/.env.local"; \
	else \
	  echo "✓ frontend/.env.local already exists"; \
	fi

.PHONY: uv-sync
uv-sync:
	uv sync

.PHONY: frontend-install
frontend-install:
	@cd $(FRONTEND_DIR) && \
	  if [ ! -d node_modules ]; then npm install; else echo "✓ frontend node_modules already installed"; fi

# ── Infra ─────────────────────────────────────────────────────────────────────

.PHONY: docker-up
docker-up:
	docker compose up -d --wait

.PHONY: docker-down
docker-down:
	docker compose down

# Bring up the full app stack (api + worker + frontend) inside docker
.PHONY: docker-up-app
docker-up-app:
	docker compose --profile app up -d --build

# Bring up the edge layer (nginx) — front of api+frontend, listens on :80
.PHONY: docker-up-edge
docker-up-edge:
	docker compose --profile app --profile edge up -d --build

.PHONY: docker-down-app
docker-down-app:
	docker compose --profile app --profile edge down

# Bring up Ollama container, then pull every model your .env references.
# Provider/model pairs scanned: EMBEDDING, EXTRACTION, AGENT, RETRIEVAL_RERANK, VISION.
# Add OLLAMA_EXTRA_MODELS="tag1 tag2" in .env to pull more (e.g. routing-only tags).
.PHONY: docker-up-llm
docker-up-llm:
	docker compose --profile local-llm up -d
	@bash scripts/pull_ollama_models.sh .env

# Pull models without (re)starting compose; same dynamic logic.
.PHONY: ollama-pull
ollama-pull:
	@bash scripts/pull_ollama_models.sh .env

# After `make nuke` or fresh checkout: re-register all models referenced in
# .env (registry pull, local Modelfile, or alias fallback to base). Idempotent.
.PHONY: reseed-models
reseed-models:
	@bash scripts/pull_ollama_models.sh .env

# Preview what would be pulled.
.PHONY: ollama-pull-dry
ollama-pull-dry:
	@DRY_RUN=1 bash scripts/pull_ollama_models.sh .env

# Pull a specific vision model on demand (manual override).
VISION_MODEL_TAG ?= llava:13b
.PHONY: vision-pull
vision-pull:
	docker exec agentrag-ollama ollama pull $(VISION_MODEL_TAG) || \
	  (echo "❌ ollama container not running. Run \`make docker-up-llm\` first."; exit 1)
	@echo "✅ Pulled $(VISION_MODEL_TAG). Set in .env:"
	@echo "    VISION_PROVIDER=ollama"
	@echo "    VISION_MODEL=$(VISION_MODEL_TAG)"
	@echo "    VISION_BASE_URL=http://127.0.0.1:11434/v1/"

.PHONY: migrate
migrate:
	uv run alembic upgrade head

# ── S5: Ontology seed + tag backfill ──────────────────────────────────────────
# Idempotent. Safe to re-run; upserts by (canonical_norm, source).
.PHONY: seed-ontology
seed-ontology: backfill-tags-prepare
	PYTHONPATH=. uv run python scripts/seed_ontology.py \
	  --yaml data/ontology/custom_terms.yaml \
	  $$([ -f data/ontology/icd10_vn.csv ] && echo "--icd10 data/ontology/icd10_vn.csv")
	$(MAKE) -s backfill-tags

.PHONY: backfill-tags-prepare
backfill-tags-prepare:
	@echo "  (assumes Postgres + ES are up and pg_trgm migration applied)"

.PHONY: backfill-tags
backfill-tags:
	PYTHONPATH=. uv run python scripts/backfill_tags.py

.PHONY: backfill-tags-dry
backfill-tags-dry:
	PYTHONPATH=. uv run python scripts/backfill_tags.py --dry-run

# ── Run ───────────────────────────────────────────────────────────────────────

.PHONY: api
api:
	uv run uvicorn main:app --host $(UVICORN_HOST) --port $(API_PORT) $(UVICORN_RELOAD)

.PHONY: api-prod
api-prod:
	uv run gunicorn -k uvicorn.workers.UvicornWorker -w $(UVICORN_WORKERS) -b 0.0.0.0:$(API_PORT) main:app

.PHONY: frontend
frontend:
	cd $(FRONTEND_DIR) && npm run dev

.PHONY: worker
worker:
	uv run arq src.agentrag.worker.settings.WorkerSettings

.PHONY: scaler
scaler:
	uv run python scaler.py

.PHONY: cli
cli:
	uv run python cli.py chat

# ── Parallel dev (foreground; Ctrl+C kills all) ──────────────────────────────

.PHONY: dev
dev:
	@mkdir -p $(LOG_DIR)
	@echo "🚀 Starting api + worker + frontend (Ctrl+C to stop all)"
	@trap 'kill 0' SIGINT SIGTERM EXIT; \
	  ( $(MAKE) -s api 2>&1 | sed 's/^/[api] /' ) & \
	  ( $(MAKE) -s worker 2>&1 | sed 's/^/[worker] /' ) & \
	  ( $(MAKE) -s frontend 2>&1 | sed 's/^/[web] /' ) & \
	  wait

# ── Background variants (write logs/pids under .run/) ────────────────────────

$(LOG_DIR):
	@mkdir -p $(LOG_DIR)

.PHONY: api-bg
api-bg: $(LOG_DIR)
	@nohup uv run uvicorn main:app --host $(UVICORN_HOST) --port $(API_PORT) $(UVICORN_RELOAD) \
	  > $(LOG_DIR)/api.log 2>&1 & echo $$! > $(LOG_DIR)/api.pid
	@echo "🟢 api → $(LOG_DIR)/api.log (pid $$(cat $(LOG_DIR)/api.pid))"

.PHONY: frontend-bg
frontend-bg: $(LOG_DIR)
	@cd $(FRONTEND_DIR) && nohup npm run dev > $(LOG_DIR)/frontend.log 2>&1 & echo $$! > $(LOG_DIR)/frontend.pid
	@echo "🟢 frontend → $(LOG_DIR)/frontend.log (pid $$(cat $(LOG_DIR)/frontend.pid))"

.PHONY: worker-bg
worker-bg: $(LOG_DIR)
	@nohup uv run arq src.agentrag.worker.settings.WorkerSettings > $(LOG_DIR)/worker.log 2>&1 & echo $$! > $(LOG_DIR)/worker.pid
	@echo "🟢 worker → $(LOG_DIR)/worker.log (pid $$(cat $(LOG_DIR)/worker.pid))"

.PHONY: scaler-bg
scaler-bg: $(LOG_DIR)
	@nohup uv run python scaler.py > $(LOG_DIR)/scaler.log 2>&1 & echo $$! > $(LOG_DIR)/scaler.pid
	@echo "🟢 scaler → $(LOG_DIR)/scaler.log (pid $$(cat $(LOG_DIR)/scaler.pid))"

.PHONY: up-bg
up-bg: api-bg worker-bg frontend-bg
	@echo "✅ All background servers started — \`make logs\` to tail, \`make stop\` to kill."

.PHONY: logs
logs:
	@for n in $(PIDS); do [ -f $(LOG_DIR)/$$n.log ] && echo "===== $$n =====" && tail -n 30 $(LOG_DIR)/$$n.log || true; done

.PHONY: stop
stop:
	@# 1) Kill any process whose pid we recorded
	@for n in $(PIDS); do \
	  if [ -f $(LOG_DIR)/$$n.pid ]; then \
	    pid=$$(cat $(LOG_DIR)/$$n.pid); \
	    kill $$pid 2>/dev/null && echo "🛑 stopped $$n (pid $$pid)" || echo "⚠️  $$n (pid $$pid) not running"; \
	    rm -f $(LOG_DIR)/$$n.pid; \
	  fi; \
	done
	@# 2) Belt-and-suspenders: kill anything still bound to API_PORT / FRONTEND_PORT.
	@# Tolerate empty results under set -e via leading `:` no-op + trailing `|| true`.
	@for p in $(API_PORT) $(FRONTEND_PORT); do \
	  pids=$$(ss -ltnp 2>/dev/null | awk -v port=":$$p" '$$4 ~ port {print}' | grep -oP 'pid=\K[0-9]+' 2>/dev/null | sort -u || true); \
	  for pid in $$pids; do \
	    kill -9 $$pid 2>/dev/null && echo "🛑 killed stray pid $$pid on :$$p" || true; \
	  done; \
	done || true
	@# 3) Sweep any orphan uvicorn / next-dev / arq still running.
	@# Use -P $$$$ exclusions via setsid would be cleaner, but here we
	@# simply run pkill in a backgrounded subshell so signals can't
	@# propagate back up to this make invocation.
	@-(pkill -9 -f "uvicorn main:app" >/dev/null 2>&1 && echo "🛑 swept uvicorn"; true)
	@-(pkill -9 -f "next dev"        >/dev/null 2>&1 && echo "🛑 swept next dev"; true)
	@-(pkill -9 -f "arq src.agentrag.worker.settings" >/dev/null 2>&1 && echo "🛑 swept arq worker"; true)
	@true

# ── Maintenance ───────────────────────────────────────────────────────────────

.PHONY: reset
reset:
	@echo "🧹 Resetting databases (postgres + ES + redis + ollama volumes)..."
	$(MAKE) -s stop || true
	docker compose --profile app --profile edge down -v --remove-orphans
	rm -rf .cache/agentrag
	docker compose up -d --wait
	$(MAKE) -s migrate
	@echo "✅ Reset complete. Run \`make dev\` to start."

.PHONY: reset-data
reset-data:
	@echo "⚠️  Wiping ALL ingested data (DBs + extracted images + cache)..."
	$(MAKE) -s stop || true
	docker compose --profile app --profile edge down -v --remove-orphans
	rm -rf .cache/agentrag
	rm -rf data/images/*
	rm -rf $(LOG_DIR)
	docker compose up -d --wait
	$(MAKE) -s migrate
	@echo "✅ Data wiped. Infra back up. Run \`make dev\`."

.PHONY: nuke
nuke:
	@echo "💣 Nuking EVERYTHING — containers, volumes, code caches, deps, builds..."
	@echo "   (.env will be kept — re-edit if needed)"
	$(MAKE) -s stop || true
	docker compose --profile app --profile edge --profile local-llm down -v --remove-orphans
	rm -rf .cache .cache/agentrag .pytest_cache .run
	rm -rf data/images/*
	find . -type d -name __pycache__ -prune -exec rm -rf {} \; 2>/dev/null || true
	rm -rf $(FRONTEND_DIR)/.next $(FRONTEND_DIR)/.turbo $(FRONTEND_DIR)/node_modules
	rm -rf .venv
	@echo "✅ Nuked. Next:"
	@echo "  1. make install        # rebuild deps + DB schema"
	@echo "  2. make docker-up-llm  # auto-reseed Ollama models (registry / local Modelfile / alias)"
	@echo "  Custom finetuned models (qwen-agentrag, agentrag-embed-v1) are aliased to their base"
	@echo "  unless models/ artifacts remain. Re-run finetune to restore quality."

.PHONY: clean
clean:
	rm -rf .cache __pycache__ .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} \;
	rm -rf $(FRONTEND_DIR)/.next $(FRONTEND_DIR)/.turbo

.PHONY: deepclean
deepclean: clean
	rm -rf $(FRONTEND_DIR)/node_modules
	rm -rf .venv

# ── Tests / health ────────────────────────────────────────────────────────────

.PHONY: health
health:
	@curl -fsS http://127.0.0.1:$(API_PORT)/config/validate || (echo "API not reachable on :$(API_PORT)"; exit 1)
	@curl -fsS http://127.0.0.1:$(API_PORT)/on/api/auth/status

.PHONY: test
test:
	uv run pytest -q

# Skips ontology/* + ingestion/* (need Postgres). Use after `make install`
# but before docker stack is fully wired.
.PHONY: test-fast
test-fast:
	uv run pytest -q --ignore=tests/ontology --ignore=tests/ingestion

# ── S1: Cost dashboard helpers ────────────────────────────────────────────────
.PHONY: cost-reset
cost-reset:
	@curl -fsS -X POST http://127.0.0.1:$(API_PORT)/on/api/metrics/cost/reset \
	  -H "Authorization: Bearer $${OPEN_NOTEBOOK_PASSWORD:-demo}" | jq . 2>/dev/null \
	  || echo "  (api not reachable or jq missing)"

.PHONY: dashboard-open
dashboard-open:
	@command -v xdg-open >/dev/null && xdg-open http://localhost:$(FRONTEND_PORT)/cost \
	  || command -v open >/dev/null && open http://localhost:$(FRONTEND_PORT)/cost \
	  || echo "Open http://localhost:$(FRONTEND_PORT)/cost"

.PHONY: bench-ingest
bench-ingest:
	uv run python scripts/benchmark_ingest.py data/test_docs/SYSTEM_DESIGN.md

# ── Finetune loop (16 GB GPU box) ─────────────────────────────────────────────
# See docs/FINETUNE_STRATEGY.md for the full plan.

FT_BASE_EMBED ?= intfloat/multilingual-e5-base
FT_BASE_RERANK ?= BAAI/bge-reranker-v2-m3
FT_BASE_LLM ?= unsloth/Qwen2.5-7B-Instruct
FT_OUT_EMBED ?= models/agentrag-embed-v1
FT_OUT_RERANK ?= models/agentrag-rerank-v1
FT_OUT_LLM ?= models/qwen-agentrag-7b
FT_OLLAMA_NAME ?= qwen-agentrag
FT_QUANT ?= Q4_K_M

.PHONY: mine-pairs
mine-pairs:
	uv run python scripts/mine_finetune_pairs.py \
	  --out data/finetune/embed_triplets.jsonl

.PHONY: split-pairs
split-pairs:
	uv run python scripts/split_pairs.py \
	  --input data/finetune/embed_triplets.jsonl \
	  --train data/finetune/embed_train.jsonl \
	  --test  data/finetune/embed_test.jsonl

.PHONY: train-embed
train-embed:
	uv run python scripts/finetune_embedding.py \
	  --base $(FT_BASE_EMBED) \
	  --train data/finetune/embed_train.jsonl \
	  --out  $(FT_OUT_EMBED)

.PHONY: eval-embed
eval-embed:
	uv run python scripts/eval_retrieval.py \
	  --baseline $(FT_BASE_EMBED) \
	  --candidate $(FT_OUT_EMBED) \
	  --test data/finetune/embed_test.jsonl

.PHONY: serve-embed
serve-embed:
	docker compose -f deploy/tei.compose.yml --profile gpu up -d
	@echo "✅ TEI on :8080. Set in .env:"
	@echo "    EMBEDDING_PROVIDER=openai"
	@echo "    EMBEDDING_MODEL=$$(basename $(FT_OUT_EMBED))"
	@echo "    EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1/"
	@echo "    OPENAI_API_KEY=tei-dummy"

.PHONY: stop-embed
stop-embed:
	docker compose -f deploy/tei.compose.yml down

.PHONY: train-rerank
train-rerank:
	uv run python scripts/finetune_reranker.py \
	  --base $(FT_BASE_RERANK) \
	  --train data/finetune/embed_train.jsonl \
	  --out  $(FT_OUT_RERANK)

.PHONY: mine-sft
mine-sft:
	uv run python scripts/mine_sft.py --out data/finetune/llm_sft.jsonl --limit 1000

.PHONY: train-llm-lora
train-llm-lora:
	uv run python scripts/finetune_qwen_lora.py \
	  --base $(FT_BASE_LLM) \
	  --train data/finetune/llm_sft.jsonl \
	  --out  $(FT_OUT_LLM)

.PHONY: convert-llm
convert-llm:
	bash scripts/convert_to_ollama.sh $(FT_OUT_LLM) $(FT_OLLAMA_NAME) $(FT_QUANT)

# End-to-end nightly: mine → split → train embed → eval → (if exit 0) restart TEI.
.PHONY: retrain-embedding-nightly
retrain-embedding-nightly:
	$(MAKE) mine-pairs
	$(MAKE) split-pairs
	$(MAKE) train-embed FT_OUT_EMBED=models/agentrag-embed-candidate
	@$(MAKE) eval-embed FT_OUT_EMBED=models/agentrag-embed-candidate && \
	  rm -rf $(FT_OUT_EMBED).prev && \
	  mv -f $(FT_OUT_EMBED) $(FT_OUT_EMBED).prev 2>/dev/null || true && \
	  mv models/agentrag-embed-candidate $(FT_OUT_EMBED) && \
	  docker compose -f deploy/tei.compose.yml restart tei-gpu && \
	  echo "✅ promoted candidate → prod" || \
	  echo "❌ candidate failed gate, kept old"
