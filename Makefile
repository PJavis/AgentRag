# AgentRag — one-stop developer Makefile
#
# Quick start:
#   make install        # docker compose up + uv sync + npm install + migrate
#   make dev            # run api + worker + frontend in parallel (foreground)
#   make api            # just the FastAPI server
#   make frontend       # just the Next.js dev server
#   make worker         # one ARQ worker
#   make scaler         # auto-scaling ARQ worker pool
#   make migrate        # alembic upgrade head
#   make reset          # nuke + rebuild docker volumes
#   make clean          # remove cached artifacts (no docker)
#   make stop           # stop background dev servers started by `make up-bg`
#
# Background variants:
#   make api-bg | make frontend-bg | make worker-bg
#   make logs           # tail logs from background servers
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
	docker compose up -d

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

.PHONY: docker-up-llm
docker-up-llm:
	docker compose --profile local-llm up -d
	docker exec agentrag-ollama ollama pull qwen2.5:14b-instruct || true
	docker exec agentrag-ollama ollama pull nomic-embed-text || true
	docker exec agentrag-ollama ollama pull dengcao/bge-reranker-v2-m3 || true

# Pull a vision model for image parsing (PDF images + standalone uploads).
# Default is llava:13b; override with VISION_MODEL_TAG=llava:7b for less VRAM.
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
	@for n in $(PIDS); do \
	  if [ -f $(LOG_DIR)/$$n.pid ]; then \
	    pid=$$(cat $(LOG_DIR)/$$n.pid); \
	    kill $$pid 2>/dev/null && echo "🛑 stopped $$n (pid $$pid)" || echo "⚠️  $$n (pid $$pid) not running"; \
	    rm -f $(LOG_DIR)/$$n.pid; \
	  fi; \
	done

# ── Maintenance ───────────────────────────────────────────────────────────────

.PHONY: reset
reset:
	docker compose down -v --remove-orphans
	rm -rf .cache/agentrag
	docker compose up -d
	$(MAKE) -s migrate

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

.PHONY: bench-ingest
bench-ingest:
	uv run python scripts/benchmark_ingest.py data/test_docs/SYSTEM_DESIGN.md
