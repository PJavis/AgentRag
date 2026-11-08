---
name: agentrag-stack-monitor
description: Use when checking whether the local AgentRag stack is healthy, before a demo, before an eval or re-ingest, after `docker compose up`, or when chat is slow, errors, or answers without citations. Triggers on "is the stack ok", "check health", "run the monitor", "why is chat slow", "is rerank on GPU", "check deepseek balance", "pre-demo check". Runs scripts/ops/stack_monitor.py and says which findings are known-benign and which need action.
---

# AgentRag stack monitor

## Run it

```bash
.venv/bin/python scripts/ops/stack_monitor.py          # free checks, ~1 s
.venv/bin/python scripts/ops/stack_monitor.py --chat   # + one real /chat turn (one answer-model call)
.venv/bin/python scripts/ops/stack_monitor.py --json
```

Exit 0 = OK, 1 = WARN, 2 = FAIL. Run it from the host (it reads the host `.env`,
the same one `make api` uses). Use `--chat` before a demo or eval, not in a loop:
each run writes a "stack_monitor smoke" conversation and spends one answer call.

`/health/providers` (what `make health` hits) only opens TCP sockets. This script
does a real embed, a real rerank, reads Ollama's model list, the DeepSeek balance,
and with `--chat` validates the `/chat` **payload** — `/chat` returns errors with
HTTP 200.

## Triage

Most of the value is knowing which lines are expected. Check this table before
investigating anything.

| Line | Verdict | What it means / what to do |
|---|---|---|
| `elasticsearch ... yellow nodes=1` | **Benign** — script already reports OK | Single-node dev cluster cannot place replicas. |
| `tei-device WARN ... on CPU` | **Real, high impact** | TEI could not see the GPU. Compose still says Healthy (TEI images have no healthcheck). CPU rerank = ~35-49 s per turn vs ~0.4 s. `docker restart` of the container alone did NOT fix it on 2026-11-08 (Ollama on the same daemon had CUDA). Try restarting Docker Desktop / `wsl --shutdown`, then re-run. |
| `tei-rerank FAIL Connection reset` right after start | Usually **warming up** | Port is open, model still loading. On CPU this takes minutes. Re-run after a minute; if `tei-device` is WARN, that is the real problem. |
| `tei-embed dim=` not 768 | **Real, silent corruption** | Index holds 768-dim e5-FT vectors. `docker-compose.yml`'s own `tei` service serves `BAAI/bge-m3` (1024) under the SAME container name `agentrag-tei` — start TEI with `make serve-embed` (deploy/tei.compose.yml), never let bare compose create it. |
| `tei-rerank FAIL scores outside [0,1]` | **Real** | Raw logits. Relevance floor `RETRIEVAL_RELEVANCE_MIN_PROB` assumes probabilities since #23; every turn would gate wrongly.  Note: this probes TEI directly — an app-side double sigmoid (the #23 bug) is NOT visible here; the reranker unit tests cover that. |
| `deepseek-balance WARN < 2.00` | **Real before ingest**, fine for chat | One full re-ingest is ~4.6k calls (~$1 per ~700). A 402 takes chat down too. Top up before any ingest. |
| `deepseek-balance FAIL is_available=false` | **Real, chat down** | Every answer call 402s. |
| `ollama FAIL missing models` | **Real** | A task in `LLM_TASK_MODEL_MAP` (e.g. decide → `llama3.2:3b`) is routed to a model Ollama lacks. `make ollama-pull`. |
| `ollama FAIL` connection refused, containers up | Check which Ollama | Containers need Ollama at `host.docker.internal:11434`; an `ollama serve` inside WSL is not reachable from containers. |
| `api WARN cannot reach [...]` | **Real** | The api container's own view differs from the host's (container URLs come from compose env, not `.env`). |
| `config FAIL` | **Real** | `validate_settings` would also fail at api startup. Fix `.env`. |
| `chat FAIL error in 200 body` | **Real** | Read the error. `APIConnectionError` in decide = Ollama unreachable from the api container. |
| `chat WARN answered without citations` | **Investigate** | Either the question got chitchat-routed (see the `_is_chitchat` whole-word fix) or retrieval came back empty. The default smoke question is in-corpus and must be grounded. |
| `langfuse` absent | Benign | Only checked when `LANGFUSE_ENABLED=true` (OFF by default, PHI). |

## Starting the stack (the way that works)

Bare `make docker-up` collides with the deploy TEI container (`Conflict. The container
name "/agentrag-tei" is already in use`) and, worse, would serve the wrong model if
that container were gone. Instead:

```bash
make serve-embed                                                     # e5-FT TEI on :8080
docker compose up -d --wait elasticsearch postgres tei-rerank valkey
docker compose --profile app --profile local-llm up -d --no-deps --no-build ollama api worker
.venv/bin/python scripts/ops/stack_monitor.py
```

Note `docker compose up ... | tail` reports exit 0 even when compose failed — check
`${PIPESTATUS[0]}` or read the output.
