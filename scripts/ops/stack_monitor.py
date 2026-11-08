#!/usr/bin/env python3
"""Application-level health check for the local AgentRag stack.

`/health/providers` only opens TCP sockets. This probes each service the way the
app uses it (a real embed, a real rerank, the Ollama model list, the DeepSeek
balance) and, with --chat, one real /chat turn whose PAYLOAD is validated —
/chat returns errors with HTTP 200.

    python scripts/ops/stack_monitor.py            # free checks only
    python scripts/ops/stack_monitor.py --chat     # + one real chat turn (costs one answer call)
    python scripts/ops/stack_monitor.py --json

Exit code: 0 all OK, 1 at least one WARN, 2 at least one FAIL.
Triage of known-benign warnings: .claude/skills/agentrag-stack-monitor/SKILL.md
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"
_SEVERITY = {OK: 0, SKIP: 0, WARN: 1, FAIL: 2}

DEEPSEEK_LOW_BALANCE_USD = 2.0
# Smoke question: in-corpus, single-hop, must come back grounded.
DEFAULT_CHAT_QUESTION = "Trình bày nguyên nhân gây hiếm muộn ở nam giới."
# Model names that are served by cloud APIs, not by Ollama.
_CLOUD_PREFIXES = ("deepseek", "gemini", "gpt", "claude", "o1", "o3", "o4")


@dataclass
class Check:
    name: str
    status: str
    detail: str
    ms: int = 0


# ── pure verdicts (unit-tested) ──────────────────────────────────────────────

def classify_chat_payload(payload: object) -> tuple[str, str]:
    """A 200 from /chat is not success: errors come back as {"error": ...}."""
    if not isinstance(payload, dict):
        return FAIL, f"non-object payload: {type(payload).__name__}"
    if payload.get("error"):
        return FAIL, f"error in 200 body: {str(payload['error'])[:160]}"
    answer = (payload.get("answer") or "").strip()
    if not answer:
        return FAIL, "empty answer"
    citations = payload.get("citations") or []
    if not citations:
        return WARN, f"answered without citations ({len(answer)} chars) — ungrounded or chitchat-routed"
    return OK, f"{len(answer)} chars, {len(citations)} citations"


def classify_es_health(payload: dict) -> tuple[str, str]:
    status = payload.get("status")
    nodes = payload.get("number_of_nodes")
    unassigned = payload.get("unassigned_shards", 0)
    detail = f"status={status} nodes={nodes} unassigned_shards={unassigned}"
    if status == "green":
        return OK, detail
    if status == "yellow" and nodes == 1:
        # Single-node dev cluster cannot place replicas — expected.
        return OK, detail + " (single node, replicas unassignable)"
    if status == "yellow":
        return WARN, detail
    return FAIL, detail


def classify_deepseek_balance(payload: dict, low: float = DEEPSEEK_LOW_BALANCE_USD) -> tuple[str, str]:
    if not payload.get("is_available", False):
        return FAIL, "is_available=false — calls will 402"
    infos = payload.get("balance_infos") or []
    usd = next((i for i in infos if i.get("currency") == "USD"), infos[0] if infos else None)
    if usd is None:
        return WARN, "no balance info returned"
    total = float(usd.get("total_balance", 0))
    detail = f"{total:.2f} {usd.get('currency')}"
    if total < low:
        return WARN, f"{detail} < {low:.2f} — one ingest pass (~4.6k calls) can drain this"
    return OK, detail


def classify_rerank_scores(payload: object) -> tuple[str, str]:
    """rerank_score must be a probability since #23 — floors assume that scale."""
    if not isinstance(payload, list) or not payload:
        return FAIL, f"unexpected rerank response: {str(payload)[:120]}"
    scores = [float(item.get("score", -1)) for item in payload]
    if any(s < 0 or s > 1 for s in scores):
        return FAIL, f"scores outside [0,1]: {scores} — raw logits? floors assume probabilities"
    return OK, "scores " + ", ".join(f"{s:.3f}" for s in scores)


def required_ollama_models(task_model_map: dict[str, str]) -> set[str]:
    return {
        model for model in task_model_map.values()
        if model and not model.lower().startswith(_CLOUD_PREFIXES)
    }


def classify_ollama_models(available: set[str], required: set[str]) -> tuple[str, str]:
    def norm(name: str) -> str:
        return name if ":" in name else name + ":latest"

    have = {norm(m) for m in available}
    missing = sorted(m for m in required if norm(m) not in have)
    if missing:
        return FAIL, f"missing models {missing} (have {sorted(available)})"
    return OK, f"{len(available)} models, required present: {sorted(required) or 'none'}"


def classify_tei_device(container: str, log_text: str) -> tuple[str, str]:
    """TEI silently falls back to CPU when it cannot see the GPU; compose still
    reports the container healthy. CPU rerank costs ~35-49 s per chat turn."""
    if "Using CPU instead" in log_text or "model on Cpu" in log_text:
        return WARN, f"{container} on CPU (no CUDA device) — expect slow embed/rerank"
    if "model on Cuda" in log_text:
        return OK, f"{container} on GPU"
    return WARN, f"{container}: device not found in logs since start"


# ── IO helpers ───────────────────────────────────────────────────────────────

def _http(method: str, url: str, body: object | None = None, headers: dict | None = None,
          timeout: float = 5.0) -> tuple[int, object]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read(), exc.code
    try:
        return status, json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return status, raw.decode(errors="replace")[:300]


def _tcp(url: str, default_port: int) -> tuple[str, str]:
    parsed = urlparse(url if "://" in url else f"tcp://{url}")
    host, port = parsed.hostname or "127.0.0.1", parsed.port or default_port
    try:
        with socket.create_connection((host, port), timeout=2):
            return OK, f"{host}:{port} accepting"
    except OSError as exc:
        return FAIL, f"{host}:{port} {exc}"


def _origin(url: str) -> str:
    """http://127.0.0.1:8080/v1/ -> http://127.0.0.1:8080"""
    parsed = urlparse(url.split()[0])
    return f"{parsed.scheme}://{parsed.netloc}"


def _timed(name: str, fn) -> Check:
    start = time.monotonic()
    try:
        status, detail = fn()
    except Exception as exc:  # a probe crashing is itself the finding
        status, detail = FAIL, f"{type(exc).__name__}: {exc}"
    return Check(name, status, detail, int((time.monotonic() - start) * 1000))


# ── probes ───────────────────────────────────────────────────────────────────

def run_checks(settings, api_url: str, chat: bool, chat_question: str) -> list[Check]:
    checks: list[Check] = []

    def config():
        from src.agentrag.config_validation import validate_settings
        validate_settings(settings)
        return OK, "validate_settings passed (host .env)"
    checks.append(_timed("config", config))

    def api():
        status, body = _http("GET", f"{api_url}/health/providers")
        if status != 200:
            return FAIL, f"HTTP {status}: {str(body)[:160]}"
        down = [k for k, v in (body.get("providers") or {}).items() if v.get("reachable") is False]
        down += [k for k, v in (body.get("infra") or {}).items() if not v.get("reachable")]
        if down:
            return WARN, f"api up, but it cannot reach: {down}"
        return OK, "api up, providers/infra reachable from api"
    checks.append(_timed("api", api))

    checks.append(_timed("postgres", lambda: _tcp(
        f"tcp://{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}", 5432)))
    if settings.REDIS_URL:
        checks.append(_timed("redis", lambda: _tcp(settings.REDIS_URL, 6379)))

    def es():
        status, body = _http("GET", f"{settings.ELASTICSEARCH_URL.rstrip('/')}/_cluster/health")
        if status != 200:
            return FAIL, f"HTTP {status}"
        return classify_es_health(body)
    checks.append(_timed("elasticsearch", es))

    if settings.EMBEDDING_BASE_URL:
        def embed():
            status, body = _http("POST", f"{_origin(settings.EMBEDDING_BASE_URL)}/embed",
                                 {"inputs": ["health check"]}, timeout=15)
            if status != 200 or not isinstance(body, list) or not body:
                return FAIL, f"HTTP {status}: {str(body)[:160]}"
            return OK, f"dim={len(body[0])}"
        checks.append(_timed("tei-embed", embed))

    if settings.RETRIEVAL_RERANK_ENABLED and settings.RETRIEVAL_RERANK_BACKEND == "tei":
        def rerank():
            status, body = _http("POST", f"{_origin(settings.RETRIEVAL_RERANK_TEI_URL)}/rerank", {
                "query": "nguyên nhân vô sinh nam",
                "texts": ["Vô sinh nam do bất thường tinh trùng.", "Giá xăng tăng."],
                "raw_scores": False,
            }, timeout=15)
            if status != 200:
                return FAIL, f"HTTP {status}: {str(body)[:160]}"
            return classify_rerank_scores(body)
        checks.append(_timed("tei-rerank", rerank))

    import shutil
    import subprocess

    def tei_devices():
        if not shutil.which("docker"):
            return SKIP, "docker CLI not available"
        results = []
        for container in ("agentrag-tei", "agentrag-tei-rerank"):
            started = subprocess.run(
                ["docker", "inspect", container, "--format", "{{.State.StartedAt}}"],
                capture_output=True, text=True, timeout=10)
            if started.returncode != 0:
                continue
            logs = subprocess.run(
                ["docker", "logs", "--since", started.stdout.strip(), container],
                capture_output=True, text=True, timeout=10)
            results.append(classify_tei_device(container, logs.stdout + logs.stderr))
        if not results:
            return SKIP, "no TEI containers found"
        worst = max(results, key=lambda r: _SEVERITY[r[0]])[0]
        return worst, "; ".join(detail for _, detail in results)
    checks.append(_timed("tei-device", tei_devices))

    def ollama():
        try:
            task_map = json.loads(settings.LLM_TASK_MODEL_MAP or "{}")
        except json.JSONDecodeError:
            task_map = {}
        required = required_ollama_models(task_map)
        if not required and settings.EXTRACTION_PROVIDER != "ollama":
            return SKIP, "no task routed to Ollama"
        status, body = _http("GET", f"{_origin(settings.OLLAMA_BASE_URL)}/api/tags")
        if status != 200:
            return FAIL, f"HTTP {status}"
        available = {m["name"] for m in body.get("models", [])}
        return classify_ollama_models(available, required)
    checks.append(_timed("ollama", ollama))

    def deepseek():
        if not settings.DEEPSEEK_API_KEY:
            return SKIP, "DEEPSEEK_API_KEY unset"
        status, body = _http("GET", "https://api.deepseek.com/user/balance",
                             headers={"Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"}, timeout=10)
        if status != 200:
            return FAIL, f"HTTP {status}: {str(body)[:160]}"
        return classify_deepseek_balance(body)
    checks.append(_timed("deepseek-balance", deepseek))

    if settings.LANGFUSE_ENABLED:
        def langfuse():
            status, body = _http("GET", f"{settings.LANGFUSE_HOST.rstrip('/')}/api/public/health")
            return (OK, "up") if status == 200 else (WARN, f"HTTP {status} — traces dropped, chat unaffected")
        checks.append(_timed("langfuse", langfuse))

    if chat:
        def chat_turn():
            status, body = _http("POST", f"{api_url}/chat", {
                "question": chat_question, "conversation_title": "stack_monitor smoke"}, timeout=300)
            if status != 200:
                return FAIL, f"HTTP {status}: {str(body)[:160]}"
            return classify_chat_payload(body)
        checks.append(_timed("chat", chat_turn))

    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--chat", action="store_true", help="also run one real /chat turn (costs an answer call)")
    parser.add_argument("--chat-question", default=DEFAULT_CHAT_QUESTION)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from src.agentrag.config import settings

    checks = run_checks(settings, args.api_url.rstrip("/"), args.chat, args.chat_question)
    worst = max((_SEVERITY[c.status] for c in checks), default=0)

    if args.json:
        print(json.dumps({"exit": worst, "checks": [asdict(c) for c in checks]}, ensure_ascii=False, indent=2))
    else:
        overall = {0: "OK", 1: "WARN", 2: "FAIL"}[worst]
        counts = {s: sum(c.status == s for c in checks) for s in (OK, WARN, FAIL, SKIP)}
        print(f"{overall}  " + " ".join(f"{k.lower()}={v}" for k, v in counts.items() if v))
        for c in checks:
            print(f"  {c.status:<4} {c.name:<17} {c.ms:>6}ms  {c.detail}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
