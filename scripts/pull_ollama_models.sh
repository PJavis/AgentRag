#!/usr/bin/env bash
# Pull every Ollama model referenced in .env.
#
# Scans these pairs and pulls MODEL when PROVIDER=ollama:
#   EMBEDDING_PROVIDER          / EMBEDDING_MODEL
#   EXTRACTION_PROVIDER         / EXTRACTION_MODEL
#   AGENT_PROVIDER              / AGENT_MODEL
#   RETRIEVAL_RERANK_PROVIDER   / RETRIEVAL_RERANK_MODEL
#   VISION_PROVIDER             / VISION_MODEL
#
# Plus any extra model tags listed in OLLAMA_EXTRA_MODELS (space-separated).
#
# Falls back through providers like the runtime does:
#   AGENT_PROVIDER blank → EXTRACTION_PROVIDER
#   RETRIEVAL_RERANK_PROVIDER blank → AGENT/EXTRACTION
#
# Usage:
#   scripts/pull_ollama_models.sh [path/to/.env]    # defaults to ./.env
#   CONTAINER=agentrag-ollama scripts/pull_ollama_models.sh
#   DRY_RUN=1 scripts/pull_ollama_models.sh         # print only

set -euo pipefail

ENV_FILE="${1:-.env}"
CONTAINER="${CONTAINER:-agentrag-ollama}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "$ENV_FILE" ]; then
  echo "❌ env file not found: $ENV_FILE" >&2
  exit 1
fi

# Read KEY=VALUE from .env: strip leading spaces, inline `# comment`,
# wrapping quotes, and surrounding whitespace.
get() {
  local key="$1"
  awk -F= -v k="$key" '
    $0 ~ "^[[:space:]]*"k"=" {
      sub("^[[:space:]]*"k"=", "", $0)
      sub(/[[:space:]]+#.*$/, "", $0)   # strip trailing inline comment
      sub(/^[[:space:]]+/, "", $0)
      sub(/[[:space:]]+$/, "", $0)
      gsub(/^["'\'']|["'\'']$/, "", $0)
      print
      exit
    }
  ' "$ENV_FILE"
}

EMBEDDING_PROVIDER=$(get EMBEDDING_PROVIDER)
EMBEDDING_MODEL=$(get EMBEDDING_MODEL)
EXTRACTION_PROVIDER=$(get EXTRACTION_PROVIDER)
EXTRACTION_MODEL=$(get EXTRACTION_MODEL)
AGENT_PROVIDER=$(get AGENT_PROVIDER)
AGENT_MODEL=$(get AGENT_MODEL)
RERANK_PROVIDER=$(get RETRIEVAL_RERANK_PROVIDER)
RERANK_MODEL=$(get RETRIEVAL_RERANK_MODEL)
RERANK_ENABLED=$(get RETRIEVAL_RERANK_ENABLED)
RERANK_BACKEND=$(get RETRIEVAL_RERANK_BACKEND)
VISION_PROVIDER=$(get VISION_PROVIDER)
VISION_MODEL=$(get VISION_MODEL)
EXTRA=$(get OLLAMA_EXTRA_MODELS)

# Fallback chain mirrors src/agentrag/agent/llm.py + reranker.py
[ -z "$AGENT_PROVIDER" ]  && AGENT_PROVIDER="$EXTRACTION_PROVIDER"
[ -z "$AGENT_MODEL" ]     && AGENT_MODEL="$EXTRACTION_MODEL"
[ -z "$RERANK_PROVIDER" ] && RERANK_PROVIDER="$AGENT_PROVIDER"
[ -z "$RERANK_MODEL" ]    && RERANK_MODEL="$AGENT_MODEL"

models=()
add_if_ollama() {
  local provider="$1" model="$2" label="$3"
  if [ "$provider" = "ollama" ] && [ -n "$model" ]; then
    models+=("$model")
    echo "   • $label → $model"
  fi
}

echo "🔎 Scanning $ENV_FILE for Ollama models..."
add_if_ollama "$EMBEDDING_PROVIDER"  "$EMBEDDING_MODEL"  "EMBEDDING"
add_if_ollama "$EXTRACTION_PROVIDER" "$EXTRACTION_MODEL" "EXTRACTION"
add_if_ollama "$AGENT_PROVIDER"      "$AGENT_MODEL"      "AGENT"
if [ "${RERANK_ENABLED,,}" = "true" ] && [ "$RERANK_BACKEND" != "local_cross_encoder" ]; then
  add_if_ollama "$RERANK_PROVIDER"   "$RERANK_MODEL"     "RERANK"
fi
add_if_ollama "$VISION_PROVIDER"     "$VISION_MODEL"     "VISION"

if [ -n "$EXTRA" ]; then
  for m in $EXTRA; do
    models+=("$m")
    echo "   • EXTRA → $m"
  done
fi

if [ "${#models[@]}" -eq 0 ]; then
  echo "ℹ️  No Ollama models referenced in $ENV_FILE — nothing to pull."
  exit 0
fi

# Dedupe preserving order
declare -A seen
unique=()
for m in "${models[@]}"; do
  if [ -z "${seen[$m]:-}" ]; then
    seen[$m]=1
    unique+=("$m")
  fi
done

if [ "$DRY_RUN" = "1" ]; then
  echo "🧪 DRY_RUN=1 — would pull: ${unique[*]}"
  exit 0
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "❌ Ollama container '$CONTAINER' not running." >&2
  echo "   Start it first:  docker compose --profile local-llm up -d" >&2
  exit 1
fi

echo "📦 Pulling ${#unique[@]} model(s) into '$CONTAINER'..."
# Fallback map: custom finetuned names → base they're derived from.
# When the custom variant isn't in registry AND no local Modelfile artifact
# exists, ensure_ollama_model.sh creates an alias `FROM <base>` so the .env
# stays functional after a `make nuke`.
declare -A FALLBACK_MAP=(
  ["qwen-agentrag"]="qwen2.5:7b-instruct"
  ["agentrag-embed-v1"]="mxbai-embed-large"
)
HERE="$(cd "$(dirname "$0")" && pwd)"
for m in "${unique[@]}"; do
  fb="${FALLBACK_MAP[$m]:-}"
  echo "── ensure $m ${fb:+(fallback: $fb)}"
  CONTAINER="$CONTAINER" bash "$HERE/ensure_ollama_model.sh" "$m" "$fb" \
    || echo "⚠️  failed: $m (continuing)"
done

echo "✅ Done."
