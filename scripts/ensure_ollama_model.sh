#!/usr/bin/env bash
# Ensure an Ollama model is available — robust against reset/nuke.
#
# Resolution order:
#   1. `ollama show <name>`             → already registered, nothing to do
#   2. `ollama pull <name>`             → public model from registry
#   3. models/<name>/Modelfile          → register from local GGUF (convert_to_ollama.sh output)
#   4. models/<name>-*/Modelfile        → any matching local dir (e.g. qwen-agentrag-7b)
#   5. fallback: FALLBACK env var       → write `FROM <fallback>` Modelfile + register as alias
#
# Usage:
#   scripts/ensure_ollama_model.sh <model_name> [fallback_base_model]
#   scripts/ensure_ollama_model.sh qwen-agentrag qwen2.5:7b-instruct
#
# Env:
#   CONTAINER=agentrag-ollama  (default)
#   MODELS_DIR=models          (default)

set -euo pipefail

NAME="${1:?usage: ensure_ollama_model.sh <name> [fallback]}"
FALLBACK="${2:-}"
CONTAINER="${CONTAINER:-agentrag-ollama}"
MODELS_DIR="${MODELS_DIR:-models}"

run_ollama() { docker exec "$CONTAINER" ollama "$@"; }

# 1. Already registered?
if run_ollama show "$NAME" >/dev/null 2>&1; then
  echo "✅ $NAME — already registered"
  exit 0
fi

# 2. Try registry pull (works for public models like qwen2.5:7b-instruct).
echo "→ pulling $NAME from registry..."
if run_ollama pull "$NAME" 2>/dev/null; then
  echo "✅ $NAME — pulled from registry"
  exit 0
fi

# 3. / 4. Local Modelfile from a previous finetune+convert run.
candidate=""
if [ -f "${MODELS_DIR}/${NAME}/Modelfile" ]; then
  candidate="${MODELS_DIR}/${NAME}/Modelfile"
else
  # Match `models/<name>*-*/Modelfile` (e.g. qwen-agentrag-7b)
  match=$(ls -d "${MODELS_DIR}/${NAME}"*/Modelfile 2>/dev/null | head -n1 || true)
  [ -n "$match" ] && candidate="$match"
fi
if [ -n "$candidate" ]; then
  echo "→ registering $NAME from local Modelfile: $candidate"
  src_dir=$(dirname "$candidate")
  # `ollama create -f -` reads Modelfile from stdin. GGUF FROM paths must be
  # absolute on the daemon's host. Use --files (newer ollama) when possible;
  # otherwise copy the artifact dir into the container with a stable path.
  if docker exec "$CONTAINER" sh -c "test -d /root/.ollama"; then
    docker exec "$CONTAINER" sh -c "mkdir -p /tmp/build_${NAME}"
    tar -C "$src_dir" -c . | docker exec -i "$CONTAINER" tar -C "/tmp/build_${NAME}" -x
    docker exec "$CONTAINER" ollama create "$NAME" -f "/tmp/build_${NAME}/$(basename "$candidate")"
    echo "✅ $NAME — registered from $candidate"
    exit 0
  fi
fi

# 5. Fallback alias.
if [ -n "$FALLBACK" ]; then
  echo "→ no artifact for $NAME; aliasing to base model $FALLBACK"
  # Base must be local before `ollama create` can resolve `FROM <tag>`.
  if ! run_ollama show "$FALLBACK" >/dev/null 2>&1; then
    echo "→ pulling base $FALLBACK first (may take a few minutes)..."
    if ! run_ollama pull "$FALLBACK"; then
      echo "❌ failed to pull base $FALLBACK; cannot alias $NAME" >&2
      exit 1
    fi
  fi
  # Write Modelfile to temp file in container, then reference it.
  # (stdin via docker exec -i has issues with ollama create -f -)
  docker exec "$CONTAINER" bash -c "cat > /tmp/Modelfile.${NAME} <<'MODELEOF'
FROM $FALLBACK
PARAMETER temperature 0.0
PARAMETER num_ctx 8192
MODELEOF
ollama create \"$NAME\" -f /tmp/Modelfile.${NAME}
"
  echo "✅ $NAME — aliased to $FALLBACK"
  exit 0
fi

echo "❌ $NAME — not in registry, no local Modelfile, no fallback specified" >&2
exit 1
