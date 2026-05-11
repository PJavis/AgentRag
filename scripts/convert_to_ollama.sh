#!/usr/bin/env bash
# Convert a merged HF model directory to GGUF (Q4_K_M) and register it with
# the local Ollama daemon.
#
# Usage:
#   bash scripts/convert_to_ollama.sh <model_dir> <ollama_name> [quant]
#   bash scripts/convert_to_ollama.sh models/qwen-agentrag-7b qwen-agentrag Q4_K_M
#
# Requires:
#   - llama.cpp checked out at ./vendor/llama.cpp (auto-cloned if missing)
#   - ollama running locally (any host — Ollama uses /usr/share/ollama)

set -euo pipefail

MODEL_DIR="${1:?usage: convert_to_ollama.sh <model_dir> <ollama_name> [quant]}"
OLLAMA_NAME="${2:?usage: convert_to_ollama.sh <model_dir> <ollama_name> [quant]}"
QUANT="${3:-Q4_K_M}"

LLAMACPP_DIR="${LLAMACPP_DIR:-vendor/llama.cpp}"
GGUF_OUT="${MODEL_DIR}/$(basename "${MODEL_DIR}").${QUANT}.gguf"
MODELFILE="${MODEL_DIR}/Modelfile"

if [ ! -d "$MODEL_DIR" ]; then
  echo "❌ model dir not found: $MODEL_DIR" >&2
  exit 1
fi

if [ ! -d "$LLAMACPP_DIR" ]; then
  echo "📥 cloning llama.cpp to $LLAMACPP_DIR ..."
  git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMACPP_DIR"
  pip install -r "$LLAMACPP_DIR/requirements.txt"
fi

echo "🔄 converting HF → GGUF ..."
python "$LLAMACPP_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" \
  --outfile "$GGUF_OUT" \
  --outtype "$(echo "$QUANT" | tr 'A-Z' 'a-z')"

echo "📝 writing Modelfile ..."
cat > "$MODELFILE" <<EOF
FROM ${GGUF_OUT}
PARAMETER temperature 0.0
PARAMETER num_ctx 8192
SYSTEM "AgentRag fine-tuned model. Answer ONLY from provided context. Be precise."
EOF

echo "📦 registering with Ollama as '$OLLAMA_NAME' ..."
ollama create "$OLLAMA_NAME" -f "$MODELFILE"

echo "✅ done. Test with: ollama run $OLLAMA_NAME 'hello'"
echo
echo "Then update .env:"
echo "    EXTRACTION_PROVIDER=ollama"
echo "    EXTRACTION_MODEL=$OLLAMA_NAME"
echo "    AGENT_PROVIDER=ollama"
echo "    AGENT_MODEL=$OLLAMA_NAME"
