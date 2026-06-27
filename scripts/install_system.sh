#!/usr/bin/env bash
# install_system.sh — system-level deps for AgentRag (Ubuntu/Debian + WSL2).
#
# Installs everything that `make install` assumes is already present:
#   - apt packages (build tools, libreoffice, tesseract, poppler, …)
#   - uv (Python package manager, Astral)
#   - Node.js 20.x (frontend)
#   - Docker Engine + compose plugin (postgres/ES/redis containers)
#   - Ollama (local LLM runtime)
#
# Idempotent: re-running skips already-installed steps.
# GPU drivers are NOT installed — on WSL2 the Windows host provides CUDA;
# on bare-metal Ubuntu install nvidia-driver-* separately before this script.
#
# Usage:
#   bash scripts/install_system.sh           # full install
#   SKIP_DOCKER=1 bash scripts/install_system.sh   # skip docker (already have it)
#   SKIP_OLLAMA=1 bash scripts/install_system.sh   # skip ollama
#   SKIP_NODE=1   bash scripts/install_system.sh   # skip Node.js
#
# After this script: cd to repo, run `make install` then `make dev`.

set -euo pipefail

# ── Pretty logging ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { printf "${CYAN}[install]${NC} %s\n" "$*"; }
ok()   { printf "${GREEN}[ok]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}[warn]${NC} %s\n" "$*"; }
err()  { printf "${RED}[err]${NC} %s\n" "$*" >&2; }

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [[ $EUID -eq 0 ]]; then
  err "Do not run as root. Script uses sudo where needed."
  exit 1
fi
if ! command -v sudo >/dev/null 2>&1; then
  err "sudo not found. Install sudo first or run individual apt commands as root."
  exit 1
fi
if ! command -v apt-get >/dev/null 2>&1; then
  err "Only Debian/Ubuntu (apt) is supported by this script."
  exit 1
fi

ARCH=$(dpkg --print-architecture)
. /etc/os-release
log "Detected: ${PRETTY_NAME} (${ARCH})"

# ── 1. APT system packages ────────────────────────────────────────────────────
log "Updating apt index…"
sudo apt-get update -y

log "Installing system packages…"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  curl \
  git \
  gnupg \
  lsb-release \
  make \
  pkg-config \
  python3 \
  python3-dev \
  python3-venv \
  software-properties-common \
  wget \
  \
  libreoffice \
  libreoffice-impress \
  \
  tesseract-ocr \
  tesseract-ocr-vie \
  tesseract-ocr-eng \
  \
  poppler-utils \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  ffmpeg

ok "APT packages installed."

# ── 2. uv (Astral Python package manager) ─────────────────────────────────────
if command -v uv >/dev/null 2>&1; then
  ok "uv already installed: $(uv --version)"
else
  log "Installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installs to ~/.local/bin; ensure it's on PATH for this shell.
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    err "uv install ran but binary not on PATH. Add \$HOME/.local/bin to PATH and re-run."
    exit 1
  fi
  ok "uv installed: $(uv --version)"
fi

# ── 3. Node.js 20.x ───────────────────────────────────────────────────────────
if [[ "${SKIP_NODE:-0}" == "1" ]]; then
  warn "SKIP_NODE=1 — skipping Node.js install."
elif command -v node >/dev/null 2>&1 && node -v | grep -qE '^v(2[0-9]|[3-9][0-9])\.'; then
  ok "Node.js already installed: $(node -v)"
else
  log "Installing Node.js 20.x via NodeSource…"
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
  ok "Node.js installed: $(node -v)"
fi

# ── 4. Docker Engine + compose plugin ─────────────────────────────────────────
if [[ "${SKIP_DOCKER:-0}" == "1" ]]; then
  warn "SKIP_DOCKER=1 — skipping Docker install."
elif command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  ok "Docker + compose already installed: $(docker --version)"
else
  log "Installing Docker Engine + compose plugin…"
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/${ID}/gpg" \
    | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${ID} ${VERSION_CODENAME} stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

  # Add current user to docker group (avoids sudo on every docker call).
  if ! groups "$USER" | grep -q docker; then
    sudo usermod -aG docker "$USER"
    warn "Added $USER to docker group. Log out + back in (or run \`newgrp docker\`) for it to apply."
  fi
  ok "Docker installed: $(docker --version)"
fi

# ── 5. Ollama (local LLM runtime) ─────────────────────────────────────────────
if [[ "${SKIP_OLLAMA:-0}" == "1" ]]; then
  warn "SKIP_OLLAMA=1 — skipping Ollama install."
elif command -v ollama >/dev/null 2>&1; then
  ok "Ollama already installed: $(ollama --version 2>/dev/null | head -1)"
else
  log "Installing Ollama…"
  curl -fsSL https://ollama.com/install.sh | sh
  ok "Ollama installed: $(ollama --version 2>/dev/null | head -1 || echo 'binary present')"
fi

# ── 6. GPU sanity check (non-fatal) ──────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
  ok "GPU detected: ${GPU_NAME}"
else
  warn "nvidia-smi not found. Vision LLM (Ollama/llava) will run on CPU (very slow)."
  warn "  On WSL2: install NVIDIA driver on Windows host + WSL2 GPU support."
  warn "  On bare-metal: \`sudo apt install nvidia-driver-550\` (or current) then reboot."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo
ok "System bootstrap complete."
echo
echo "Next steps:"
echo "  1. cd $(dirname "$(realpath "$0")")/.."
echo "  2. make install        # docker-up + uv sync + frontend install + migrate"
echo "  3. make docker-up-llm  # pull Ollama models referenced in .env"
echo "  4. make dev            # start API + worker + frontend"
echo
if ! groups "$USER" | grep -q docker; then
  warn "Reminder: log out + back in (or \`newgrp docker\`) before running docker commands."
fi
